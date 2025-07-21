#!/usr/bin/env python3
"""
Real-time Audio Transcription and Diarization Service
- Transcribes audio in real-time using faster-whisper.
- Performs speaker diarization on the complete audio using pyannote.audio.
"""

import sys
import json
import time
import os
from datetime import datetime
import logging

from flask import Flask, request, jsonify
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Helper Classes & Functions ---

class DiarizationAlignment:
    """Aligns Whisper transcription with pyannote.audio diarization."""
    def __init__(self, transcription_segments, diarization_result):
        self.segments = transcription_segments
        self.diarization = diarization_result

    def align(self):
        """
        Aligns segments and returns speaker-labeled transcription.
        Returns:
            list: A list of segments, each with a 'speaker' field.
        """
        aligned_segments = []
        for segment in self.segments:
            speaker = self.get_speaker_for_segment(segment['start'], segment['end'])
            segment_with_speaker = {**segment, 'speaker': speaker}
            aligned_segments.append(segment_with_speaker)
        return aligned_segments
    
    def get_speaker_for_segment(self, start_time, end_time):
        """Finds the speaker for a given time segment by checking overlap."""
        best_speaker = "UNKNOWN"
        best_overlap = 0
        
        for turn, _, speaker in self.diarization.itertracks(yield_label=True):
            # Calculate overlap between transcription segment and speaker segment
            overlap_start = max(start_time, turn.start)
            overlap_end = min(end_time, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # If there's any overlap, calculate the percentage
            if overlap_duration > 0:
                segment_duration = end_time - start_time
                overlap_percentage = overlap_duration / segment_duration
                
                # Use the speaker with the highest overlap percentage
                if overlap_percentage > best_overlap:
                    best_speaker = speaker
                    best_overlap = overlap_percentage
        
        return best_speaker



# --- Main Service Class ---

class RealtimeTranscriber:
    """Handles audio processing, including transcription and diarization."""

    def __init__(self, model_size=MODEL_SIZE):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        if self.device == "cpu":
            logger.warning("⚠️ GPU not available, falling back to CPU. Performance will be significantly slower.")

        self.whisper_model = None
        self.diarization_pipeline = None
        self.chunk_counter = 0
        
        self._load_models()

    def _validate_gpu(self):
        """Ensures a CUDA-enabled GPU is available."""
        if not torch.cuda.is_available():
            logger.error("❌ GPU not available. This service requires a CUDA-enabled GPU.")
            sys.exit(1)
        logger.info("✅ GPU validation successful.")

    def _load_models(self):
        """Loads both Whisper and pyannote.audio models onto the GPU."""
        try:
            # Load Whisper model
            logger.info(f"Loading Whisper model: {self.model_size} with {self.compute_type} on {self.device}")
            self.whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("✅ Whisper model loaded.")

            # Load Diarization pipeline
            if not HF_TOKEN:
                logger.warning("⚠️ HF_TOKEN not set. Diarization will be disabled.")
                return

            logger.info("Loading diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            
            if self.diarization_pipeline is None:
                raise RuntimeError(
                    "Failed to load diarization pipeline. Please:\n"
                    "1. Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept user conditions\n"
                    "2. Visit https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM and accept user conditions\n"
                    "3. Visit https://hf.co/pyannote/segmentation-3.0 and accept user conditions\n"
                    "4. Ensure your HF_TOKEN has the correct permissions"
                )
            
            self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
            logger.info("✅ Diarization pipeline loaded.")

        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise

    def transcribe_chunk(self, audio_data):
        """Transcribes a small chunk of audio for real-time feedback."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.chunk_counter += 1
        logger.info(f"🎯 Processing real-time chunk #{self.chunk_counter}")

        segments, _ = self.whisper_model.transcribe(
            audio_np, beam_size=1, word_timestamps=True, language="en"
        )
        
        segment_list = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]
        return segment_list

    def diarize_full_audio(self, audio_data):
        """Transcribes and diarizes the full audio recording."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        logger.info(f"🎤 Processing full audio for diarization ({len(audio_np)} samples)...")

        # 1. Transcription
        segments, info = self.whisper_model.transcribe(
            audio_np, beam_size=5, word_timestamps=True, language="en"
        )
        transcription_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        
        if not self.diarization_pipeline:
            logger.warning("Diarization pipeline not loaded. Returning transcription only.")
            return {"status": "success", "segments": transcription_segments, "full_text": "".join(s['text'] for s in transcription_segments)}

        # 2. Diarization
        diarization_input = {"waveform": torch.from_numpy(audio_np).unsqueeze(0), "sample_rate": 16000}
        diarization = self.diarization_pipeline(diarization_input)
        
        # Debug: Log detected speakers
        detected_speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            detected_speakers.add(speaker)
            logger.info(f"🗣️ Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
        logger.info(f"🔍 Total speakers detected: {len(detected_speakers)} - {list(detected_speakers)}")
        
        # 3. Alignment
        alignment = DiarizationAlignment(transcription_segments, diarization)
        aligned_result = alignment.align()
        
        # Debug: Log speaker assignments and alignment details
        speaker_assignments = {}
        for i, segment in enumerate(aligned_result):
            speaker = segment.get('speaker', 'UNKNOWN')
            if speaker not in speaker_assignments:
                speaker_assignments[speaker] = 0
            speaker_assignments[speaker] += 1
            logger.info(f"📝 Segment {i+1}: {segment['start']:.2f}s-{segment['end']:.2f}s → {speaker}: \"{segment['text'][:50]}...\"")
        logger.info(f"💬 Final speaker assignments: {speaker_assignments}")

        # Keep the full text clean, without speaker labels
        full_text_plain = " ".join(s['text'].strip() for s in aligned_result)
        
        # The frontend expects the diarized segments in 'segments_with_speakers'
        return {"status": "success", "segments_with_speakers": aligned_result, "full_text": full_text_plain}

# --- Flask App ---
app = Flask(__name__)
transcriber = None

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    """Handles real-time transcription of small audio chunks."""
    if not request.data:
        return jsonify({"error": "No audio data in request"}), 400
    
    try:
        segments = transcriber.transcribe_chunk(request.data)
        return jsonify({"status": "success", "segments": segments, "full_text": " ".join(s['text'] for s in segments)})
    except Exception as e:
        logger.error(f"API Error (transcribe): {e}", exc_info=True)
        return jsonify({"error": "Server error during real-time transcription"}), 500

@app.route('/diarize', methods=['POST'])
def diarize_endpoint():
    """Handles final transcription and diarization of the full recording."""
    if not request.data:
        return jsonify({"error": "No audio data in request"}), 400

    try:
        result = transcriber.diarize_full_audio(request.data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"API Error (diarize): {e}", exc_info=True)
        return jsonify({"error": "Server error during diarization"}), 500

def main():
    global transcriber
    
    try:
        transcriber = RealtimeTranscriber(model_size=MODEL_SIZE)
        logger.info("🚀 Starting Flask server on port 8001...")
        app.run(host="0.0.0.0", port=8001)
    except Exception as e:
        logger.error(f"❌ Failed to start the transcription service: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 