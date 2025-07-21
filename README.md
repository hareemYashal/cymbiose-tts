# Cymbiose TTS - Real-time Transcription Server

A high-performance Python Flask server for real-time speech transcription and speaker diarization using Whisper and pyannote.audio.

## Features

- **Real-time Audio Processing**: Stream audio chunks for live transcription
- **Speaker Diarization**: Identify different speakers in conversations (requires HF_TOKEN)
- **GPU Acceleration**: CUDA support for fast processing
- **REST API**: Simple HTTP endpoints for audio processing
- **Docker Support**: Easy deployment with all dependencies included

## API Endpoints

### Real-time Transcription
```
POST /transcribe
Content-Type: application/octet-stream
Body: Raw PCM audio data (16kHz, 16-bit, mono)
```

### Full Audio Diarization
```
POST /diarize
Content-Type: application/octet-stream
Body: Raw PCM audio data (16kHz, 16-bit, mono)
```

## Quick Start with Docker

### Prerequisites
- Docker with NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA support (optional, will fallback to CPU)
- Hugging Face token (for speaker diarization)

### 1. Build the Docker Image
```bash
docker build -t cymbiose-tts .
```

### 2. Run the Container
```bash
# With GPU support (recommended)
docker run --gpus all -p 8001:8001 -e HF_TOKEN="your_hf_token_here" cymbiose-tts

# CPU only (if no GPU available)
docker run -p 8001:8001 -e HF_TOKEN="your_hf_token_here" -e CUDA_VISIBLE_DEVICES="" cymbiose-tts
```

### 3. Test the Server
```bash
curl -X POST http://localhost:8001/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@audio_file.raw"
```

## AWS Deployment

### Recommended Instance Types
- **Cost-effective**: `g4dn.xlarge` (NVIDIA T4, 16GB VRAM) - ~$0.53/hour
- **High-performance**: `g4dn.2xlarge` (NVIDIA T4, 16GB VRAM) - ~$0.75/hour
- **Production**: `p3.2xlarge` (NVIDIA V100, 16GB VRAM) - ~$3.06/hour

### Deployment Steps
1. Launch EC2 instance with Deep Learning AMI
2. Install Docker and NVIDIA Container Toolkit
3. Pull or build your Docker image
4. Run with appropriate security groups (port 8001)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SIZE` | `tiny` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `COMPUTE_TYPE` | `float16` | Computation precision (`float16`, `float32`) |
| `HF_TOKEN` | None | Hugging Face token for diarization models |
| `CUDA_VISIBLE_DEVICES` | All | GPU devices to use (set to "" for CPU-only) |

## Model Performance

| Model Size | Speed | Accuracy | VRAM Usage |
|------------|-------|----------|------------|
| `tiny` | Fastest | Good | ~1GB |
| `base` | Fast | Better | ~2GB |
| `small` | Medium | Very Good | ~3GB |
| `medium` | Slow | Excellent | ~5GB |

## Local Development

### Prerequisites
- Python 3.10+
- CUDA 12.0+ (optional)
- FFmpeg

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_token_here"
export MODEL_SIZE="tiny"

# Run the server
python realtime_transcriber.py
```

## License

This project is licensed under the MIT License. 