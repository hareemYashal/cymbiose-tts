#!/bin/bash

# Cymbiose TTS Deployment Script

set -e

echo "🚀 Cymbiose TTS Deployment Script"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN environment variable not set."
    echo "   Speaker diarization will be disabled."
    echo "   Set HF_TOKEN=your_token to enable diarization."
    echo
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t cymbiose-tts .

echo "✅ Docker image built successfully!"
echo

# Check for NVIDIA Docker support
if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU support detected"
    GPU_FLAG="--gpus all"
else
    echo "💻 No GPU support detected, using CPU mode"
    GPU_FLAG="-e CUDA_VISIBLE_DEVICES="
fi

# Run the container
echo "🚀 Starting container..."
docker run -d \
    --name cymbiose-tts \
    -p 8001:8001 \
    -e HF_TOKEN="$HF_TOKEN" \
    $GPU_FLAG \
    cymbiose-tts

echo "✅ Container started successfully!"
echo
echo "📡 Server running at: http://localhost:8001"
echo "🧪 Test with: curl -X POST http://localhost:8001/transcribe -H 'Content-Type: application/octet-stream' --data-binary @audio.raw"
echo
echo "📋 Container management:"
echo "   View logs: docker logs cymbiose-tts"
echo "   Stop:      docker stop cymbiose-tts"
echo "   Remove:    docker rm cymbiose-tts" 