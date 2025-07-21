# Cymbiose TTS Deployment Script (PowerShell)

Write-Host "🚀 Cymbiose TTS Deployment Script" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker first." -ForegroundColor Red
    exit 1
}

# Check if HF_TOKEN is provided
if (-not $env:HF_TOKEN) {
    Write-Host "⚠️  HF_TOKEN environment variable not set." -ForegroundColor Yellow
    Write-Host "   Speaker diarization will be disabled." -ForegroundColor Yellow
    Write-Host "   Set `$env:HF_TOKEN='your_token' to enable diarization." -ForegroundColor Yellow
    Write-Host ""
}

# Build the Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Blue
docker build -t cymbiose-tts .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
Write-Host ""

# Check for NVIDIA Docker support
$gpuSupport = $false
try {
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi *>$null
    $gpuSupport = $true
    Write-Host "🎮 NVIDIA GPU support detected" -ForegroundColor Green
    $gpuFlag = "--gpus all"
} catch {
    Write-Host "💻 No GPU support detected, using CPU mode" -ForegroundColor Yellow
    $gpuFlag = "-e CUDA_VISIBLE_DEVICES="
}

# Run the container
Write-Host "🚀 Starting container..." -ForegroundColor Blue

$dockerArgs = @(
    "run", "-d",
    "--name", "cymbiose-tts",
    "-p", "8001:8001",
    "-e", "HF_TOKEN=$env:HF_TOKEN"
)

if ($gpuSupport) {
    $dockerArgs += "--gpus", "all"
} else {
    $dockerArgs += "-e", "CUDA_VISIBLE_DEVICES="
}

$dockerArgs += "cymbiose-tts"

& docker @dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📡 Server running at: http://localhost:8001" -ForegroundColor Cyan
    Write-Host "🧪 Test with: Invoke-RestMethod -Uri http://localhost:8001/transcribe -Method Post -ContentType 'application/octet-stream' -InFile audio.raw" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 Container management:" -ForegroundColor White
    Write-Host "   View logs: docker logs cymbiose-tts" -ForegroundColor Gray
    Write-Host "   Stop:      docker stop cymbiose-tts" -ForegroundColor Gray
    Write-Host "   Remove:    docker rm cymbiose-tts" -ForegroundColor Gray
} else {
    Write-Host "❌ Failed to start container!" -ForegroundColor Red
    exit 1
} 