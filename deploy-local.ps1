# Local Docker Deployment Script for MLOps Climate Predictor

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Local Docker Deployment..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found. Creating template..." -ForegroundColor Red
    $envTemplate = @"
# AWS S3 Configuration
AWS_ACCESS_KEY=your_aws_access_key_here
AWS_SECRET_KEY=your_aws_secret_key_here
AWS_BUCKET=ml-ops-datos-prediccion-clima-uadec22025-ml
AWS_REGION=us-east-2
"@
    $envTemplate | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "📝 Please edit .env file with your AWS credentials before running again." -ForegroundColor Yellow
    exit 1
}

# Build the Docker image
Write-Host "🏗️  Building Docker image..." -ForegroundColor Yellow
docker build -t mlops-climate-predictor .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

# Stop and remove existing container if running
Write-Host "🔄 Stopping existing container..." -ForegroundColor Yellow
docker stop climate-mlops-system 2>$null
docker rm climate-mlops-system 2>$null

# Run the container
Write-Host "🚀 Starting MLOps Climate Predictor container..." -ForegroundColor Yellow
docker run -d `
    --name climate-mlops-system `
    --env-file .env `
    -v "${PWD}/logs:/app/logs" `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/models/saved:/app/models/saved" `
    -v "${PWD}/models/metrics_history.json:/app/models/metrics_history.json" `
    -p 8080:8080 `
    --restart unless-stopped `
    mlops-climate-predictor

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container started successfully!" -ForegroundColor Green
    Write-Host "📊 Container status:" -ForegroundColor Cyan
    docker ps --filter "name=climate-mlops-system"
    
    Write-Host "`n🔍 Useful commands:" -ForegroundColor Cyan
    Write-Host "View logs: docker logs climate-mlops-system -f" -ForegroundColor Yellow
    Write-Host "Stop container: docker stop climate-mlops-system" -ForegroundColor Yellow
    Write-Host "Restart container: docker restart climate-mlops-system" -ForegroundColor Yellow
    Write-Host "Remove container: docker rm -f climate-mlops-system" -ForegroundColor Yellow
    
    Write-Host "`n📈 Monitor the application:" -ForegroundColor Cyan
    Write-Host "Check container health: docker inspect climate-mlops-system --format='{{.State.Health.Status}}'" -ForegroundColor Yellow
    Write-Host "Access logs directory: ./logs/" -ForegroundColor Yellow
    
    # Wait a moment and check if container is still running
    Start-Sleep -Seconds 5
    $containerStatus = docker inspect climate-mlops-system --format='{{.State.Status}}'
    if ($containerStatus -eq "running") {
        Write-Host "✅ Container is running successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Container may have issues. Check logs:" -ForegroundColor Yellow
        docker logs climate-mlops-system --tail 20
    }
    
} else {
    Write-Host "❌ Failed to start container!" -ForegroundColor Red
    exit 1
}