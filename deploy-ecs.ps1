# AWS ECS Deployment Script for MLOps Climate Predictor (PowerShell)
# This script deploys the containerized application to AWS ECS Fargate

$ErrorActionPreference = "Stop"

# Configuration
$CLUSTER_NAME = "mlops-climate-cluster"
$SERVICE_NAME = "climate-predictor-service"
$TASK_DEFINITION = "climate-predictor-task"
$ECR_REPOSITORY = "mlops-climate-predictor"
$REGION = "us-east-2"
$AWS_ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)

Write-Host "🚀 Starting AWS ECS Deployment..." -ForegroundColor Green
Write-Host "Region: $REGION" -ForegroundColor Cyan
Write-Host "Account ID: $AWS_ACCOUNT_ID" -ForegroundColor Cyan

# Step 1: Create ECR repository if it doesn't exist
Write-Host "📦 Setting up ECR repository..." -ForegroundColor Yellow
try {
    aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION
} catch {
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $REGION
}

# Step 2: Get ECR login token and login to Docker
Write-Host "🔐 Authenticating with ECR..." -ForegroundColor Yellow
$loginToken = aws ecr get-login-password --region $REGION
$loginToken | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Step 3: Build and tag Docker image
Write-Host "🏗️  Building Docker image..." -ForegroundColor Yellow
docker build -t $ECR_REPOSITORY .
docker tag "${ECR_REPOSITORY}:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:latest"

# Step 4: Push image to ECR
Write-Host "⬆️  Pushing image to ECR..." -ForegroundColor Yellow
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:latest"

# Step 5: Create ECS cluster if it doesn't exist
Write-Host "🏭 Setting up ECS cluster..." -ForegroundColor Yellow
try {
    aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION
} catch {
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --capacity-providers FARGATE --region $REGION
}

# Step 6: Store AWS credentials in Parameter Store (if not already done)
Write-Host "🔑 Setting up AWS credentials in Parameter Store..." -ForegroundColor Yellow
try {
    # Check if parameters exist, if not, create them
    aws ssm get-parameter --name "/mlops/aws-access-key" --region $REGION
} catch {
    Write-Host "Please set your AWS credentials in Parameter Store manually:" -ForegroundColor Red
    Write-Host "aws ssm put-parameter --name '/mlops/aws-access-key' --value 'YOUR_ACCESS_KEY' --type 'SecureString' --region $REGION" -ForegroundColor Yellow
    Write-Host "aws ssm put-parameter --name '/mlops/aws-secret-key' --value 'YOUR_SECRET_KEY' --type 'SecureString' --region $REGION" -ForegroundColor Yellow
}

# Step 7: Register task definition
Write-Host "📋 Registering ECS task definition..." -ForegroundColor Yellow

$taskDefinition = @{
    family = $TASK_DEFINITION
    networkMode = "awsvpc"
    requiresCompatibilities = @("FARGATE")
    cpu = "512"
    memory = "1024"
    executionRoleArn = "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole"
    taskRoleArn = "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskRole"
    containerDefinitions = @(
        @{
            name = "climate-predictor"
            image = "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:latest"
            essential = $true
            portMappings = @(
                @{
                    containerPort = 8080
                    protocol = "tcp"
                }
            )
            environment = @(
                @{
                    name = "AWS_REGION"
                    value = $REGION
                },
                @{
                    name = "AWS_BUCKET"
                    value = "ml-ops-datos-prediccion-clima-uadec22025-ml"
                }
            )
            secrets = @(
                @{
                    name = "AWS_ACCESS_KEY"
                    valueFrom = "arn:aws:ssm:${REGION}:${AWS_ACCOUNT_ID}:parameter/mlops/aws-access-key"
                },
                @{
                    name = "AWS_SECRET_KEY"
                    valueFrom = "arn:aws:ssm:${REGION}:${AWS_ACCOUNT_ID}:parameter/mlops/aws-secret-key"
                }
            )
            logConfiguration = @{
                logDriver = "awslogs"
                options = @{
                    "awslogs-group" = "/ecs/climate-predictor"
                    "awslogs-region" = $REGION
                    "awslogs-stream-prefix" = "ecs"
                }
            }
            healthCheck = @{
                command = @("CMD-SHELL", "python -c 'import models.model; print(`"Health OK`")'")
                interval = 30
                timeout = 5
                retries = 3
                startPeriod = 60
            }
        }
    )
}

$taskDefinitionJson = $taskDefinition | ConvertTo-Json -Depth 10
$taskDefinitionJson | Out-File -FilePath "task-definition.json" -Encoding UTF8

aws ecs register-task-definition --cli-input-json file://task-definition.json --region $REGION

# Step 8: Create CloudWatch log group
Write-Host "📊 Setting up CloudWatch logging..." -ForegroundColor Yellow
try {
    aws logs create-log-group --log-group-name "/ecs/climate-predictor" --region $REGION
} catch {
    Write-Host "Log group already exists" -ForegroundColor Cyan
}

Write-Host "✅ Task definition registered successfully!" -ForegroundColor Green
Write-Host "🔍 Next steps:" -ForegroundColor Cyan
Write-Host "1. Create or update your VPC, subnets, and security groups" -ForegroundColor Yellow
Write-Host "2. Create the ECS service with proper network configuration" -ForegroundColor Yellow
Write-Host "3. Monitor your service at: https://console.aws.amazon.com/ecs/home?region=$REGION#/clusters/$CLUSTER_NAME/services" -ForegroundColor Yellow