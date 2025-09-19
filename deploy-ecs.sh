#!/bin/bash

# AWS ECS Deployment Script for MLOps Climate Predictor
# This script deploys the containerized application to AWS ECS Fargate

set -e

# Configuration
CLUSTER_NAME="mlops-climate-cluster"
SERVICE_NAME="climate-predictor-service"
TASK_DEFINITION="climate-predictor-task"
ECR_REPOSITORY="mlops-climate-predictor"
REGION="us-east-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Starting AWS ECS Deployment..."
echo "Region: $REGION"
echo "Account ID: $AWS_ACCOUNT_ID"

# Step 1: Create ECR repository if it doesn't exist
echo "📦 Setting up ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $REGION || \
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $REGION

# Step 2: Get ECR login token and login to Docker
echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Step 3: Build and tag Docker image
echo "🏗️  Building Docker image..."
docker build -t $ECR_REPOSITORY .
docker tag $ECR_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Step 4: Push image to ECR
echo "⬆️  Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Step 5: Create ECS cluster if it doesn't exist
echo "🏭 Setting up ECS cluster..."
aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION || \
aws ecs create-cluster --cluster-name $CLUSTER_NAME --capacity-providers FARGATE --region $REGION

# Step 6: Register task definition
echo "📋 Registering ECS task definition..."
cat > task-definition.json << EOF
{
    "family": "$TASK_DEFINITION",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "climate-predictor",
            "image": "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8080,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": "$REGION"
                },
                {
                    "name": "AWS_BUCKET",
                    "value": "ml-ops-datos-prediccion-clima-uadec22025-ml"
                }
            ],
            "secrets": [
                {
                    "name": "AWS_ACCESS_KEY",
                    "valueFrom": "arn:aws:ssm:$REGION:$AWS_ACCOUNT_ID:parameter/mlops/aws-access-key"
                },
                {
                    "name": "AWS_SECRET_KEY",
                    "valueFrom": "arn:aws:ssm:$REGION:$AWS_ACCOUNT_ID:parameter/mlops/aws-secret-key"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/climate-predictor",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "python -c 'import models.model; print(\"Health OK\")'"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json --region $REGION

# Step 7: Create CloudWatch log group
echo "📊 Setting up CloudWatch logging..."
aws logs create-log-group --log-group-name /ecs/climate-predictor --region $REGION || echo "Log group already exists"

# Step 8: Create or update ECS service
echo "🔄 Creating/updating ECS service..."
aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION || \
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_DEFINITION \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}" \
    --region $REGION

echo "✅ Deployment completed successfully!"
echo "🔍 Monitor your service at: https://console.aws.amazon.com/ecs/home?region=$REGION#/clusters/$CLUSTER_NAME/services"