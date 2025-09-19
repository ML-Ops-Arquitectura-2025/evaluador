# Complete AWS Infrastructure and Application Deployment Script

param(
    [Parameter(Mandatory=$false)]
    [string]$StackName = "mlops-climate-predictor",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-2",
    
    [Parameter(Mandatory=$false)]
    [switch]$DeployInfrastructure = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$DeployApplication = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SetupCredentials = $false
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 MLOps Climate Predictor - Complete AWS Deployment" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

# Function to check if AWS CLI is configured
function Test-AWSConfiguration {
    try {
        $identity = aws sts get-caller-identity --output json | ConvertFrom-Json
        Write-Host "✅ AWS CLI configured - Account: $($identity.Account)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ AWS CLI not configured properly" -ForegroundColor Red
        return $false
    }
}

# Function to setup AWS credentials in Parameter Store
function Set-AWSCredentials {
    Write-Host "🔑 Setting up AWS credentials in Parameter Store..." -ForegroundColor Yellow
    
    # Get credentials from .env file
    if (Test-Path ".env") {
        $envContent = Get-Content ".env"
        $accessKey = ($envContent | Where-Object { $_ -match "AWS_ACCESS_KEY=" }).Split("=")[1]
        $secretKey = ($envContent | Where-Object { $_ -match "AWS_SECRET_KEY=" }).Split("=")[1]
        
        if ($accessKey -and $secretKey) {
            # Store in Parameter Store
            try {
                aws ssm put-parameter --name "/mlops/aws-access-key" --value $accessKey --type "SecureString" --region $Region --overwrite
                aws ssm put-parameter --name "/mlops/aws-secret-key" --value $secretKey --type "SecureString" --region $Region --overwrite
                Write-Host "✅ Credentials stored in Parameter Store" -ForegroundColor Green
            } catch {
                Write-Host "❌ Failed to store credentials in Parameter Store" -ForegroundColor Red
                throw
            }
        } else {
            Write-Host "❌ Could not find AWS credentials in .env file" -ForegroundColor Red
            throw "Missing AWS credentials"
        }
    } else {
        Write-Host "❌ .env file not found" -ForegroundColor Red
        throw ".env file required"
    }
}

# Function to deploy CloudFormation infrastructure
function Deploy-Infrastructure {
    Write-Host "🏗️  Deploying AWS infrastructure..." -ForegroundColor Yellow
    
    # Check if stack exists
    try {
        aws cloudformation describe-stacks --stack-name $StackName --region $Region | Out-Null
        $stackExists = $true
        Write-Host "📋 Stack exists - updating..." -ForegroundColor Cyan
    } catch {
        $stackExists = $false
        Write-Host "📋 Creating new stack..." -ForegroundColor Cyan
    }
    
    # Deploy or update stack
    try {
        if ($stackExists) {
            aws cloudformation update-stack `
                --stack-name $StackName `
                --template-body file://infrastructure.yaml `
                --capabilities CAPABILITY_NAMED_IAM `
                --region $Region
        } else {
            aws cloudformation create-stack `
                --stack-name $StackName `
                --template-body file://infrastructure.yaml `
                --capabilities CAPABILITY_NAMED_IAM `
                --region $Region
        }
        
        Write-Host "⏳ Waiting for stack operation to complete..." -ForegroundColor Yellow
        aws cloudformation wait stack-create-complete --stack-name $StackName --region $Region
        aws cloudformation wait stack-update-complete --stack-name $StackName --region $Region
        
        Write-Host "✅ Infrastructure deployed successfully!" -ForegroundColor Green
        
        # Get stack outputs
        $outputs = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output json | ConvertFrom-Json
        
        Write-Host "`n📋 Infrastructure Details:" -ForegroundColor Cyan
        foreach ($output in $outputs) {
            Write-Host "  $($output.OutputKey): $($output.OutputValue)" -ForegroundColor Yellow
        }
        
        return $outputs
        
    } catch {
        Write-Host "❌ Infrastructure deployment failed!" -ForegroundColor Red
        throw
    }
}

# Function to deploy application to ECS
function Deploy-Application {
    param($InfrastructureOutputs)
    
    Write-Host "🚢 Deploying application to ECS..." -ForegroundColor Yellow
    
    # Get ECR repository URI from outputs
    $ecrUri = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "ECRRepositoryURI" }).OutputValue
    $clusterName = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "ECSClusterName" }).OutputValue
    $subnet1 = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "PublicSubnet1Id" }).OutputValue
    $subnet2 = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "PublicSubnet2Id" }).OutputValue
    $securityGroup = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "ECSSecurityGroupId" }).OutputValue
    $targetGroupArn = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "TargetGroupArn" }).OutputValue
    $taskExecutionRoleArn = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "ECSTaskExecutionRoleArn" }).OutputValue
    $taskRoleArn = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "ECSTaskRoleArn" }).OutputValue
    
    # Login to ECR
    Write-Host "🔐 Authenticating with ECR..." -ForegroundColor Yellow
    $loginToken = aws ecr get-login-password --region $Region
    $loginToken | docker login --username AWS --password-stdin $ecrUri.Split("/")[0]
    
    # Build and push Docker image
    Write-Host "🏗️  Building and pushing Docker image..." -ForegroundColor Yellow
    docker build -t mlops-climate-predictor .
    docker tag mlops-climate-predictor:latest $ecrUri:latest
    docker push $ecrUri:latest
    
    # Create task definition
    Write-Host "📋 Creating ECS task definition..." -ForegroundColor Yellow
    
    $taskDefinition = @{
        family = "climate-predictor-task"
        networkMode = "awsvpc"
        requiresCompatibilities = @("FARGATE")
        cpu = "512"
        memory = "1024"
        executionRoleArn = $taskExecutionRoleArn
        taskRoleArn = $taskRoleArn
        containerDefinitions = @(
            @{
                name = "climate-predictor"
                image = "$ecrUri:latest"
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
                        value = $Region
                    },
                    @{
                        name = "AWS_BUCKET"
                        value = "ml-ops-datos-prediccion-clima-uadec22025-ml"
                    }
                )
                secrets = @(
                    @{
                        name = "AWS_ACCESS_KEY"
                        valueFrom = "arn:aws:ssm:${Region}:$(aws sts get-caller-identity --query Account --output text):parameter/mlops/aws-access-key"
                    },
                    @{
                        name = "AWS_SECRET_KEY"
                        valueFrom = "arn:aws:ssm:${Region}:$(aws sts get-caller-identity --query Account --output text):parameter/mlops/aws-secret-key"
                    }
                )
                logConfiguration = @{
                    logDriver = "awslogs"
                    options = @{
                        "awslogs-group" = "/ecs/mlops-climate-predictor"
                        "awslogs-region" = $Region
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
    
    # Register task definition
    $taskDefResult = aws ecs register-task-definition --cli-input-json file://task-definition.json --region $Region | ConvertFrom-Json
    $taskDefArn = $taskDefResult.taskDefinition.taskDefinitionArn
    
    # Create or update ECS service
    Write-Host "🔄 Creating/updating ECS service..." -ForegroundColor Yellow
    
    try {
        # Check if service exists
        aws ecs describe-services --cluster $clusterName --services "climate-predictor-service" --region $Region | Out-Null
        
        # Update existing service
        aws ecs update-service `
            --cluster $clusterName `
            --service "climate-predictor-service" `
            --task-definition $taskDefArn `
            --region $Region
            
        Write-Host "🔄 Service updated" -ForegroundColor Cyan
        
    } catch {
        # Create new service
        aws ecs create-service `
            --cluster $clusterName `
            --service-name "climate-predictor-service" `
            --task-definition $taskDefArn `
            --desired-count 1 `
            --launch-type FARGATE `
            --load-balancers "targetGroupArn=$targetGroupArn,containerName=climate-predictor,containerPort=8080" `
            --network-configuration "awsvpcConfiguration={subnets=[$subnet1,$subnet2],securityGroups=[$securityGroup],assignPublicIp=ENABLED}" `
            --region $Region
            
        Write-Host "🆕 Service created" -ForegroundColor Cyan
    }
    
    Write-Host "✅ Application deployed successfully!" -ForegroundColor Green
    
    # Get load balancer URL
    $albUrl = ($InfrastructureOutputs | Where-Object { $_.OutputKey -eq "LoadBalancerURL" }).OutputValue
    Write-Host "🌐 Application URL: $albUrl" -ForegroundColor Green
}

# Main execution
try {
    # Check AWS configuration
    if (-not (Test-AWSConfiguration)) {
        throw "AWS CLI not properly configured"
    }
    
    # Setup credentials if requested
    if ($SetupCredentials) {
        Set-AWSCredentials
    }
    
    # Deploy infrastructure if requested
    $infrastructureOutputs = $null
    if ($DeployInfrastructure) {
        $infrastructureOutputs = Deploy-Infrastructure
    } else {
        # Get existing stack outputs
        try {
            $outputs = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output json | ConvertFrom-Json
            $infrastructureOutputs = $outputs
        } catch {
            Write-Host "⚠️  Infrastructure stack not found. Run with -DeployInfrastructure first." -ForegroundColor Yellow
        }
    }
    
    # Deploy application if requested
    if ($DeployApplication -and $infrastructureOutputs) {
        Deploy-Application $infrastructureOutputs
    }
    
    if (-not $DeployInfrastructure -and -not $DeployApplication -and -not $SetupCredentials) {
        Write-Host "ℹ️  Usage examples:" -ForegroundColor Cyan
        Write-Host "  Setup credentials:     .\deploy-aws.ps1 -SetupCredentials" -ForegroundColor Yellow
        Write-Host "  Deploy infrastructure: .\deploy-aws.ps1 -DeployInfrastructure" -ForegroundColor Yellow
        Write-Host "  Deploy application:    .\deploy-aws.ps1 -DeployApplication" -ForegroundColor Yellow
        Write-Host "  Deploy everything:     .\deploy-aws.ps1 -SetupCredentials -DeployInfrastructure -DeployApplication" -ForegroundColor Yellow
    }
    
    Write-Host "`n✅ Deployment completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}