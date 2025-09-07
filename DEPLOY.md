# Configuración de Deploy para AWS Lambda

## 🚀 Deploy con AWS CLI

### 1. Crear función Lambda
```bash
# Crear package de deployment
pip install -r requirements.txt -t package/
cp evaluate.py package/
cd package && zip -r ../lambda_function.zip .

# Crear función
aws lambda create-function \
  --function-name model-evaluator-clima-ski \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-mlops-role \
  --handler evaluate.lambda_handler \
  --zip-file fileb://../lambda_function.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables='{PYTHONPATH=/var/runtime}' \
  --tags Project=MLOps-ClimaSkI,Environment=Production
```

### 2. Actualizar función existente
```bash
aws lambda update-function-code \
  --function-name model-evaluator-clima-ski \
  --zip-file fileb://lambda_function.zip
```

## 🐳 Deploy con Docker (Container Image)

### 1. Build y push imagen
```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Crear repositorio ECR (si no existe)
aws ecr create-repository --repository-name model-evaluator-clima

# Build imagen
docker build -t model-evaluator-clima .

# Tag imagen
docker tag model-evaluator-clima:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/model-evaluator-clima:latest

# Push imagen
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/model-evaluator-clima:latest
```

### 2. Crear función Lambda con imagen
```bash
aws lambda create-function \
  --function-name model-evaluator-clima-ski \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-mlops-role \
  --code ImageUri=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/model-evaluator-clima:latest \
  --timeout 300 \
  --memory-size 512 \
  --environment Variables='{PYTHONPATH=/var/runtime}'
```

## ⚙️ Configuración de Triggers

### EventBridge (CloudWatch Events)
```bash
# Crear regla para evaluación diaria
aws events put-rule \
  --name daily-model-evaluation \
  --schedule-expression "cron(0 6 * * ? *)" \
  --description "Evaluación diaria del modelo climático"

# Agregar target Lambda
aws events put-targets \
  --rule daily-model-evaluation \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:ACCOUNT_ID:function:model-evaluator-clima-ski","Input"='{"model_path":"s3://bucket/model.pkl","data_path":"s3://bucket/daily_data.csv","target_col":"temperature"}'

# Dar permisos a EventBridge
aws lambda add-permission \
  --function-name model-evaluator-clima-ski \
  --statement-id allow-eventbridge \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/daily-model-evaluation
```

### S3 Event Notification
```bash
# Configurar notificación S3 (JSON)
{
  "LambdaConfiguration": {
    "Id": "NewDataEvaluation",
    "LambdaFunctionArn": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:model-evaluator-clima-ski",
    "Events": ["s3:ObjectCreated:*"],
    "Filter": {
      "Key": {
        "FilterRules": [
          {
            "Name": "prefix",
            "Value": "datasets/new/"
          },
          {
            "Name": "suffix", 
            "Value": ".csv"
          }
        ]
      }
    }
  }
}

# Aplicar configuración
aws s3api put-bucket-notification-configuration \
  --bucket mi-bucket-clima \
  --notification-configuration file://s3-notification.json

# Dar permisos a S3
aws lambda add-permission \
  --function-name model-evaluator-clima-ski \
  --statement-id allow-s3-invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::mi-bucket-clima
```

## 🔐 Configuración IAM

### 1. Crear rol para Lambda
```bash
# Trust policy (trust-policy.json)
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

# Crear rol
aws iam create-role \
  --role-name lambda-mlops-role \
  --assume-role-policy-document file://trust-policy.json

# Attachar políticas
aws iam attach-role-policy \
  --role-name lambda-mlops-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
  --role-name lambda-mlops-role \
  --policy-name S3ModelAccess \
  --policy-document file://aws/iam_policy.json
```

## 🧪 Testing Lambda

### 1. Test local con SAM
```bash
# Instalar SAM CLI
# https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html

# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ModelEvaluatorFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: evaluate.lambda_handler
      Runtime: python3.11
      Timeout: 300
      MemorySize: 512

# Test local
sam local invoke ModelEvaluatorFunction -e examples/lambda_event.json
```

### 2. Test en AWS
```bash
# Invoke función
aws lambda invoke \
  --function-name model-evaluator-clima-ski \
  --payload file://examples/lambda_event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

# Ver respuesta
cat response.json
```

## 📊 Monitoreo CloudWatch

### 1. Crear dashboard
```bash
# dashboard.json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "model-evaluator-clima-ski"],
          [".", "Errors", ".", "."],
          [".", "Invocations", ".", "."]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Model Evaluator Metrics"
      }
    }
  ]
}

# Crear dashboard
aws cloudwatch put-dashboard \
  --dashboard-name MLOps-ModelEvaluation \
  --dashboard-body file://dashboard.json
```

### 2. Crear alarmas
```bash
# Alarma por errores
aws cloudwatch put-metric-alarm \
  --alarm-name ModelEvaluator-Errors \
  --alarm-description "Errores en evaluador de modelo" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --dimensions Name=FunctionName,Value=model-evaluator-clima-ski \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:mlops-alerts

# Alarma por duración
aws cloudwatch put-metric-alarm \
  --alarm-name ModelEvaluator-Duration \
  --alarm-description "Duración alta en evaluador" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --threshold 240000 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=FunctionName,Value=model-evaluator-clima-ski \
  --evaluation-periods 2
```

## 🔄 CI/CD con GitHub Actions

El workflow ya está configurado en `.github/workflows/model_evaluation.yml`

### Variables de entorno requeridas:
- `AWS_ROLE_ARN`: ARN del rol IAM para OIDC
- `TEAMS_WEBHOOK_URL`: Webhook para notificaciones
- `RETRAIN_WEBHOOK_URL`: Webhook para activar reentrenamiento

### Secrets necesarios:
```bash
# En GitHub repository settings > Secrets
AWS_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/github-actions-role
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
RETRAIN_WEBHOOK_URL=https://api.example.com/retrain/trigger
```

## 📋 Checklist de Deploy

- [ ] Crear bucket S3 para modelos y datos
- [ ] Configurar credenciales AWS (IAM role/user)
- [ ] Crear función Lambda
- [ ] Configurar triggers (EventBridge/S3)
- [ ] Configurar monitoreo CloudWatch
- [ ] Configurar notificaciones SNS/Teams
- [ ] Probar evaluación end-to-end
- [ ] Configurar GitHub Actions
- [ ] Documentar proceso operativo
