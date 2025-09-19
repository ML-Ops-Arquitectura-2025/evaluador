# MLOps Climate Predictor - Deployment Guide

## 🌟 Overview

Este proyecto es un sistema MLOps completo para predicción climática que incluye:
- Modelo de ML para predicción de temperatura y precipitación
- Monitoreo automático de archivos
- Reentrenamiento automático basado en métricas
- Integración con AWS S3 para datos reales
- Containerización con Docker
- Deployment en AWS ECS Fargate

## 🚀 Opciones de Deployment

### 1. Deployment Local con Docker

#### Prerrequisitos
- Docker Desktop instalado y ejecutándose
- Archivo `.env` configurado con credenciales AWS

#### Pasos

1. **Configurar credenciales:**
   ```powershell
   # El archivo .env ya existe, solo edita las credenciales si es necesario
   # .env contiene:
   # AWS_ACCESS_KEY=tu_access_key
   # AWS_SECRET_KEY=tu_secret_key
   # AWS_BUCKET=ml-ops-datos-prediccion-clima-uadec22025-ml
   # AWS_REGION=us-east-2
   ```

2. **Ejecutar deployment local:**
   ```powershell
   .\deploy-local.ps1
   ```

3. **Monitorear la aplicación:**
   ```powershell
   # Ver logs en tiempo real
   docker logs climate-mlops-system -f
   
   # Verificar estado del contenedor
   docker ps --filter "name=climate-mlops-system"
   
   # Verificar salud del contenedor
   docker inspect climate-mlops-system --format='{{.State.Health.Status}}'
   ```

### 2. Deployment en AWS ECS Fargate (Completo)

#### Prerrequisitos
- AWS CLI instalado y configurado
- Docker Desktop
- Permisos AWS para CloudFormation, ECS, ECR, IAM

#### Deployment Completo (Recomendado)

```powershell
# Deployment completo en un comando
.\deploy-aws.ps1 -SetupCredentials -DeployInfrastructure -DeployApplication
```

#### Deployment Paso a Paso

1. **Configurar credenciales en Parameter Store:**
   ```powershell
   .\deploy-aws.ps1 -SetupCredentials
   ```

2. **Deployar infraestructura:**
   ```powershell
   .\deploy-aws.ps1 -DeployInfrastructure
   ```

3. **Deployar aplicación:**
   ```powershell
   .\deploy-aws.ps1 -DeployApplication
   ```

### 3. Deployment Simplificado con Scripts Individuales

#### Para ECS (requiere infraestructura existente):
```powershell
.\deploy-ecs.ps1
```

#### Con Docker Compose:
```powershell
docker-compose up -d
```

## 📋 Estructura de Archivos de Deployment

```
evaluador2/
├── Dockerfile                 # Containerización de la aplicación
├── docker-compose.yml        # Orquestación local con Docker
├── requirements.txt          # Dependencias Python actualizadas
├── infrastructure.yaml       # CloudFormation para AWS
├── deploy-local.ps1          # Script de deployment local
├── deploy-aws.ps1            # Script completo de AWS
├── deploy-ecs.ps1            # Script específico para ECS
├── deploy-ecs.sh             # Script bash para ECS
└── .env                      # Variables de entorno (no versionado)
```

## 🔧 Configuración

### Variables de Entorno

La aplicación usa las siguientes variables de entorno:

```env
AWS_ACCESS_KEY=tu_access_key_aqui
AWS_SECRET_KEY=tu_secret_key_aqui
AWS_BUCKET=ml-ops-datos-prediccion-clima-uadec22025-ml
AWS_REGION=us-east-2
```

### Configuración AWS

El deployment en AWS crea:
- **VPC** con subnets públicas
- **ECS Cluster** con Fargate
- **ECR Repository** para imágenes Docker
- **Application Load Balancer** para acceso web
- **Security Groups** configurados
- **IAM Roles** con permisos mínimos
- **CloudWatch Logs** para monitoreo

## 📊 Monitoreo y Logging

### Local (Docker)
```powershell
# Logs de la aplicación
docker logs climate-mlops-system -f

# Métricas del contenedor
docker stats climate-mlops-system

# Archivos de log locales
./logs/
```

### AWS (ECS)
- **CloudWatch Logs**: `/ecs/mlops-climate-predictor`
- **ECS Console**: Monitor de servicios y tareas
- **ALB Health Checks**: Verificación automática de salud

## 🔍 Validación de Deployment

### Verificar Funcionamiento Local
```powershell
# Verificar que el contenedor está corriendo
docker ps --filter "name=climate-mlops-system"

# Verificar logs por errores
docker logs climate-mlops-system --tail 50

# Verificar archivos de salida
ls ./data/temp/
ls ./models/saved/
```

### Verificar Funcionamiento en AWS
```powershell
# Estado del cluster ECS
aws ecs describe-clusters --clusters mlops-climate-predictor-cluster

# Estado del servicio
aws ecs describe-services --cluster mlops-climate-predictor-cluster --services climate-predictor-service

# Logs de CloudWatch
aws logs describe-log-streams --log-group-name /ecs/mlops-climate-predictor
```

## 🚨 Troubleshooting

### Problemas Comunes

#### Error de credenciales AWS:
```powershell
# Verificar configuración AWS CLI
aws configure list
aws sts get-caller-identity

# Verificar archivo .env
Get-Content .env
```

#### Container no inicia:
```powershell
# Ver logs detallados
docker logs climate-mlops-system --details

# Verificar dependencias
docker exec -it climate-mlops-system pip list
```

#### Error de permisos S3:
```powershell
# Verificar acceso al bucket
aws s3 ls s3://ml-ops-datos-prediccion-clima-uadec22025-ml --region us-east-2
```

### Limpieza de Recursos

#### Local:
```powershell
# Parar y eliminar contenedor
docker stop climate-mlops-system
docker rm climate-mlops-system

# Eliminar imagen
docker rmi mlops-climate-predictor
```

#### AWS:
```powershell
# Eliminar stack de CloudFormation
aws cloudformation delete-stack --stack-name mlops-climate-predictor --region us-east-2

# Eliminar imágenes ECR
aws ecr batch-delete-image --repository-name mlops-climate-predictor --image-ids imageTag=latest --region us-east-2
```

## 📈 Escalabilidad

### Para escalar en AWS:
```powershell
# Aumentar número de tareas
aws ecs update-service --cluster mlops-climate-predictor-cluster --service climate-predictor-service --desired-count 3

# Cambiar recursos de CPU/memoria en task definition
# Editar infrastructure.yaml y hacer redeploy
```

## 🔐 Seguridad

- Las credenciales AWS se almacenan en Parameter Store (AWS)
- Variables de entorno locales en archivo `.env` (no versionado)
- Security Groups restringen acceso a puertos necesarios
- IAM Roles con permisos mínimos requeridos
- ECR con escaneo de vulnerabilidades habilitado

## 📚 Recursos Adicionales

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Docker Documentation](https://docs.docker.com/)
- [CloudFormation Documentation](https://docs.aws.amazon.com/cloudformation/)

---

¡Tu sistema MLOps está listo para producción! 🎉