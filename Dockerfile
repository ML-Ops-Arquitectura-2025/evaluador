# Usa la imagen base oficial de Python para Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copiar requirements y instalar dependencias
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copiar el código fuente
COPY evaluate.py ${LAMBDA_TASK_ROOT}

# Definir el handler
CMD ["evaluate.lambda_handler"]
