FROM python:3.9

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Establecer el directorio de trabajo
WORKDIR /workspace

# Exponer el puerto para la visualización de OpenCV (opcional)
EXPOSE 8080