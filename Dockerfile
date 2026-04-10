# Python ka base image
FROM python:3.9-slim

# System level updates aur OpenCV ki dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Project folder setup
WORKDIR /app
COPY . /app

# Requirements install karna
RUN pip install --no-cache-dir -r requirements.txt

# Tera project run karne ki command
# Yahan 'main.py' ko apni file ke naam se replace kar dena
CMD ["python", "main.py"]
