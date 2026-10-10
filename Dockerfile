# Dockerfile per SmallLLM
FROM python:3.11-slim

WORKDIR /app

# Copia i file
COPY requirements.txt .
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta
EXPOSE 5000

# Comando di avvio
CMD ["python", "start.py"]
