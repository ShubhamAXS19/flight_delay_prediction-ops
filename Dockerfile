FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy the shell script into the container
COPY start.sh /app/

# Ensure the script is executable
RUN chmod +x /app/start.sh

# Use the shell script as the entry point
CMD ["/app/start.sh"]
