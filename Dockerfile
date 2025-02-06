FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /project/fyp24_gib4/

# Copy the requirements file
COPY requirements.txt .

RUN --mount=type=secret,id=OPENWEATHER_API_KEY \
    --mount=type=secret,id=WINDY_WEBCAM_API_KEY \
    bash -c 'export OPENWEATHER_API_KEY=$(cat /run/secrets/OPENWEATHER_API_KEY) && \
    export WINDY_WEBCAM_API_KEY=$(cat /run/secrets/WINDY_WEBCAM_API_KEY)'
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set environment variables if needed
ENV PYTHONUNBUFFERED=1

# Expose any ports if needed (example: 5000)
# EXPOSE 5000

# Command to run your application (example: python main.py)
# CMD ["python", "main.py"]