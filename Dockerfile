FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /model

# Copy the requirements file
COPY requirements.txt .

RUN --mount=type=secret,id=openWeather_api_key\
    --mount=type=secret,id=Windy_webCam_api_key\
    export openWeather_api_key=$(cat /run/secrets/openWeather_api_key) &&\
    export Windy_webCam_api_key=$(cat /run/secrets/Windy_webCam_api_key)

RUN env | grep openWeather_api_key
RUN env | grep Windy_webCam_api_key
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