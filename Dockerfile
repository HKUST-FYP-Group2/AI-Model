FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /model

# Copy the requirements file
COPY requirements.txt .

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