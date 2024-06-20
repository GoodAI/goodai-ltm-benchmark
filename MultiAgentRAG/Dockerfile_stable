# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create necessary directories
RUN mkdir -p /app/logs

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_lite.txt

# Make port 8000 available to the world outside this container (if needed)
EXPOSE 8000

# Define environment variable
ENV NAME Multi-Agent-RAG

# Run app.py when the container launches
CMD ["python", "src/app.py"]
