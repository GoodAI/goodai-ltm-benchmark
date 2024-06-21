#!/bin/bash

# Script to reset Docker Compose setup

# Step 1: Stop and remove all containers, networks, and volumes created by docker-compose
echo "Stopping and removing all containers, networks, and volumes..."
sudo docker-compose down

# Step 2: Build the Docker images specified in the docker-compose file
echo "Building Docker images..."
sudo docker-compose build

# Step 3: Start the Docker containers in detached mode
echo "Starting Docker containers..."
sudo docker-compose up -d

echo "Docker reset complete."
