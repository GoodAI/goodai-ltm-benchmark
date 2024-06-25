#!/bin/bash

# Stop and remove the existing container
docker-compose down

# Remove the old image
docker rmi multi-agent-rag_multi-agent-rag

# Rebuild the image
docker-compose build

# Start the new container
docker-compose up -d

# Print the container name
echo "Container name: $(docker-compose ps --services)"