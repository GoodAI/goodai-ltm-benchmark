#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --mode <cli|api>    Specify the mode to run (CLI or API)"
    echo "  -r, --rebuild           Force rebuild of the Docker image"
    echo "  -h, --help              Display this help message"
    exit 1
}

# Default values
MODE=""
REBUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--mode)
        MODE="$2"
        shift
        shift
        ;;
        -r|--rebuild)
        REBUILD=true
        shift
        ;;
        -h|--help)
        usage
        ;;
        *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Check if mode is specified
if [ -z "$MODE" ]; then
    echo "Error: Mode not specified."
    usage
fi

# Validate mode
if [ "$MODE" != "cli" ] && [ "$MODE" != "api" ]; then
    echo "Error: Invalid mode. Use 'cli' or 'api'."
    usage
fi

# Stop and remove the existing containers
echo "Stopping and removing existing containers..."
docker-compose down

# Remove the old image if rebuild is requested
if [ "$REBUILD" = true ]; then
    echo "Removing old image..."
    docker rmi multi-agent-rag_multi-agent-rag
fi

# Rebuild the image
echo "Building Docker image..."
docker-compose build

# Start the new container based on the mode
if [ "$MODE" = "cli" ]; then
    echo "Starting container in CLI mode..."
    docker-compose run --rm multi-agent-rag-cli
elif [ "$MODE" = "api" ]; then
    echo "Starting container in API mode..."
    docker-compose up -d multi-agent-rag
    
    # Wait for the container to be ready
    echo "Waiting for the API to be ready..."
    while ! curl -s http://localhost:8080/health > /dev/null; do
        sleep 1
    done
    echo "API is ready!"

    # Print the container name
    CONTAINER_NAME=$(docker-compose ps --services)
    echo "Container name: $CONTAINER_NAME"
    
    # Print the logs
    echo "Container logs:"
    docker-compose logs -f
fi