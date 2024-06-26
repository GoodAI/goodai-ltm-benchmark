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

# Function to check API health
check_api_health() {
    local max_attempts=30
    local attempt=0
    local wait_time=2

    echo "Waiting for the API to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        echo "Attempt $((attempt+1)) of $max_attempts..."
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
        if [ "$response" = "200" ]; then
            echo "API is ready!"
            return 0
        fi
        echo "Response code: $response"
        echo "Container logs (last 10 lines):"
        docker-compose logs --tail=10 multi-agent-rag
        attempt=$((attempt+1))
        sleep $wait_time
    done

    echo "API failed to start after $max_attempts attempts."
    echo "Full container logs:"
    docker-compose logs multi-agent-rag
    return 1
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
    
    # Check API health
    if check_api_health; then
        # Print the container name
        CONTAINER_NAME=$(docker-compose ps --services | grep multi-agent-rag)
        echo "Container name: $CONTAINER_NAME"
        
        # Print the logs
        echo "Container logs:"
        docker-compose logs -f multi-agent-rag
    else
        echo "Error: API failed to start. Please check the logs above for more information."
        exit 1
    fi
fi