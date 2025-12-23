#!/bin/bash

# Helper script for running neckenml-analyzer in Docker

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_help() {
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  start         Start the container"
    echo "  stop          Stop the container"
    echo "  shell         Enter container shell"
    echo "  evaluate      Run evaluate_classification.py --verbose"
    echo "  matrix        Generate and display confusion matrix visualization"
    echo "  check         Run check_setup.py"
    echo "  models        Download required Essentia models"
    echo "  logs          View container logs"
    echo "  clean         Stop and remove containers/volumes"
    echo ""
}

function download_models() {
    echo -e "${GREEN}Downloading Essentia models...${NC}"
    mkdir -p ~/.neckenml/models

    if [ ! -f ~/.neckenml/models/msd-musicnn-1.pb ]; then
        echo "Downloading MusiCNN embedding model..."
        curl -L https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb \
          -o ~/.neckenml/models/msd-musicnn-1.pb
    else
        echo "MusiCNN model already exists, skipping..."
    fi

    if [ ! -f ~/.neckenml/models/voice_instrumental-msd-musicnn-1.pb ]; then
        echo "Downloading voice/instrumental classifier..."
        curl -L https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.pb \
          -o ~/.neckenml/models/voice_instrumental-msd-musicnn-1.pb
    else
        echo "Voice/instrumental model already exists, skipping..."
    fi

    # Create symlink with expected filename if it doesn't exist
    if [ ! -e ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb ]; then
        echo "Creating symlink for voice_instrumental model..."
        ln -s ~/.neckenml/models/voice_instrumental-msd-musicnn-1.pb \
          ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb
    fi

    echo -e "${GREEN}Models downloaded successfully!${NC}"
}

case "$1" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        echo -e "${YELLOW}Note: First build on Apple Silicon may take 10-15 minutes due to x86_64 emulation${NC}"
        docker-compose build neckenml-analyzer
        ;;
    start)
        echo -e "${GREEN}Starting container...${NC}"
        docker-compose up -d neckenml-analyzer
        ;;
    stop)
        echo -e "${GREEN}Stopping container...${NC}"
        docker-compose down
        ;;
    shell)
        echo -e "${GREEN}Entering container shell...${NC}"
        docker-compose exec neckenml-analyzer bash
        ;;
    evaluate)
        echo -e "${GREEN}Running evaluate_classification.py...${NC}"
        docker-compose exec neckenml-analyzer python evaluate_classification.py --verbose
        ;;
    check)
        echo -e "${GREEN}Running check_setup.py...${NC}"
        docker-compose exec neckenml-analyzer python check_setup.py
        ;;
    matrix)
        echo -e "${GREEN}Generating confusion matrix...${NC}"
        docker-compose exec neckenml-analyzer python visualize_confusion_matrix.py
        echo ""
        echo -e "${GREEN}Confusion matrix saved!${NC}"
        echo "View the matrix at: $(pwd)/confusion_matrix.png"
        if command -v open &> /dev/null; then
            echo -e "${GREEN}Opening image...${NC}"
            open confusion_matrix.png
        elif command -v xdg-open &> /dev/null; then
            echo -e "${GREEN}Opening image...${NC}"
            xdg-open confusion_matrix.png
        else
            echo "Open 'confusion_matrix.png' manually to view the visualization"
        fi
        ;;
    models)
        download_models
        ;;
    logs)
        docker-compose logs -f neckenml-analyzer
        ;;
    clean)
        echo -e "${YELLOW}Stopping and removing containers/volumes...${NC}"
        docker-compose down -v
        echo -e "${GREEN}Cleanup complete!${NC}"
        ;;
    help|--help|-h|"")
        print_help
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac
