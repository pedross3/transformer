#!/bin/bash

# DirectionNet Docker Quick Start Script
# This script helps build and run the DirectionNet Docker container

# Link to download dataset: 
# https://drive.google.com/file/d/1SB1g8ectHyhG23g8FqiwKJApyvmK5Sc9/view?usp=sharing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$HOME/Documents/data"
IMAGE_NAME="directionnet_v2"
IMAGE_TAG="latest"
CONTAINER_NAME="optimus_prime"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu18.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime is not properly configured."
        print_info "Please install nvidia-docker2 and restart Docker daemon."
        exit 1
    fi
    print_success "NVIDIA Docker runtime is configured"
    
    # Check GPU
    if ! nvidia-smi &> /dev/null; then
        print_error "NVIDIA GPU driver is not installed or not working."
        exit 1
    fi
    print_success "NVIDIA GPU driver is working"
}

# Build Docker image
build_image() {
    print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    cd "$SCRIPT_DIR"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    print_success "Docker image built successfully"
}

# Detect GPU access method
detect_gpu_method() {
    # Try --gpus all
    if docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" python3.6 -c "import sys; sys.exit(0)" &>/dev/null; then
        echo "gpus"
        return 0
    fi
    
    # Try --runtime=nvidia
    if docker run --rm --runtime=nvidia "${IMAGE_NAME}:${IMAGE_TAG}" python3.6 -c "import sys; sys.exit(0)" &>/dev/null; then
        echo "runtime"
        return 0
    fi
    
    # Fallback to no GPU
    echo "none"
    return 1
}

# Get GPU flags based on detected method
get_gpu_flags() {
    local method=$(detect_gpu_method)
    case $method in
        gpus)
            echo "--gpus all"
            ;;
        runtime)
            echo "--runtime=nvidia"
            ;;
        none)
            print_warning "GPU access not available. Running without GPU." >&2
            echo ""
            ;;
    esac
}

# Run container interactively
run_interactive() {
    print_info "Starting container in interactive mode..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
        --name "${CONTAINER_NAME}" \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
        -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
        -v "${DATA_DIR}:/app/data"
        -v "${SCRIPT_DIR}/outputs:/app/outputs" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        /bin/bash
}

# Run training
run_training() {
    print_info "Starting training..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
        --name "${CONTAINER_NAME}_training" \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -v "${SCRIPT_DIR}/app/checkpoints" \
        -v "${SCRIPT_DIR}/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/app/checkpoints_cvt" \
        -v "${DATA_DIR}:/app/data" \
        -v "${SCRIPT_DIR}/app/outputs" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3.6 train.py "$@"
}

# Run rotation-only training (train_R.py)
run_training_r() {
    print_info "Starting rotation-only training (train_R.py)..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
    --name "${CONTAINER_NAME}_training_r" \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
    -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
    -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
    -v "${DATA_DIR}:/app/data" \
    -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python3.6 train_R.py "$@"
}

# Run derotation training (train_T.py)
run_training_t() {
    print_info "Starting rotation-only training (train_T.py)..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
    --name "${CONTAINER_NAME}_training_r" \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
    -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
    -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
    -v "${DATA_DIR}:/app/data"\
    -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python3.6 train_T.py "$@"
}

# Run training with pdb (interactive CLI debugger)
run_training_pdb() {
    print_info "Starting training under pdb (Python debugger)..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
        --name "${CONTAINER_NAME}_training_pdb" \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
        -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
        -v "${DATA_DIR}:/app/data" \
        -v "${SCRIPT_DIR}/outputs:/app/outputs" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3.6 -m pdb train.py "$@"
}

# Run training with VS Code debugpy (remote attach)
run_training_debugpy() {
    print_info "Starting training with debugpy (VS Code attach on localhost:5678)..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
        --name "${CONTAINER_NAME}_training_debug" \
        -p 5678:5678 \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
        -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
        -v "${DATA_DIR}:/app/data"\
        -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -lc 'python3.6 -m pip install --no-cache-dir debugpy && python3.6 -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py "$@"' bash "$@"
}

# Run evaluation
run_eval() {
    print_info "Starting evaluation..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -it --rm \
        --name "${CONTAINER_NAME}_eval" \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -v /home/Documents/data:/home/Documents/data \
        -v "${SCRIPT_DIR}/eval_data:/app/eval_data" \
        -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
        -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
        -v "${DATA_DIR}:/app/data" \
        -v "${SCRIPT_DIR}/eval_summary:/app/eval_summary" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3.6 eval_fixed.py "$@"
}

# Run TensorBoard
run_tensorboard() {
    print_info "Starting TensorBoard on port 6006..."
    local gpu_flags=$(get_gpu_flags)
    docker run $gpu_flags -d \
        --name "${CONTAINER_NAME}_tensorboard" \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -p 6006:6006 \
        -v "${SCRIPT_DIR}/checkpoints_cvt:/app/checkpoints_cvt" \
        -v "${SCRIPT_DIR}/checkpoints_baseline:/app/checkpoints_baseline" \
        -v "${SCRIPT_DIR}/checkpoints:/app/checkpoints" \
        -v "${SCRIPT_DIR}/eval_summary:/app/eval_summary" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        tensorboard --logdir_spec=train_baseline:/app/checkpoints_baseline,train_cvt:/app/checkpoints_cvt,train_old:/app/checkpoints,eval:/app/eval_summary --host=0.0.0.0
    print_success "TensorBoard started. Access it at http://localhost:6006"
}

# Test GPU access
test_gpu() {
    print_info "Testing GPU access..."
    local gpu_flags=$(get_gpu_flags)
    if [ -z "$gpu_flags" ]; then
        print_error "No GPU access method available"
        exit 1
    fi
    
    print_info "Testing with flags: $gpu_flags"
    docker run --rm $gpu_flags -e TF_FORCE_GPU_ALLOW_GROWTH=true "${IMAGE_NAME}:${IMAGE_TAG}" \
        python3.6 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('\n=== GPU Test Results ==='); print('TensorFlow version:', tf.__version__); print('Number of GPUs:', len(gpus)); [print(f'  - {gpu}') for gpu in gpus]; print('========================\n')"
}

# Stop and remove containers
cleanup() {
    print_info "Cleaning up containers..."
    docker ps -a | grep "${CONTAINER_NAME}" | awk '{print $1}' | xargs -r docker stop
    docker ps -a | grep "${CONTAINER_NAME}" | awk '{print $1}' | xargs -r docker rm
    print_success "Cleanup completed"
}

# Display usage
usage() {
    cat << EOF
DirectionNet Docker Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    check           Check prerequisites (Docker, NVIDIA drivers, etc.)
    build           Build the Docker image
    run             Run container in interactive mode
    train           Run training script with optional arguments
    train-r         Run rotation-only training (train_R.py) with optional arguments
    train-pdb       Run training under pdb (CLI debugger)
    train-debug     Run training with debugpy (VS Code remote attach on :5678)
    eval            Run evaluation script with optional arguments
    tensorboard     Start TensorBoard service
    test-gpu        Test GPU access in container
    cleanup         Stop and remove all related containers
    help            Show this help message

Examples:
    $0 check                        # Check prerequisites
    $0 build                        # Build Docker image
    $0 run                          # Run interactive shell
    $0 train --flag value           # Run training with flags
    $0 train-r --flag value         # Run rotation-only training with flags
    $0 train-pdb --flag value       # Run training with pdb
    $0 train-debug --flag value     # Run training and wait for VS Code attach
    $0 eval --checkpoint path       # Run evaluation
    $0 tensorboard                  # Start TensorBoard
    $0 test-gpu                     # Test GPU access
    $0 cleanup                      # Clean up containers

VS Code attach (add to .vscode/launch.json):
{
  "name": "Attach to Docker: DirectionNet",
  "type": "python",
  "request": "attach",
  "connect": { "host": "localhost", "port": 5678 },
  "justMyCode": false
}

Then run:
  $0 train-debug --data_dir -v /home/Documents/data:/app/data --checkpoint_dir /app/checkpoints
and press F5 to attach.

EOF
}

# Main script logic
main() {
    case "${1:-}" in
        check)
            check_prerequisites
            ;;
        build)
            check_prerequisites
            build_image
            ;;
        run)
            run_interactive
            ;;
        train)
            shift
            run_training "$@"
            ;;
        train-r)
            shift
            run_training_r "$@"
            ;;
        train-t)
            shift
            run_training_t "$@"
            ;;
        train-pdb)
            shift
            run_training_pdb "$@"
            ;;
        train-debug)
            shift
            run_training_debugpy "$@"
            ;;
        eval)
            shift
            run_eval "$@"
            ;;
        tensorboard)
            run_tensorboard
            ;;
        test-gpu)
            test_gpu
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: ${1:-}"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
