#!/bin/bash
# =============================================================================
# ComfyUI Wan2.2 Provisioning Script
# Auto-setup models + custom nodes saat instance start
# =============================================================================

set -e

echo "=========================================="
echo "ComfyUI Wan2.2 Setup Starting..."
echo "=========================================="

# Detect ComfyUI path (vastai/comfy biasanya di /workspace/ComfyUI atau /opt/ComfyUI)
if [ -d "/workspace/ComfyUI" ]; then
    COMFY_DIR="/workspace/ComfyUI"
elif [ -d "/opt/ComfyUI" ]; then
    COMFY_DIR="/opt/ComfyUI"
elif [ -d "/ComfyUI" ]; then
    COMFY_DIR="/ComfyUI"
else
    echo "ComfyUI not found, cloning..."
    cd /workspace
    git clone https://github.com/comfyanonymous/ComfyUI.git
    COMFY_DIR="/workspace/ComfyUI"
fi

echo "ComfyUI directory: $COMFY_DIR"

# =============================================================================
# Install aria2 for fast downloads
# =============================================================================
echo "Installing aria2..."
apt-get update && apt-get install -y aria2 || true

# =============================================================================
# Create model directories
# =============================================================================
echo "Creating model directories..."
mkdir -p $COMFY_DIR/models/checkpoints
mkdir -p $COMFY_DIR/models/loras
mkdir -p $COMFY_DIR/models/vae
mkdir -p $COMFY_DIR/models/diffusion_models
mkdir -p $COMFY_DIR/models/text_encoders
mkdir -p $COMFY_DIR/models/clip_vision

# =============================================================================
# Download Models (~41GB)
# =============================================================================
echo "=========================================="
echo "Downloading Models (this may take 15-20 minutes)..."
echo "=========================================="

# Function to download with aria2 (fast, resume support)
download_model() {
    local url=$1
    local output_dir=$2
    local filename=$3
    
    if [ -f "$output_dir/$filename" ]; then
        echo "✓ $filename already exists, skipping..."
    else
        echo "Downloading $filename..."
        aria2c -x 16 -s 16 -k 1M -d "$output_dir" -o "$filename" "$url" || \
        wget -P "$output_dir" -O "$output_dir/$filename" "$url"
    fi
}

# Text Encoder (11GB)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$COMFY_DIR/models/text_encoders" \
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# Diffusion Model - 14B I2V (10GB)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_14b_480p_bf16.safetensors" \
    "$COMFY_DIR/models/diffusion_models" \
    "wan2.1_i2v_14b_480p_bf16.safetensors"

# Diffusion Model - 14B T2V (10GB)  
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14b_bf16.safetensors" \
    "$COMFY_DIR/models/diffusion_models" \
    "wan2.1_t2v_14b_bf16.safetensors"

# VAE (243MB)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae_bf16.safetensors" \
    "$COMFY_DIR/models/vae" \
    "wan_2.1_vae_bf16.safetensors"

# CLIP Vision (1.6GB)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "$COMFY_DIR/models/clip_vision" \
    "clip_vision_h.safetensors"

# =============================================================================
# Clone Custom Nodes
# =============================================================================
echo "=========================================="
echo "Cloning Custom Nodes..."
echo "=========================================="

cd $COMFY_DIR/custom_nodes

# Function to clone if not exists
clone_node() {
    local repo=$1
    local name=$(basename $repo .git)
    
    if [ -d "$name" ]; then
        echo "✓ $name already exists, pulling latest..."
        cd $name && git pull || true && cd ..
    else
        echo "Cloning $name..."
        git clone $repo || true
    fi
}

# Core Wan2.2 Nodes
clone_node https://github.com/kijai/ComfyUI-WanVideoWrapper.git
clone_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
clone_node https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
clone_node https://github.com/kijai/ComfyUI-segment-anything-2.git
clone_node https://github.com/kijai/ComfyUI-DepthAnythingV2.git
clone_node https://github.com/kijai/ComfyUI-Florence2.git

# Utility Nodes
clone_node https://github.com/kijai/ComfyUI-KJNodes.git
clone_node https://github.com/WASasquatch/was-node-suite-comfyui.git
clone_node https://github.com/yolain/ComfyUI-Easy-Use.git
clone_node https://github.com/city96/CRT-Nodes.git
clone_node https://github.com/chrisgoringe/cg-use-everywhere.git
clone_node https://github.com/MinusZoneAI/ComfyUI-mxToolkit.git
clone_node https://github.com/chibiace/ComfyUI-Chibi-Nodes.git
clone_node https://github.com/city96/ComfyUI-VAE-Utils.git
clone_node https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
clone_node https://github.com/cubiq/ComfyUI_essentials.git
clone_node https://github.com/rgthree/rgthree-comfy.git
clone_node https://github.com/jamesWalker55/comfyui-various.git
clone_node https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git

# =============================================================================
# Install Custom Node Requirements
# =============================================================================
echo "=========================================="
echo "Installing Custom Node Requirements..."
echo "=========================================="

# Activate venv if exists
if [ -f "/venv/main/bin/activate" ]; then
    source /venv/main/bin/activate
fi

# Install requirements for each node
for dir in $COMFY_DIR/custom_nodes/*/; do
    if [ -f "$dir/requirements.txt" ]; then
        echo "Installing requirements for $(basename $dir)..."
        pip install -r "$dir/requirements.txt" --quiet || true
    fi
done

# Install additional common dependencies
pip install --quiet soundfile accelerate || true

# =============================================================================
# Done!
# =============================================================================
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "ComfyUI directory: $COMFY_DIR"
echo "Models downloaded to: $COMFY_DIR/models/"
echo "Custom nodes installed: 19 nodes"
echo ""
echo "ComfyUI should be available at port 8188"
echo "=========================================="
