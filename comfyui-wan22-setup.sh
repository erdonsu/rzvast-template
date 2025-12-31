#!/bin/bash
# =============================================================================
# ComfyUI Wan2.2 Provisioning Script
# Auto-setup models + custom nodes saat instance start
# =============================================================================

set -e

echo "=========================================="
echo "ComfyUI Wan2.2 Setup Starting..."
echo "=========================================="

# Detect ComfyUI path (vastai/comfy uses /opt/workspace-internal/ComfyUI)
if [ -d "/opt/workspace-internal/ComfyUI" ]; then
    COMFY_DIR="/opt/workspace-internal/ComfyUI"
elif [ -d "/workspace/ComfyUI" ]; then
    COMFY_DIR="/workspace/ComfyUI"
elif [ -d "/opt/ComfyUI" ]; then
    COMFY_DIR="/opt/ComfyUI"
elif [ -d "/ComfyUI" ]; then
    COMFY_DIR="/ComfyUI"
else
    echo "ComfyUI not found, cloning..."
    mkdir -p /opt/workspace-internal
    cd /opt/workspace-internal
    git clone https://github.com/comfyanonymous/ComfyUI.git
    COMFY_DIR="/opt/workspace-internal/ComfyUI"
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
# Download Models (~45GB total)
# =============================================================================
echo "=========================================="
echo "Downloading Models (this may take 15-20 minutes)..."
echo "=========================================="

# Function to download with wget (follows redirects automatically)
download_model() {
    local url=$1
    local output_dir=$2
    local filename=$3
    
    if [ -f "$output_dir/$filename" ] && [ -s "$output_dir/$filename" ]; then
        echo "✓ $filename already exists, skipping..."
    else
        echo "Downloading $filename..."
        rm -f "$output_dir/$filename" 2>/dev/null
        wget -c -P "$output_dir" -O "$output_dir/$filename" "$url"
    fi
}

# TEXT ENCODER (6.3GB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$COMFY_DIR/models/text_encoders" \
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# DIFFUSION MODELS - I2V (16GB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors" \
    "$COMFY_DIR/models/diffusion_models" \
    "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"

# DIFFUSION MODELS - T2V (14GB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors" \
    "$COMFY_DIR/models/diffusion_models" \
    "wan2.1_t2v_14B_fp8_e4m3fn.safetensors"

# DIFFUSION MODELS - I2V LOW (14GB) - From Kijai (REQUIRES LOGIN)
# This model requires HuggingFace authentication
# To download: Set HF_TOKEN env var or download manually
echo "NOTE: Wan2_2-I2V-A14B-LOW requires HuggingFace login"
echo "If you have a token, run:"
echo "  wget --header='Authorization: Bearer YOUR_TOKEN' 'https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors'"

# VAE (243MB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$COMFY_DIR/models/vae" \
    "wan_2.1_vae.safetensors"

# CLIP VISION (1.2GB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "$COMFY_DIR/models/clip_vision" \
    "clip_vision_h.safetensors"

# LORA (298MB) - From Comfy-Org (PUBLIC)
download_model \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/loras/wan_alpha_2.1_rgba_lora.safetensors" \
    "$COMFY_DIR/models/loras" \
    "wan_alpha_2.1_rgba_lora.safetensors"

# =============================================================================
# OPTIONAL: Download LIGHTX2V LoRA (Requires HuggingFace Login)
# =============================================================================
echo ""
echo "=========================================="
echo "OPTIONAL: LIGHTX2V LoRA Models"
echo "=========================================="
echo "These models require HuggingFace authentication:"
echo "  - LIGHTX2V_I2V_2.2_HIGH.safetensors (~100MB)"
echo "  - LIGHTX2V_I2V_2.2_LOW.safetensors (~100MB)"
echo ""
echo "To download manually:"
echo "  wget --header='Authorization: Bearer YOUR_TOKEN' \\"
echo "    'https://huggingface.co/Kijai/wan2.2_comfyui_wrapper/resolve/main/LIGHTX2V_I2V_2.2_HIGH.safetensors'"
echo ""
echo "Or set HF_TOKEN environment variable before running this script."

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
echo ""
echo "Models installed:"
echo "  - Text Encoder: umt5_xxl_fp8_e4m3fn_scaled (6.3GB)"
echo "  - I2V Model: wan2.1_i2v_480p_14B_fp8_e4m3fn (16GB)"
echo "  - T2V Model: wan2.1_t2v_14B_fp8_e4m3fn (14GB)"
echo "  - VAE: wan_2.1_vae (243MB)"
echo "  - CLIP Vision: clip_vision_h (1.2GB)"
echo "  - LoRA: wan_alpha_2.1_rgba_lora (298MB)"
echo ""
echo "Optional (requires login):"
echo "  - Wan2_2-I2V-A14B-LOW (14GB)"
echo "  - LIGHTX2V LoRA models (~200MB)"
echo ""
echo "ComfyUI should be available at port 8188"
echo "=========================================="
