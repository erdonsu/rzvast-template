"""
ComfyUI Modal Deployment - Simple & Stable Version
Direct install from GitHub (no comfy-cli)
"""

import os
import shutil
import subprocess
from typing import Optional
import modal

# =============================================================================
# PATHS
# =============================================================================
COMFY_DIR = "/root/ComfyUI"
CUSTOM_NODES_DIR = f"{COMFY_DIR}/custom_nodes"
MODELS_DIR = f"{COMFY_DIR}/models"

# Volume mount point
VOL_PATH = "/data"
VOL_COMFY = f"{VOL_PATH}/ComfyUI"

# =============================================================================
# IMAGE BUILD - Using stable CUDA 12.1 + PyTorch 2.5
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender1"
    )
    .pip_install(
        # PyTorch stable (CUDA 12.1)
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Core dependencies
        "aiohttp",
        "einops",
        "transformers>=4.28.1",
        "safetensors>=0.4.2",
        "accelerate",
        "pyyaml",
        "Pillow",
        "scipy",
        "tqdm",
        "psutil",
        "kornia>=0.7.1",
        "spandrel",
        "soundfile",
        "av",
    )
    .run_commands([
        # Clone ComfyUI from official repo
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}",
        f"pip install -r {COMFY_DIR}/requirements.txt",
        
        # Clone ComfyUI-Manager
        f"git clone https://github.com/Comfy-Org/ComfyUI-Manager.git {CUSTOM_NODES_DIR}/ComfyUI-Manager",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-Manager/requirements.txt || true",
    ])
    .pip_install(
        # Extra dependencies for nodes
        "huggingface_hub[hf_transfer]",
        "opencv-python-headless",
        "scikit-image",
        "diffusers",
        "ftfy",
        "regex",
        "sentencepiece",
    )
    .run_commands([
        # Clone essential nodes
        f"git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git {CUSTOM_NODES_DIR}/ComfyUI-LTXVideo",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-LTXVideo/requirements.txt || true",
        
        f"git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git {CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite/requirements.txt || true",
        
        f"git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git {CUSTOM_NODES_DIR}/ComfyUI-Custom-Scripts",
        
        f"git clone https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git {CUSTOM_NODES_DIR}/Comfyui_TTP_Toolset",
        f"pip install -r {CUSTOM_NODES_DIR}/Comfyui_TTP_Toolset/requirements.txt || true",
        
        f"git clone https://github.com/rgthree/rgthree-comfy.git {CUSTOM_NODES_DIR}/rgthree-comfy",
        
        f"git clone https://github.com/cubiq/ComfyUI_essentials.git {CUSTOM_NODES_DIR}/ComfyUI_essentials",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI_essentials/requirements.txt || true",
        
        f"git clone https://github.com/evanspearman/ComfyMath.git {CUSTOM_NODES_DIR}/ComfyMath",
        
        f"git clone https://github.com/city96/ComfyUI-GGUF.git {CUSTOM_NODES_DIR}/ComfyUI-GGUF",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-GGUF/requirements.txt || true",
        
        f"git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git {CUSTOM_NODES_DIR}/ComfyUI-Impact-Pack --recursive",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-Impact-Pack/requirements.txt || true",
        
        f"git clone https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management.git {CUSTOM_NODES_DIR}/ComfyUI_LTX-2_VRAM_Memory_Management",
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# =============================================================================
# APP
# =============================================================================

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)
app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    gpu=os.environ.get('MODAL_GPU_TYPE', 'A100-40GB'),
    volumes={VOL_PATH: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():
    from huggingface_hub import hf_hub_download
    
    # Use volume for persistent storage
    work_dir = VOL_COMFY if os.path.exists(f"{VOL_COMFY}/main.py") else COMFY_DIR
    
    # First run: copy ComfyUI to volume
    if not os.path.exists(f"{VOL_COMFY}/main.py"):
        print("First run - copying ComfyUI to volume...")
        shutil.copytree(COMFY_DIR, VOL_COMFY, dirs_exist_ok=True)
        work_dir = VOL_COMFY
    
    # Ensure models directory exists
    models_dir = f"{work_dir}/models"
    os.makedirs(f"{models_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{models_dir}/loras", exist_ok=True)
    os.makedirs(f"{models_dir}/text_encoders", exist_ok=True)
    os.makedirs(f"{models_dir}/vae", exist_ok=True)
    os.makedirs(f"{models_dir}/clip", exist_ok=True)
    os.makedirs(f"{models_dir}/diffusion_models", exist_ok=True)
    os.makedirs(f"{models_dir}/upscale_models", exist_ok=True)
    
    # Download LTX-2 models if not exists
    ltx_models = [
        ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2"),
        ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2"),
        ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2"),
        ("text_encoders", "gemma_3_12B_it_fp8_e4m3fn.safetensors", "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn"),
    ]
    
    for subdir, filename, repo in ltx_models:
        target = f"{models_dir}/{subdir}/{filename}"
        if not os.path.exists(target):
            print(f"Downloading {filename}...")
            try:
                downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir="/tmp/hf")
                shutil.move(downloaded, target)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
    
    # Sync custom nodes from image if missing in volume
    vol_nodes = f"{work_dir}/custom_nodes"
    img_nodes = f"{COMFY_DIR}/custom_nodes"
    
    if os.path.exists(img_nodes):
        for node in os.listdir(img_nodes):
            src = f"{img_nodes}/{node}"
            dst = f"{vol_nodes}/{node}"
            if os.path.isdir(src) and not os.path.exists(dst):
                print(f"Syncing node: {node}")
                shutil.copytree(src, dst)
    
    # Configure Manager
    manager_config = f"{work_dir}/user/default/comfy.settings.json"
    os.makedirs(os.path.dirname(manager_config), exist_ok=True)
    
    # Commit changes
    vol.commit()
    
    # Launch ComfyUI
    print(f"Starting ComfyUI from {work_dir}...")
    os.chdir(work_dir)
    
    cmd = ["python", "main.py", "--listen", "0.0.0.0", "--port", "8000"]
    subprocess.Popen(cmd, cwd=work_dir)
