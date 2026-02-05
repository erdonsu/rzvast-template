"""
ComfyUI Modal Deployment Script (FIXED VERSION)
For LTX-2 + Qwen Image Edit
"""

import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

# CACHE BUSTER - Change this to force rebuild image with latest ComfyUI/Nodes
BUILD_VERSION = "2026.02.05.v2"

# Nodes that cause issues and should be removed from volume
PROBLEMATIC_NODES = [
    "ComfyUI_HFDownLoad",
    "hf-model-downloader", 
    "comfyui_hf_model_downloader",
    "comfyui-model-downloader",
    "comfyui-doctor",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    """Clone custom node from GitHub to default image location"""
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd
  
def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    """Download model from HuggingFace Hub"""
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

# =============================================================================
# IMAGE DEFINITION
# =============================================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "curl")
    .run_commands([
        "pip install --upgrade pip",
        # Install comfy-cli latest
        "pip install --no-cache-dir --upgrade comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        
        # PyTorch CUDA 13.0 (Official ComfyUI Recommendation)
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130",
        "pip install triton>=3.0.0",
        
        # Install ComfyUI Core
        "comfy --skip-prompt install --nvidia --skip-torch-or-directml",
        
        # Version check
        f"echo 'BUILD_VERSION={BUILD_VERSION}' && cat /root/comfy/ComfyUI/comfy/version.py || echo 'version file not found'",
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0"
    })
    # Core Dependencies
    .run_commands([
        "pip install ftfy accelerate einops diffusers sentencepiece sageattention",
        "pip install onnx onnxruntime onnxruntime-gpu",
        "pip install opencv-contrib-python-headless easyocr",
        # LTXVideo Dependencies
        "pip install ninja~=1.11.1.4 'transformers[timm]>=4.50.0' 'huggingface_hub>=0.25.2'",
        # Audio Dependencies
        "pip install soundfile librosa scipy",
    ])
)

# Install Registry Nodes
image = image.run_commands([
    "comfy --skip-prompt node install "
    "rgthree-comfy "
    "comfyui-impact-pack "
    "comfyui-impact-subpack "
    "ComfyUI-YOLO "
    "comfyui-inspire-pack "
    "comfyui_ipadapter_plus "
    "wlsh_nodes "
    "ComfyUI_Comfyroll_CustomNodes "
    "comfyui_essentials "
    "ComfyUI-GGUF "
    "ComfyUI-LTXVideo "
    "ComfyUI-Manager "
    "comfyui-kjnodes "
    "ComfyUI-VideoHelperSuite "
    "ComfyUI-Custom-Scripts "
    "ComfyMath "
    "ComfyUI-Easy-Use"
])

# Install Git-based Nodes (including MISSING ones)
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("luguoli/ComfyUI-Qwen-Image-Integrated-KSampler", {'install_reqs': True}),
    ("jtydhr88/ComfyUI-qwenmultiangle", {}),
    
    # FIX: Added missing TTP Toolset for LTXVFirstLastFrameControl_TTP
    ("TTPlanetPig/Comfyui_TTP_Toolset", {'install_reqs': True}),
    
    # FIX: Added missing Tensor Parallel for LTX-2
    ("RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management", {}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Install Requirements for Specific Nodes
image = image.run_commands([
    f"pip install -r {DEFAULT_COMFY_DIR}/custom_nodes/ComfyUI-Manager/requirements.txt || true",
    f"pip install -r {DEFAULT_COMFY_DIR}/custom_nodes/ComfyUI-LTXVideo/requirements.txt || true",
    f"pip install -r {DEFAULT_COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt || true",
])

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

model_tasks = [
    # LTX-2 Models
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-temporal-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    ("loras", "ltx-2-19b-ic-lora-detailer.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Detailer", None),
    ("text_encoders", "gemma_3_12B_it_fp8_e4m3fn.safetensors", "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn", None),
    
    # Qwen Image Edit Models
    ("diffusion_models", "qwen_image_edit_2511_bf16.safetensors", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models"),
    ("vae", "qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae"),
    ("clip", "qwen_2.5_vl_7b.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders"),
    ("loras", "Qwen-Image-Lightning-8steps-V1.1.safetensors", "lightx2v/Qwen-Image-Lightning", None),
    ("loras", "qwen-image-edit-2511-multiple-angles-lora.safetensors", "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", None),
]

extra_cmds = [
    f"wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# =============================================================================
# APP LOGIC
# =============================================================================

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=1800,
    gpu=os.environ.get('MODAL_GPU_TYPE', 'A100-40GB'),
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():
    # Setup directories
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(DATA_BASE, exist_ok=True)
    
    # 1. Sync Logic (Image -> Volume)
    PRESERVE_DIRS = ["models", "custom_nodes", "user", "input", "output"]
    
    if os.path.exists(DEFAULT_COMFY_DIR):
        print("Syncing ComfyUI from image to volume...")
        for item in os.listdir(DEFAULT_COMFY_DIR):
            src = os.path.join(DEFAULT_COMFY_DIR, item)
            dst = os.path.join(DATA_BASE, item)
            
            if item in PRESERVE_DIRS and os.path.exists(dst):
                print(f"Preserving existing: {item}")
                continue
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print("Sync completed.")

    # 2. Cleanup
    for node_name in PROBLEMATIC_NODES:
        node_path = os.path.join(CUSTOM_NODES_DIR, node_name)
        if os.path.exists(node_path):
            shutil.rmtree(node_path)
            print(f"Removed problematic node: {node_name}")

    # 3. Git Fixes
    os.chdir(DATA_BASE)
    try:
        subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True)
        subprocess.run("git config pull.ff only", shell=True)
        subprocess.run("git pull --ff-only", shell=True)
    except Exception as e:
        print(f"Git update warning: {e}")

    # 4. Sync Custom Nodes again (to be safe)
    image_nodes_dir = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes")
    if os.path.exists(image_nodes_dir):
        for node_name in os.listdir(image_nodes_dir):
            src = os.path.join(image_nodes_dir, node_name)
            dst = os.path.join(CUSTOM_NODES_DIR, node_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                print(f"Restored node: {node_name}")

    # 5. Manager Config
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    os.makedirs(manager_config_dir, exist_ok=True)
    with open(os.path.join(manager_config_dir, "config.ini"), "w") as f:
        f.write("[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\n")

    # 6. Install Runtime Requirements
    print("Installing requirements...")
    subprocess.run("pip install --no-cache-dir --upgrade pip comfy-cli", shell=True)
    if os.path.exists(f"{DATA_BASE}/requirements.txt"):
        subprocess.run(f"pip install -r {DATA_BASE}/requirements.txt", shell=True)

    # 7. Download Models
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)
        
    print("Checking models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            try:
                print(f"Downloading {fn}...")
                hf_download(sub, fn, repo, subf)
            except Exception as e:
                print(f"Failed to download {fn}: {e}")

    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True, cwd=DATA_BASE)

    # 8. Launch
    vol.commit()
    os.environ["COMFY_DIR"] = DATA_BASE
    
    print("Launching ComfyUI...")
    subprocess.run(f"comfy --workspace={DATA_BASE} manager enable-gui", shell=True)
    
    cmd = f"comfy --workspace={DATA_BASE} launch -- --listen 0.0.0.0 --port 8000"
    subprocess.Popen(cmd, shell=True, cwd=DATA_BASE, env=os.environ.copy())
