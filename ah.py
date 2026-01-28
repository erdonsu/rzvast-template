"""
ComfyUI Modal Deployment Script
===============================

Sumber Resmi (Verified):
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- ComfyUI-Manager: https://github.com/Comfy-Org/ComfyUI-Manager
- comfy-cli: https://github.com/Comfy-Org/comfy-cli (pip install comfy-cli)

PyTorch CUDA 13.0 (dari README resmi ComfyUI):
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
"""

import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"

# ComfyUI default install location (by comfy-cli)
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

# Nodes yang bermasalah dan harus dihapus dari volume
PROBLEMATIC_NODES = [
    "ComfyUI_HFDownLoad",
    "hf-model-downloader", 
    "comfyui_hf_model_downloader",
    "comfyui-model-downloader",
]

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    """Clone custom node dari GitHub ke lokasi default ComfyUI"""
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd
  
def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    """Download model dari HuggingFace Hub"""
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# =============================================================================
# IMAGE BUILD
# =============================================================================
# Menggunakan NVIDIA CUDA base image dengan PyTorch CUDA 13.0
# Sesuai dokumentasi resmi ComfyUI: https://github.com/comfyanonymous/ComfyUI
# =============================================================================

image = (
    modal.Image.from_registry(
        # NVIDIA CUDA 12.8 runtime dengan cuDNN (kompatibel dengan PyTorch cu130)
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "curl")
    .run_commands([
        "pip install --upgrade pip",
        # Install comfy-cli dari sumber resmi (Comfy-Org)
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # ============================================================
        # PyTorch dengan CUDA 13.0 - SESUAI DOKUMENTASI RESMI ComfyUI
        # Source: https://github.com/comfyanonymous/ComfyUI#nvidia
        # ============================================================
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130",
        # Install triton untuk optimized operations
        "pip install triton>=3.0.0",
        # ============================================================
        # Install ComfyUI menggunakan comfy-cli resmi
        # Source: https://github.com/Comfy-Org/comfy-cli
        # ============================================================
        "comfy --skip-prompt install --nvidia --skip-torch-or-directml"
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0"  # Support untuk A100, A10, L40S, H100
    })
    # Dependencies untuk custom nodes
    .run_commands([
        "pip install ftfy accelerate einops diffusers sentencepiece sageattention",
        "pip install onnx onnxruntime onnxruntime-gpu",
        # Fix untuk opencv ximgproc (guidedFilter) - dibutuhkan beberapa nodes
        "pip install opencv-contrib-python-headless",
        # Fix untuk easyocr - dibutuhkan beberapa nodes
        "pip install easyocr",
    ])
)

# =============================================================================
# INSTALL CUSTOM NODES
# =============================================================================
# Menggunakan comfy-cli node install (sumber resmi)
# Nodes diinstall ke lokasi default: /root/comfy/ComfyUI/custom_nodes/
# =============================================================================

image = image.run_commands([
    # Install nodes menggunakan comfy-cli (akan mengambil dari registry resmi)
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
    # Nodes untuk workflow LTX-2 Distilled
    "ComfyUI-VideoHelperSuite "  # VHS_VideoCombine - https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
    "ComfyUI-Custom-Scripts "    # MathExpression|pysssss - https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    "ComfyMath "                 # CM_FloatToInt - https://github.com/evanspearman/ComfyMath
    "ComfyUI-Easy-Use"           # easy showAnything - https://github.com/yolain/ComfyUI-Easy-Use
])

# Git-based nodes yang di-bake ke image (untuk nodes yang tidak ada di registry)
for repo, flags in [
    # UltimateSDUpscale - sumber: https://github.com/ssitu/ComfyUI_UltimateSDUpscale
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    # SaveImageWithMetaData - sumber: https://github.com/nkchocoai/ComfyUI-SaveImageWithMetaData
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    # Prompt Reader Node - sumber: https://github.com/receyuki/comfyui-prompt-reader-node
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# =============================================================================
# MODEL DOWNLOADS (Runtime)
# =============================================================================
# Models akan di-download saat runtime untuk menghemat build time
# =============================================================================

model_tasks = [
    # LTX-2 Models dari Lightricks (HuggingFace)
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-temporal-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),  # Temporal upscaler
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    # IC-LoRA Detailer untuk video enhancement
    ("loras", "ltx-2-19b-ic-lora-detailer.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Detailer", None),
    # Gemma FP8 text encoder (lebih ringan dari FP4)
    ("text_encoders", "gemma_3_12B_it_fp8_e4m3fn.safetensors", "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn", None),
]

extra_cmds = [
    # RealESRGAN upscaler dari xinntao
    f"wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# =============================================================================
# MODAL APP CONFIGURATION
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
    """
    Main UI function yang menjalankan ComfyUI
    
    Flow:
    1. Cek apakah volume kosong (first run)
    2. Copy ComfyUI dari image ke volume jika perlu
    3. Cleanup problematic nodes
    4. Sync missing nodes dari image
    5. Update ComfyUI dan Manager
    6. Download models
    7. Launch ComfyUI
    """
    
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(DEFAULT_COMFY_DIR):
            print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
            os.makedirs(DATA_BASE, exist_ok=True)

    # Remove problematic nodes from volume
    print("Removing problematic custom nodes...")
    for node_name in PROBLEMATIC_NODES:
        node_path = os.path.join(CUSTOM_NODES_DIR, node_name)
        if os.path.exists(node_path):
            print(f"Removing {node_name}...")
            shutil.rmtree(node_path)
            print(f"Removed {node_name}")

    # Fix detached HEAD and update ComfyUI backend
    print("Fixing git branch and updating ComfyUI backend...")
    os.chdir(DATA_BASE)
    try:
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Detected detached HEAD, checking out master branch...")
            subprocess.run("git checkout -B master origin/master", shell=True, check=True, capture_output=True, text=True)
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")

    # Sync custom nodes from image to volume (ensure all nodes exist)
    print("Syncing custom nodes from image to volume...")
    image_nodes_dir = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes")
    if os.path.exists(image_nodes_dir):
        for node_name in os.listdir(image_nodes_dir):
            src = os.path.join(image_nodes_dir, node_name)
            dst = os.path.join(CUSTOM_NODES_DIR, node_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                print(f"Copying missing node: {node_name}")
                shutil.copytree(src, dst)

    # Configure Manager
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    legacy_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")

    if os.path.exists(legacy_dir):
        print("Migrating Manager data from legacy path...")
        os.makedirs(manager_config_dir, exist_ok=True)
        shutil.copytree(legacy_dir, manager_config_dir, dirs_exist_ok=True)
        shutil.rmtree(legacy_dir)
        print("Migration completed")

    backup_dir = os.path.join(manager_config_dir, ".legacy-manager-backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    print("Configuring ComfyUI-Manager...")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)

    # Update ComfyUI-Manager
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager...")
        os.chdir(manager_dir)
        try:
            subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
            subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error updating Manager: {e.stderr}")
        os.chdir(DATA_BASE)
    else:
        print("Installing ComfyUI-Manager...")
        subprocess.run("comfy --skip-prompt node install ComfyUI-Manager", shell=True, check=False)

    # Upgrade pip and comfy-cli
    print("Upgrading pip...")
    subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=False, capture_output=True)
    
    print("Upgrading comfy-cli...")
    subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=False, capture_output=True)

    # Install ComfyUI requirements
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(f"pip install -r {requirements_path}", shell=True, check=False, capture_output=True)
    
    # Install Manager requirements
    manager_requirements_path = os.path.join(DATA_BASE, "manager_requirements.txt")
    if os.path.exists(manager_requirements_path):
        subprocess.run(f"pip install -r {manager_requirements_path}", shell=True, check=False, capture_output=True)

    # Ensure directories
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Download models
    print("Checking and downloading missing models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn}...")
            try:
                hf_download(sub, fn, repo, subf)
                print(f"Downloaded {fn}")
            except Exception as e:
                print(f"Error downloading {fn}: {e}")
        else:
            print(f"{fn} already exists")

    # Extra downloads
    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True)

    # Commit volume changes
    print("Committing volume changes...")
    vol.commit()

    # Launch ComfyUI
    os.environ["COMFY_DIR"] = DATA_BASE
    print(f"Starting ComfyUI from {DATA_BASE}...")
    
    # Setup comfy-cli config untuk disable tracking
    comfy_config_dir = os.path.expanduser("~/.config/comfy-cli")
    os.makedirs(comfy_config_dir, exist_ok=True)
    with open(os.path.join(comfy_config_dir, "config.toml"), "w") as f:
        f.write("tracking_enabled = false\n")
    
    # Launch dengan --enable-manager flag
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--enable-manager"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.Popen(cmd, cwd=DATA_BASE, env=os.environ.copy())
