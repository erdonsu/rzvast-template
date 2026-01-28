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

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd
  
def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # dependencies install for some nodes
    .run_commands([
        "pip install ftfy accelerate einops diffusers sentencepiece sageattention onnx onnxruntime onnxruntime-gpu"
    ])
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy --skip-prompt node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF ComfyUI-LTXVideo"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Model download tasks (will be done at runtime)
model_tasks = [
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    # HAPUS baris ComfyUI-LTXVideo dari sini - itu custom node, bukan model!
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# Create volume
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

    # Update frontend
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(f"pip install -r {requirements_path}", shell=True, check=False, capture_output=True)

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
                print(f"✓ Downloaded {fn}")
            except Exception as e:
                print(f"✗ Error downloading {fn}: {e}")
        else:
            print(f"✓ {fn} already exists")

    # Extra downloads
    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True)

    # Launch ComfyUI
    os.environ["COMFY_DIR"] = DATA_BASE
    print(f"Starting ComfyUI from {DATA_BASE}...")
    
    # Setup comfy-cli config untuk disable tracking
    comfy_config_dir = os.path.expanduser("~/.config/comfy-cli")
    os.makedirs(comfy_config_dir, exist_ok=True)
    with open(os.path.join(comfy_config_dir, "config.toml"), "w") as f:
        f.write("tracking_enabled = false\n")
    
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--enable-manager"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.Popen(cmd, cwd=DATA_BASE, env=os.environ.copy())
