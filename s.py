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

# Build image dengan HANYA dependencies dasar
image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "git clone https://github.com/Comfy-Org/ComfyUI.git /root/comfy/ComfyUI",
        "cd /root/comfy/ComfyUI && pip install -r requirements.txt",
        "pip install ftfy accelerate einops diffusers sentencepiece sageattention onnx onnxruntime onnxruntime-gpu",
        # PENTING: Set comfy-cli config untuk skip tracking prompt
        "comfy tracking disable"  # ← TAMBAHKAN INI
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# HAPUS bagian ini - pindahkan ke runtime
# image = image.run_commands([
#     "comfy node install rgthree-comfy ..."
# ])

# Git-based nodes - install via git clone saja
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Model download tasks
model_tasks = [
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", "main"),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", "main"),
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", "main"),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    ("vae/LTX", "audio_vae.safetensors", "Lightricks/LTX-2", "main/audio_vae"),
    ("text_encoder/LTX", "gemma-3-12b-it-qat-q4_0.gguf", "google/gemma-3-12b-it-qat-q4_0-unquantized", None),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# Nodes yang akan diinstall via comfy-cli di runtime
COMFY_NODES = [
    "rgthree-comfy",
    "comfyui-impact-pack", 
    "comfyui-impact-subpack",
    "ComfyUI-YOLO",
    "comfyui-inspire-pack",
    "comfyui_ipadapter_plus",
    "wlsh_nodes",
    "ComfyUI_Comfyroll_CustomNodes",
    "comfyui_essentials",
    "ComfyUI-GGUF"
]

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

    # Install custom nodes via comfy-cli di RUNTIME (hanya sekali)
    nodes_installed_flag = os.path.join(DATA_BASE, ".nodes_installed")
    if not os.path.exists(nodes_installed_flag):
        print("Installing custom nodes via comfy-cli...")
        os.chdir(DATA_BASE)
        os.environ["COMFY_DIR"] = DATA_BASE
        
        for node in COMFY_NODES:
            print(f"Installing {node}...")
            try:
                result = subprocess.run(
                    f"comfy node install {node}",
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=DATA_BASE
                )
                print(f"✓ {node} installed: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {node}: {e.stderr}")
                # Lanjutkan ke node berikutnya
                continue
        
        # Tandai bahwa nodes sudah diinstall
        with open(nodes_installed_flag, "w") as f:
            f.write("installed")
        print("Custom nodes installation completed!")

    # Fix detached HEAD dan update ComfyUI
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

    # [Sisa kode tetap sama...]
    
    # Configure Manager
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)

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

    # Extra downloads
    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True)

    # Launch ComfyUI
    os.environ["COMFY_DIR"] = DATA_BASE
    print(f"Starting ComfyUI from {DATA_BASE}...")
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--enable-manager"]
    subprocess.Popen(cmd, cwd=DATA_BASE, env=os.environ.copy())
