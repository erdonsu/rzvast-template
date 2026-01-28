import os
import modal
import shutil
import subprocess

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")

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

# Build image
image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "COMFY_TRACKING_ENABLED": "0",
        "COMFYUI_TRACKING": "0"
    })
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "git clone https://github.com/Comfy-Org/ComfyUI.git /root/comfy/ComfyUI",
        "cd /root/comfy/ComfyUI && pip install -r requirements.txt",
        "pip install ftfy accelerate einops diffusers sentencepiece sageattention onnx onnxruntime onnxruntime-gpu"
    ])
)

# Git-based nodes di build
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Custom nodes untuk install di runtime
GIT_NODES = [
    ("rgthree/rgthree-comfy", True, False),
    ("ltdrdata/ComfyUI-Impact-Pack", False, True),
    ("ltdrdata/ComfyUI-Impact-Subpack", False, True),
    ("Acly/comfyui-tooling-nodes", False, True),
    ("ltdrdata/ComfyUI-Inspire-Pack", False, True),
    ("cubiq/ComfyUI_IPAdapter_plus", False, True),
    ("WASasquatch/was-node-suite-comfyui", False, True),
    ("RockOfFire/ComfyUI_Comfyroll_CustomNodes", False, True),
    ("cubiq/ComfyUI_essentials", False, True),
    ("city96/ComfyUI-GGUF", False, True),
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
    # First run setup
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            os.makedirs(DATA_BASE, exist_ok=True)

    # Install custom nodes
    nodes_installed_flag = os.path.join(DATA_BASE, ".nodes_installed")
    if not os.path.exists(nodes_installed_flag):
        print("Installing custom nodes...")
        os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
        
        for repo, recursive, install_reqs in GIT_NODES:
            name = repo.split("/")[-1]
            dest = os.path.join(CUSTOM_NODES_DIR, name)
            
            if os.path.exists(dest):
                print(f"✓ {name} exists")
                continue
                
            print(f"Installing {name}...")
            try:
                cmd = f"git clone https://github.com/{repo} {dest}"
                if recursive:
                    cmd += " --recursive"
                
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print(f"✓ {name} cloned")
                
                if install_reqs:
                    req_file = os.path.join(dest, "requirements.txt")
                    if os.path.exists(req_file):
                        subprocess.run(f"pip install -r {req_file}", shell=True, check=True, capture_output=True, text=True)
                        print(f"✓ {name} requirements installed")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {name}")
        
        with open(nodes_installed_flag, "w") as f:
            f.write("installed")
        print("Custom nodes installed!")

    # Update ComfyUI
    print("Updating ComfyUI...")
    os.chdir(DATA_BASE)
    try:
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run("git checkout -B master origin/master", shell=True, check=True, capture_output=True, text=True)
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
    except:
        pass

    # Configure Manager
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\n"
    with open(os.path.join(manager_config_dir, "config.ini"), "w") as f:
        f.write(config_content)

   # ... (semua kode sebelumnya tetap sama sampai Configure Manager)

    # Ensure directories
    for d in [CUSTOM_NODES_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Check models
    print("Checking models...")
    model_dirs = ["checkpoints", "loras", "upscale_models"]
    models_ok = True
    for md in model_dirs:
        full_path = os.path.join(MODELS_DIR, md)
        if os.path.exists(full_path) and os.listdir(full_path):
            print(f"✓ {md} ready")
        else:
            print(f"⚠️  {md} empty")
            models_ok = False
    
    if not models_ok:
        print("\n⚠️  Some models missing! Run: modal run download_models.py")

    # Launch ComfyUI directly via Python (bypass comfy-cli)
    print(f"🚀 Starting ComfyUI from {DATA_BASE}...")
    os.chdir(DATA_BASE)
    
    env = os.environ.copy()
    env["COMFYUI_TRACKING"] = "0"
    
    cmd = [
        "python", "main.py",
        "--listen", "0.0.0.0",
        "--port", "8000"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.Popen(cmd, cwd=DATA_BASE, env=env)
