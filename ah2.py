"""
ComfyUI Modal Deployment - Based on MxC (Modified for LTX-2)
============================================================
Original: https://github.com/Renks/MxC
Modified: Added LTX-2 nodes and dependencies
"""

import subprocess
from pathlib import Path
import modal
import os

# ===========================
# Configuration (Simplified)
# ===========================

APP_NAME = "comfyui"
VOLUME_NAME = "comfyui-app"
VOLUME_MOUNT = "/data"
COMFYUI_DIR = "/root/comfy/ComfyUI"
CUSTOM_NODES_DIR = f"{COMFYUI_DIR}/custom_nodes"
OUTPUT_DIR = f"{VOLUME_MOUNT}/output"
MODELS_DIR = f"{VOLUME_MOUNT}/models"

# GPU Type - change as needed
GPU_TYPE = os.environ.get('MODAL_GPU_TYPE', 'A100-40GB')

# Web server
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# ===========================
# Modal Image Configuration
# ===========================

comfy_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "ffmpeg",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1"
    )
    .pip_install(
        "comfy-cli",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    # Install ComfyUI via comfy-cli
    .run_commands("comfy --skip-prompt install --nvidia")
    
    # =========================================================================
    # Core Dependencies for LTX-2 and Audio Nodes
    # =========================================================================
    .pip_install(
        "gguf",
        "sentencepiece",
        "opencv-python-headless",
        "soundfile",
        "librosa",
        "scipy",
        "einops",
        "accelerate",
        "diffusers",
        "transformers>=4.50.0",
        "ftfy",
    )
    
    # =========================================================================
    # Install Custom Nodes via comfy-cli (Registry)
    # =========================================================================
    .run_commands(
        "comfy node install ComfyUI-Manager",
        "comfy node install ComfyUI-LTXVideo",       # LTX-2 Video nodes
        "comfy node install ComfyUI-VideoHelperSuite", # VHS_VideoCombine
        "comfy node install ComfyUI-Custom-Scripts",  # MathExpression|pysssss
        "comfy node install ComfyMath",               # CM_FloatToInt
        "comfy node install comfyui-easy-use",
        "comfy node install comfyui-kjnodes",
        "comfy node install rgthree-comfy",
        "comfy node install comfyui_essentials",
        "comfy node install ComfyUI-GGUF",
        "comfy node install comfyui-impact-pack",
        "comfy node install comfyui-inspire-pack",
    )
    
    # =========================================================================
    # Git Clone Nodes (Not in registry or need latest version)
    # =========================================================================
    .run_commands(
        # TTP Toolset - untuk LTXVFirstLastFrameControl_TTP
        f"git clone https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git {CUSTOM_NODES_DIR}/Comfyui_TTP_Toolset",
        f"pip install -r {CUSTOM_NODES_DIR}/Comfyui_TTP_Toolset/requirements.txt || true",
        
        # LTX-2 VRAM Memory Management - untuk TensorParallelV3Node
        f"git clone https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management.git {CUSTOM_NODES_DIR}/ComfyUI_LTX-2_VRAM_Memory_Management",
        
        # Ultimate SD Upscale
        f"git clone --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git {CUSTOM_NODES_DIR}/ComfyUI_UltimateSDUpscale",
        
        # Qwen nodes
        f"git clone https://github.com/luguoli/ComfyUI-Qwen-Image-Integrated-KSampler.git {CUSTOM_NODES_DIR}/ComfyUI-Qwen-Image-Integrated-KSampler",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-Qwen-Image-Integrated-KSampler/requirements.txt || true",
        
        f"git clone https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git {CUSTOM_NODES_DIR}/ComfyUI-qwenmultiangle",
    )
    
    # =========================================================================
    # Install Requirements for Installed Nodes
    # =========================================================================
    .run_commands(
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-Manager/requirements.txt || true",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-LTXVideo/requirements.txt || true",
        f"pip install -r {CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite/requirements.txt || true",
    )
)

# ===========================
# Modal App Configuration
# ===========================

app = modal.App(name=APP_NAME, image=comfy_image)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.cls(
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: model_volume},
)
@modal.concurrent(max_inputs=10)
class ComfyUIContainer:
    
    @modal.enter()
    def setup(self):
        """Setup on container start"""
        from huggingface_hub import hf_hub_download
        import shutil
        
        # Create directories
        dirs = [
            f"{MODELS_DIR}/checkpoints",
            f"{MODELS_DIR}/loras", 
            f"{MODELS_DIR}/vae",
            f"{MODELS_DIR}/clip",
            f"{MODELS_DIR}/text_encoders",
            f"{MODELS_DIR}/diffusion_models",
            f"{MODELS_DIR}/upscale_models",
            f"{VOLUME_MOUNT}/custom_nodes",
            OUTPUT_DIR,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # Sync custom nodes from volume to ComfyUI
        vol_nodes = Path(f"{VOLUME_MOUNT}/custom_nodes")
        if vol_nodes.exists():
            for node in vol_nodes.iterdir():
                if node.is_dir():
                    target = Path(CUSTOM_NODES_DIR) / node.name
                    if not target.exists():
                        print(f"Syncing node from volume: {node.name}")
                        shutil.copytree(node, target)
        
        # Install requirements for volume nodes
        nodes_path = Path(CUSTOM_NODES_DIR)
        if nodes_path.exists():
            print("Checking custom node requirements...")
            for node_dir in nodes_path.iterdir():
                if node_dir.is_dir():
                    req_file = node_dir / "requirements.txt"
                    if req_file.exists():
                        subprocess.run(
                            ["pip", "install", "-q", "-r", str(req_file)], 
                            check=False
                        )
        
        # Download essential models if not exist
        model_tasks = [
            ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2"),
            ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2"),
            ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2"),
            ("text_encoders", "gemma_3_12B_it_fp8_e4m3fn.safetensors", "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn"),
        ]
        
        print("Checking models...")
        for subdir, filename, repo in model_tasks:
            target = Path(MODELS_DIR) / subdir / filename
            if not target.exists():
                print(f"Downloading {filename}...")
                try:
                    downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir="/tmp/hf")
                    shutil.move(downloaded, str(target))
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
        
        # Create extra_model_paths.yaml for volume models
        extra_paths = f"""
comfyui:
    base_path: {VOLUME_MOUNT}/models
    checkpoints: checkpoints/
    loras: loras/
    vae: vae/
    clip: clip/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    upscale_models: upscale_models/
"""
        extra_path_file = Path(COMFYUI_DIR) / "extra_model_paths.yaml"
        extra_path_file.write_text(extra_paths)
        
        # Configure Manager
        manager_config = Path(COMFYUI_DIR) / "user" / "__manager"
        manager_config.mkdir(parents=True, exist_ok=True)
        (manager_config / "config.ini").write_text(
            "[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\n"
        )
        
        # Commit volume changes
        model_volume.commit()
        print("Setup complete!")

    @modal.web_server(WEB_PORT, startup_timeout=120)
    def ui(self):
        """Launch ComfyUI web server"""
        print(f"Starting ComfyUI on {WEB_HOST}:{WEB_PORT}...")
        subprocess.Popen(
            f"comfy launch -- --output-directory {OUTPUT_DIR} --listen {WEB_HOST} --port {WEB_PORT}",
            shell=True
        )
