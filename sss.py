import os
import modal
import subprocess
DATA_ROOT = "/data"
MODELS_DIR = "/data/ComfyUI/models"
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "curl")
    .pip_install("huggingface_hub[hf_transfer]==0.28.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="download-ltx-upscaler", image=image)
@app.function(timeout=3600, volumes={DATA_ROOT: vol})
def download_model():
    from huggingface_hub import hf_hub_download
    import shutil
    
    print("=" * 60)
    print("Downloading LTX-2 Spatial Upscaler")
    print("=" * 60)
    
    filename = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    
    # Check multiple possible locations
    possible_dirs = [
        f"{MODELS_DIR}/checkpoints",
        f"{MODELS_DIR}/upscale_models",
        f"{MODELS_DIR}/latent_upscale_models",
    ]
    
    # Check if already exists
    for dir_path in possible_dirs:
        target = os.path.join(dir_path, filename)
        if os.path.exists(target):
            size_gb = os.path.getsize(target) / (1024**3)
            print(f"[SKIP] Already exists: {target} ({size_gb:.2f} GB)")
            return
    
    # Download to upscale_models
    target_dir = f"{MODELS_DIR}/upscale_models"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, filename)
    
    print(f"\n[DOWNLOAD] {filename}")
    print(f"  From: Lightricks/LTX-2")
    print(f"  To: {target_file}")
    
    try:
        # Download using huggingface_hub
        downloaded = hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename=filename,
            local_dir="/tmp/download"
        )
        
        # Move to target
        shutil.move(downloaded, target_file)
        
        size_gb = os.path.getsize(target_file) / (1024**3)
        print(f"\n[OK] Downloaded: {filename} ({size_gb:.2f} GB)")
        
        vol.commit()
        print("[OK] Committed to volume")
        
    except Exception as e:
        print(f"[ERROR] {e}")
@app.local_entrypoint()
def main():
    download_model.remote()
