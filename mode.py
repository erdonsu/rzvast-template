import os
import modal
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

DATA_ROOT = "/data/comfy"
MODELS_DIR = os.path.join(DATA_ROOT, "ComfyUI", "models")
TMP_DL = "/tmp/download"

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    """Download file dari HuggingFace ke volume"""
    print(f"  📥 Downloading from {repo_id}/{subfolder or ''}/{filename}")
    out = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        subfolder=subfolder, 
        local_dir=TMP_DL
    )
    target_dir = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, filename)
    shutil.move(out, target_file)
    print(f"  ✅ Saved to {target_file}")

# Image minimal untuk download
image = (
    modal.Image.debian_slim()
    .apt_install("wget")
    .pip_install("huggingface_hub[hf_transfer]==0.28.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="download-comfyui-models", image=image)

# Model list
# Model list - HAPUS baris vae/LTX yang error
MODEL_TASKS = [
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    # HAPUS baris ini karena path salah/404:
    # ("vae/LTX", "audio_vae.safetensors", "Lightricks/LTX-2", "audio_vae"),
]

@app.function(
    timeout=3600,  # 1 hour
    volumes={DATA_ROOT: vol},
)
def download_models():
    """Download semua model ke Modal volume"""
    print("=" * 70)
    print("🚀 Starting Model Download to Modal Volume")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    # Download HuggingFace models
    for sub, fn, repo, subf in MODEL_TASKS:
        target = os.path.join(MODELS_DIR, sub, fn)
        
        print(f"\n📦 Model: {fn}")
        
        if os.path.exists(target):
            print(f"  ⏭️  Already exists, skipping")
            skipped += 1
            continue
        
        try:
            hf_download(sub, fn, repo, subf)
            vol.commit()  # Commit after each file
            downloaded += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    # Download RealESRGAN upscale model
    print(f"\n📦 Model: RealESRGAN_x4plus_anime_6B.pth")
    upscale_dir = os.path.join(MODELS_DIR, "upscale_models")
    os.makedirs(upscale_dir, exist_ok=True)
    upscale_file = os.path.join(upscale_dir, "RealESRGAN_x4plus_anime_6B.pth")
    
    if os.path.exists(upscale_file):
        print(f"  ⏭️  Already exists, skipping")
        skipped += 1
    else:
        try:
            print(f"  📥 Downloading from GitHub releases...")
            result = subprocess.run(
                f"wget -q --show-progress https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {upscale_dir}",
                shell=True, 
                check=True,
                capture_output=False
            )
            vol.commit()
            print(f"  ✅ Saved to {upscale_file}")
            downloaded += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Download Summary:")
    print(f"  ✅ Downloaded: {downloaded}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    print("=" * 70)
    
    if failed == 0:
        print("🎉 All models ready!")
    else:
        print("⚠️  Some models failed. Check logs above.")

@app.local_entrypoint()
def main():
    """Entry point saat run dari local/Colab"""
    download_models.remote()
