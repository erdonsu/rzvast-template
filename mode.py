import os
import modal
import shutil
from typing import Optional
from huggingface_hub import hf_hub_download

DATA_ROOT = "/data/comfy"
MODELS_DIR = os.path.join(DATA_ROOT, "ComfyUI", "models")
TMP_DL = "/tmp/download"

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

# Image minimal untuk download saja
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.28.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui-download-models", image=image)

model_tasks = [
    ("checkpoints", "ltx-2-spatial-upscaler-x2-1.0.safetensors", "Lightricks/LTX-2", None),
    ("checkpoints", "ltx-2-19b-dev.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-distilled-lora-384.safetensors", "Lightricks/LTX-2", None),
    ("loras", "ltx-2-19b-ic-lora-canny-control.safetensors", "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", None),
    ("vae/LTX", "audio_vae.safetensors", "Lightricks/LTX-2", "audio_vae"),
]

@app.function(
    timeout=3600,  # 1 jam untuk download
    volumes={DATA_ROOT: vol},
)
def download_all_models():
    """Download semua model ke volume"""
    print("Starting model downloads...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if os.path.exists(target):
            print(f"✓ {fn} already exists, skipping")
            continue
            
        print(f"📥 Downloading {fn} from {repo}...")
        try:
            hf_download(sub, fn, repo, subf)
            vol.commit()  # Commit setiap file selesai
            print(f"✅ Downloaded {fn}")
        except Exception as e:
            print(f"❌ Error downloading {fn}: {e}")
    
    # Download upscale model
    import subprocess
    upscale_dir = os.path.join(MODELS_DIR, "upscale_models")
    os.makedirs(upscale_dir, exist_ok=True)
    upscale_file = os.path.join(upscale_dir, "RealESRGAN_x4plus_anime_6B.pth")
    
    if not os.path.exists(upscale_file):
        print("📥 Downloading RealESRGAN model...")
        subprocess.run(
            f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {upscale_dir}",
            shell=True, check=True
        )
        vol.commit()
        print("✅ Downloaded RealESRGAN model")
    
    print("🎉 All models downloaded!")

@app.local_entrypoint()
def main():
    """Jalankan download dari local machine"""
    download_all_models.remote()
