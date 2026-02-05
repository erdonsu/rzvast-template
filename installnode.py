import os
import modal
import subprocess
DATA_ROOT = "/data/comfy"
CUSTOM_NODES_DIR = os.path.join(DATA_ROOT, "ComfyUI", "custom_nodes")
image = modal.Image.debian_slim().apt_install("git")
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="install-custom-nodes", image=image)
CUSTOM_NODES = [
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("ComfyUI-Custom-Scripts", "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"),
]
@app.function(timeout=1800, volumes={DATA_ROOT: vol})
def install_nodes():
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
    
    for name, url in CUSTOM_NODES:
        target_dir = os.path.join(CUSTOM_NODES_DIR, name)
        print(f"[NODE] {name}")
        
        if os.path.exists(target_dir):
            print(f"  [SKIP] Already exists")
            continue
        
        try:
            subprocess.run(["git", "clone", "--depth", "1", url, target_dir], check=True)
            vol.commit()
            print(f"  [OK] Installed")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\nDone!")
@app.local_entrypoint()
def main():
    install_nodes.remote()
