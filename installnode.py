import os
import modal
import subprocess
# PATH YANG BENAR - langsung /data/ComfyUI bukan /data/comfy/ComfyUI
DATA_ROOT = "/data"
CUSTOM_NODES_DIR = "/data/ComfyUI/custom_nodes"
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
    
    # Verify
    print("\nVerifying installation:")
    for name, _ in CUSTOM_NODES:
        path = os.path.join(CUSTOM_NODES_DIR, name)
        if os.path.exists(path):
            print(f"  [FOUND] {name}")
        else:
            print(f"  [MISSING] {name}")
@app.local_entrypoint()
def main():
    install_nodes.remote()
