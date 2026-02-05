import os
import modal
import subprocess

DATA_ROOT = "/data/comfy"
CUSTOM_NODES_DIR = os.path.join(DATA_ROOT, "ComfyUI", "custom_nodes")

# Image dengan git untuk clone repos
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("gitpython")
)

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="install-custom-nodes", image=image)

# Custom nodes yang perlu diinstall
CUSTOM_NODES = [
    {
        "name": "ComfyUI-VideoHelperSuite",
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "provides": ["VHS_VideoCombine", "VHS_LoadVideo", "VHS_SplitVideo"]
    },
    {
        "name": "ComfyUI-Custom-Scripts",
        "url": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git",
        "provides": ["MathExpression|pysssss", "ShowText|pysssss"]
    },
]

@app.function(
    timeout=1800,  # 30 minutes
    volumes={DATA_ROOT: vol},
)
def install_nodes():
    """Install custom nodes ke Modal volume"""
    print("=" * 70)
    print("Installing Custom Nodes to Modal Volume")
    print("=" * 70)
    
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
    
    installed = 0
    skipped = 0
    failed = 0
    
    for node in CUSTOM_NODES:
        name = node["name"]
        url = node["url"]
        target_dir = os.path.join(CUSTOM_NODES_DIR, name)
        
        print(f"\n[NODE] {name}")
        print(f"  URL: {url}")
        print(f"  Provides: {', '.join(node['provides'])}")
        
        if os.path.exists(target_dir):
            print(f"  [SKIP] Already exists")
            skipped += 1
            continue
        
        try:
            # Git clone
            print(f"  [CLONE] Cloning repository...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, target_dir],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  [OK] Cloned successfully")
            
            # Check for requirements.txt
            req_file = os.path.join(target_dir, "requirements.txt")
            if os.path.exists(req_file):
                print(f"  [INFO] Found requirements.txt (will be installed at runtime)")
            
            vol.commit()
            installed += 1
            print(f"  [DONE] {name} installed")
            
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Git clone failed: {e.stderr}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    
    # List all custom nodes
    print("\n" + "=" * 70)
    print("All Custom Nodes in Volume:")
    print("=" * 70)
    
    if os.path.exists(CUSTOM_NODES_DIR):
        for item in sorted(os.listdir(CUSTOM_NODES_DIR)):
            item_path = os.path.join(CUSTOM_NODES_DIR, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                print(f"  - {item}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Installation Summary:")
    print(f"  Installed: {installed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print("=" * 70)
    
    if failed == 0:
        print("All custom nodes ready!")
        print("\nNOTE: Restart ComfyUI to load new nodes.")
    else:
        print("Some nodes failed. Check logs above.")

@app.local_entrypoint()
def main():
    """Entry point"""
    install_nodes.remote()
