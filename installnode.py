"""
Script to install missing custom nodes to the volume
"""
import modal

VOLUME_NAME = "comfyui-app"
VOL_NODES = "/data/ComfyUI/custom_nodes"

vol = modal.Volume.from_name(VOLUME_NAME)

# Image with git installed
image = modal.Image.debian_slim(python_version="3.11").apt_install("git")

app = modal.App("install-nodes")

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
)
def install_missing_nodes():
    import os
    import subprocess
    
    os.makedirs(VOL_NODES, exist_ok=True)
    os.chdir(VOL_NODES)
    
    nodes_to_install = [
        {
            "name": "Comfyui_TTP_Toolset",
            "url": "https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git",
            "provides": "LTXVFirstLastFrameControl_TTP"
        },
        {
            "name": "ComfyUI_LTX-2_VRAM_Memory_Management",
            "url": "https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management.git",
            "provides": "TensorParallelV3Node"
        }
    ]
    
    for node in nodes_to_install:
        node_path = f"{VOL_NODES}/{node['name']}"
        if not os.path.exists(node_path):
            print(f"Installing {node['name']} (provides: {node['provides']})...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                node["url"],
                node_path
            ], check=True)
            print(f"  ✓ {node['name']} installed!")
        else:
            print(f"  ✓ {node['name']} already exists")
    
    # List all nodes
    print("\n📦 Custom nodes in volume:")
    nodes = sorted(os.listdir(VOL_NODES))
    for item in nodes:
        if os.path.isdir(f"{VOL_NODES}/{item}"):
            print(f"  - {item}")
    
    print(f"\nTotal: {len(nodes)} custom nodes")
    
    # Commit changes
    vol.commit()
    print("\n✅ Volume committed!")

@app.local_entrypoint()
def main():
    install_missing_nodes.remote()
