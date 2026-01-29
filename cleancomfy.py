"""
Cleanup ComfyUI Volume
======================
Hapus ComfyUI core lama dari volume, preserve models dan user data.
Jalankan ini SEBELUM deploy comfyui_app.py

Usage:
  set MODAL_TOKEN_ID=xxx
  set MODAL_TOKEN_SECRET=xxx
  modal run cleanup_volume.py
"""

import modal

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="cleanup-comfyui")

# Minimal image untuk cleanup
image = modal.Image.debian_slim().pip_install("rich")

@app.function(
    image=image,
    volumes={"/data/comfy": vol},
    timeout=300,
)
def cleanup():
    import os
    import shutil
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    DATA_BASE = "/data/comfy/ComfyUI"
    
    # Directories yang PRESERVE (user data & models)
    PRESERVE = ["models", "input", "output", "user"]
    
    # Directories yang HAPUS (core ComfyUI - akan di-sync ulang dari image baru)
    DELETE_DIRS = [
        "comfy", "comfy_extras", "app", "api_server", 
        "notebooks", "script_examples", "tests", "web",
        ".git"  # Hapus git juga supaya fresh
    ]
    
    # Files yang HAPUS
    DELETE_FILES = [
        "main.py", "nodes.py", "server.py", "requirements.txt",
        "pyproject.toml", "README.md", "LICENSE", ".gitignore"
    ]
    
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       COMFYUI VOLUME CLEANUP SCRIPT       [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    if not os.path.exists(DATA_BASE):
        console.print("[yellow]Volume kosong, tidak ada yang perlu di-cleanup.[/yellow]")
        return
    
    # Show current state
    console.print("[bold]Current volume contents:[/bold]")
    for item in sorted(os.listdir(DATA_BASE)):
        path = os.path.join(DATA_BASE, item)
        if os.path.isdir(path):
            console.print(f"  📁 {item}/")
        else:
            console.print(f"  📄 {item}")
    
    console.print("\n[bold green]PRESERVING (tidak dihapus):[/bold green]")
    for p in PRESERVE:
        path = os.path.join(DATA_BASE, p)
        if os.path.exists(path):
            if p == "models":
                # Count models
                model_count = sum(len(files) for _, _, files in os.walk(path))
                console.print(f"  ✅ {p}/ ({model_count} files)")
            else:
                console.print(f"  ✅ {p}/")
    
    console.print("\n[bold red]DELETING (akan dihapus):[/bold red]")
    
    deleted_dirs = []
    deleted_files = []
    
    # Delete directories
    for d in DELETE_DIRS:
        path = os.path.join(DATA_BASE, d)
        if os.path.exists(path):
            console.print(f"  🗑️  {d}/")
            shutil.rmtree(path)
            deleted_dirs.append(d)
    
    # Delete files
    for f in DELETE_FILES:
        path = os.path.join(DATA_BASE, f)
        if os.path.exists(path):
            console.print(f"  🗑️  {f}")
            os.remove(path)
            deleted_files.append(f)
    
    # Also clean custom_nodes that might have issues (optional - keep nodes)
    # We preserve custom_nodes but can optionally clean problematic ones
    problematic_nodes = [
        "ComfyUI_HFDownLoad",
        "hf-model-downloader", 
        "comfyui_hf_model_downloader",
        "comfyui-model-downloader",
        "comfyui-doctor",
    ]
    
    custom_nodes_dir = os.path.join(DATA_BASE, "custom_nodes")
    if os.path.exists(custom_nodes_dir):
        console.print("\n[bold yellow]CLEANING problematic nodes:[/bold yellow]")
        for node in problematic_nodes:
            node_path = os.path.join(custom_nodes_dir, node)
            if os.path.exists(node_path):
                console.print(f"  🗑️  custom_nodes/{node}")
                shutil.rmtree(node_path)
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold green]CLEANUP COMPLETE![/bold green]")
    console.print(f"  Deleted {len(deleted_dirs)} directories")
    console.print(f"  Deleted {len(deleted_files)} files")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    
    console.print("\n[bold]Remaining in volume:[/bold]")
    if os.path.exists(DATA_BASE):
        for item in sorted(os.listdir(DATA_BASE)):
            path = os.path.join(DATA_BASE, item)
            if os.path.isdir(path):
                console.print(f"  📁 {item}/")
            else:
                console.print(f"  📄 {item}")
    
    console.print("\n[bold yellow]Next step:[/bold yellow]")
    console.print("  modal deploy comfyui_app.py")
    console.print("\nVolume akan di-populate dengan ComfyUI fresh dari image.\n")

@app.local_entrypoint()
def main():
    cleanup.remote()
