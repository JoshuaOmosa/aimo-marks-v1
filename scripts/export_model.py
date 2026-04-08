"""
Export Model Script
Creates a zip file of your trained model for sharing or backup
"""

import os
import shutil
import zipfile
from pathlib import Path

def export_model(model_path="./marks_v1_model", output_name="aimo_marks_v1_export"):
    """
    Export the trained model to a zip file
    
    Args:
        model_path: Path to the trained model folder
        output_name: Name of the output zip file (without extension)
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model folder not found at {model_path}")
        print("   Make sure you have trained the model first!")
        return False
    
    # Required files
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]
    
    # Check for required files
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Create zip
    zip_path = f"{output_name}.zip"
    
    print(f"📦 Exporting model from {model_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                # Skip checkpoint folders (they're large and not needed)
                if "checkpoint" in root or "checkpoint" in file:
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                zipf.write(file_path, arcname)
                print(f"   Added: {arcname}")
    
    # Show results
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"\n✅ Export complete!")
    print(f"📁 File: {zip_path}")
    print(f"📦 Size: {size_mb:.2f} MB")
    print(f"\n📍 Location: {os.path.abspath(zip_path)}")
    
    return True

def list_model_contents(model_path="./marks_v1_model"):
    """List all files in the model folder"""
    if not os.path.exists(model_path):
        print(f"❌ Model folder not found: {model_path}")
        return
    
    print(f"\n📁 Contents of {model_path}:")
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path):
            print(f"   📂 {item}/")
        else:
            size = os.path.getsize(item_path) / 1024
            print(f"   📄 {item} ({size:.0f} KB)")

if __name__ == "__main__":
    # First, show what's in the model folder
    list_model_contents()
    
    print("\n" + "="*50)
    
    # Export the model
    export_model()
    
    print("\n" + "="*50)
    print("💡 To share this model:")
    print("   1. Upload the .zip file to Google Drive or HuggingFace")
    print("   2. Share the link with others")
    print("   3. They can extract it to ./marks_v1_model/")