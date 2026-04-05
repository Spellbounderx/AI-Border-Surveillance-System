import zipfile
import os
import shutil

folder = r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model2"
output = r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model2_fixed.pt"

print("Repacking model into valid .pt zip file...")

with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as zf:
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            # Archive path must be relative to the parent of the folder
            arcname = os.path.relpath(full_path, os.path.dirname(folder))
            zf.write(full_path, arcname)
            print(f"  Added: {arcname}")

print(f"\nDone! Saved to: {output}")

# Verify it loads
import torch
ckpt = torch.load(output, map_location="cpu", weights_only=False)
print("Load successful! Keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))