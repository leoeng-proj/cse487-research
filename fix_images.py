from PIL import ImageFile
from PIL import Image
import os

source_folder = "/data/add_disk0/leeng/rails/train"
output_folder = "/data/add_disk0/leeng/rails/train_repaired"

fixed = 0
skipped = 0

ImageFile.LOAD_TRUNCATED_IMAGES = True

for root, _, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg")):

            path = os.path.join(root, file)

            # Preserve the directory structure
            relative = os.path.relpath(path, source_folder)
            output_path = os.path.join(output_folder, relative)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                img = Image.open(path)
                img.load()

                # Re-save as a fresh JPEG
                img.save(output_path, format="JPEG", quality=95)

                fixed += 1

            except Exception as e:
                print(f"Skipped {path}: {type(e).__name__} — {e}")
                skipped += 1

print(f"\nFixed/re-encoded: {fixed} image(s)")
print(f"Skipped: {skipped} image(s)")