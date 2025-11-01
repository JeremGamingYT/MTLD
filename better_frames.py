# dedupe_frames.py
from pathlib import Path
from PIL import Image, ImageChops
import os

DATASET_DIR = Path("animes_dataset")
# plus c'est petit, plus c'est strict (0.0 = rien ne passe, 10 = cool)
DIFF_THRESHOLD = 8  # à ajuster

def is_similar(img1, img2, threshold=DIFF_THRESHOLD):
    # diff pixel à pixel
    diff = ImageChops.difference(img1, img2).convert("L")
    # on prend la moyenne
    hist = diff.histogram()
    # calcule une sorte de "énergie" moyenne
    total_pixels = img1.size[0] * img1.size[1]
    diff_sum = sum(i * hist[i] for i in range(256))
    avg_diff = diff_sum / total_pixels
    return avg_diff < threshold

def dedupe_folder(folder: Path):
    files = sorted(folder.glob("*.png"))
    if len(files) < 2:
        return
    print(f"[INFO] dedupe {folder} ({len(files)} images)")
    prev_img = Image.open(files[0]).convert("RGB")
    for f in files[1:]:
        cur_img = Image.open(f).convert("RGB")
        if is_similar(prev_img, cur_img):
            # trop proche → on supprime
            print(f"  - remove {f.name}")
            os.remove(f)
        else:
            prev_img = cur_img  # on avance

def main():
    for anime_dir in DATASET_DIR.iterdir():
        if anime_dir.is_dir():
            dedupe_folder(anime_dir)

if __name__ == "__main__":
    main()