# rename_frames.py
from pathlib import Path

DATASET_DIR = Path("animes_dataset")
PATTERN = "frame_{:04d}.png"  # frame_0001.png, frame_0002.png, ...

def rename_folder(folder: Path):
    # on prend toutes les .png
    files = sorted(folder.glob("*.png"))
    if not files:
        return

    print(f"[INFO] {folder.name} -> {len(files)} images")

    # 1) on renomme d'abord en noms temporaires pour éviter les collisions
    tmp_files = []
    for i, f in enumerate(files):
        tmp_name = folder / f"__tmp_{i:04d}.png"
        f.rename(tmp_name)
        tmp_files.append(tmp_name)

    # 2) on renomme proprement en ordre
    for i, f in enumerate(sorted(tmp_files)):
        new_name = folder / PATTERN.format(i + 1)
        f.rename(new_name)
        print(f"  {new_name.name}")

def main():
    if not DATASET_DIR.exists():
        print("⚠️ Le dossier 'animes_dataset' n'existe pas.")
        return

    for anime_dir in DATASET_DIR.iterdir():
        if anime_dir.is_dir():
            rename_folder(anime_dir)

if __name__ == "__main__":
    main()