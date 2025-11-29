# download_videos.py
import os
import subprocess

# 1. mets ici tes vidéos YouTube + le nom du dossier final
VIDEOS = [
    ("https://www.youtube.com/watch?v=tF4faMbs5oQ", "dr_stone_opening"),
    ("https://www.youtube.com/watch?v=rMweKnCCQXg", "beastars_ending"),
    ("https://www.youtube.com/watch?v=tyEtFdCkmHk", "kore_wa_zombie_desuka_opening"),
    ("https://www.youtube.com/watch?v=Bw-5Lka7gPE", "mob_psycho_100_opening"),
    ("https://www.youtube.com/watch?v=5ALFVv_tdYw", "my_hero_academia_opening_final"),
    ("https://www.youtube.com/watch?v=gcgKUcJKxIs", "jujutsu_kaisen_opening"),
    ("https://www.youtube.com/watch?v=YOIJOJsUkUg", "spyxfamily_opening"),
    ("https://www.youtube.com/watch?v=z9JZB08qy44", "the_apothecary_diaries_opening"),
    ("https://www.youtube.com/watch?v=6p6CIOtB2AI", "blue_spring_ride_opening"),
    ("https://www.youtube.com/watch?v=b6-2P8RgT0A", "imouto_umaru_chan_opening"),
    ("https://www.youtube.com/watch?v=7wKO5vWzxwA", "orange_opening"),
]

# -----------------------------
# 2. CONFIG
# -----------------------------
RAW_DIR = "videos_raws"     # fichiers complets téléchargés
CUT_DIR = "videos_20sec"   # fichiers coupés
CUT_DURATION = 20          # durée en secondes

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CUT_DIR, exist_ok=True)

# -----------------------------
# FONCTIONS
# -----------------------------
def download_video(url: str, name: str):
    """Télécharge une vidéo YouTube en .mp4 avec yt-dlp."""
    output_path = os.path.join(RAW_DIR, f"{name}.mp4")

    if os.path.exists(output_path):
        print(f"[SKIP] {name} déjà téléchargé.")
        return output_path

    print(f"[INFO] Téléchargement → {name}")
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)
    print(f"[OK] Téléchargé : {output_path}")
    return output_path


def cut_first_seconds(input_path: str, name: str, seconds: int = CUT_DURATION):
    """Coupe les X premières secondes avec ffmpeg."""
    output_path = os.path.join(CUT_DIR, f"{name}_cut.mp4")

    if os.path.exists(output_path):
        print(f"[SKIP] {name} déjà coupé.")
        return

    print(f"[INFO] Découpage des {seconds} premières secondes → {name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", str(seconds),
        "-c", "copy",
        output_path,
    ]

    subprocess.run(cmd, check=True)
    print(f"[OK] Fichier coupé → {output_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not VIDEOS:
        print("⚠️ La liste VIDEOS est vide.")
        return

    for url, name in VIDEOS:
        try:
            raw_path = download_video(url, name)
            cut_first_seconds(raw_path, name)
        except Exception as e:
            print(f"[ERREUR] Impossible pour {name} : {e}")


if __name__ == "__main__":
    main()