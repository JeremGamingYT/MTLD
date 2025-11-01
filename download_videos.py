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

# 2. dossier où on met les .mp4 téléchargés
DOWNLOAD_DIR = "videos_raw"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_video(url: str, name: str):
    """
    Télécharge une vidéo YouTube en .mp4 avec yt-dlp.
    Nécessite : pip install yt-dlp
    """
    output_path = os.path.join(DOWNLOAD_DIR, f"{name}.mp4")
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_path,
        url
    ]
    print(f"[INFO] Téléchargement de {url} -> {output_path}")
    subprocess.run(cmd, check=True)
    print(f"[OK] {name} téléchargé.")

def main():
    if not VIDEOS:
        print("⚠️ Ajoute tes vidéos dans la liste VIDEOS en haut du fichier.")
        return

    for url, name in VIDEOS:
        download_video(url, name)

if __name__ == "__main__":
    main()
