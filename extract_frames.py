# extract_frames.py
import subprocess
from pathlib import Path

# dossier des vidéos téléchargées
VIDEOS_DIR = Path("videos_raw")
# dossier de sortie
OUTPUT_DIR = Path("animes_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# fps de base (on part pas trop haut sinon ça spam)
BASE_FPS = 15  # tu peux mettre 12 ou 10 si c’est encore trop

# durée par défaut si non précisée
DEFAULT_MAX_DURATION = "00:01:28"
DEFAULT_START = "00:00:00"

# seuil de détection de scènes (plus c’est petit, plus il garde de frames)
SCENE_THRESHOLD = 0.006  # 0.003 = sensible, 0.01 = très agressif

# config par vidéo
VIDEO_CUTS = {
    "dr_stone_opening": {"start": "00:00:00", "duration": "00:01:30"},
    "beastars_ending": {"start": "00:00:01", "duration": "00:01:30"},
    "kore_wa_zombie_desuka_opening": {"start": "00:00:01", "duration": "00:01:30"},
    "mob_psycho_100_opening": {"start": "00:00:00", "duration": "00:01:30"},
    "my_hero_academia_opening_final": {"start": "00:00:00", "duration": "00:01:16"},
    "jujutsu_kaisen_opening": {"start": "00:00:00", "duration": "00:01:30"},
    "spyxfamily_opening": {"start": "00:00:00", "duration": "00:01:28"},
    "the_apothecary_diaries_opening": {"start": "00:00:02", "duration": "00:01:29"},
    "blue_spring_ride_opening": {"start": "00:00:01", "duration": "00:01:30"},
    "imouto_umaru_chan_opening": {"start": "00:00:00", "duration": "00:01:30"},
    "orange_opening": {"start": "00:00:00", "duration": "00:01:11"},
}

def extract_frames_from_video(video_path: Path, anime_name: str):
    cfg = VIDEO_CUTS.get(anime_name, {})
    start_time = cfg.get("start", DEFAULT_START)
    duration = cfg.get("duration", DEFAULT_MAX_DURATION)

    out_dir = OUTPUT_DIR / anime_name
    out_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(out_dir / "frame_%04d.png")

    # Explication du filtre :
    # 1. fps=BASE_FPS → on commence par baisser le débit d’images
    # 2. mpdecimate=hi=64:lo=32:frac=0.2 → plus agressif que le défaut
    # 3. select=gt(scene,SCENE_THRESHOLD) → on garde les frames où ça bouge vraiment
    # 4. scale + pad → pour avoir 1920x1080 sans déformer
    vf_filter = (
        f"fps={BASE_FPS},"
        "mpdecimate=hi=64:lo=32:frac=0.2,"
        f"select='gt(scene,{SCENE_THRESHOLD})',"
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(1920-iw)/2:(1080-ih)/2:black,"
        "setpts=N/FRAME_RATE/TB"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-i", str(video_path),
        "-t", duration,
        "-vf", vf_filter,
        output_pattern,
    ]

    print(f"[INFO] {anime_name} -> start={start_time}, duration={duration}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Frames extraites dans {out_dir}")

def main():
    if not VIDEOS_DIR.exists():
        print("⚠️ Le dossier 'videos_raw' n'existe pas. Télécharge d'abord les vidéos.")
        return

    for video_file in VIDEOS_DIR.glob("*.mp4"):
        anime_name = video_file.stem
        extract_frames_from_video(video_file, anime_name)

if __name__ == "__main__":
    main()