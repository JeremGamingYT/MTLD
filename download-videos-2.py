from pytube import YouTube
from moviepy.editor import VideoFileClip
import os

def download_first_20_seconds(url, output_folder="videos"):
    # Crée le dossier si nécessaire
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Téléchargement de la vidéo
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    print("Téléchargement en cours...")
    temp_path = stream.download(output_path=output_folder, filename="temp_video.mp4")

    # Découpage des 20 premières secondes
    print("Découpage...")
    clip = VideoFileClip(temp_path).subclip(0, 20)

    final_path = os.path.join(output_folder, "video_20sec.mp4")
    clip.write_videofile(final_path)

    # Nettoyage du fichier temporaire
    os.remove(temp_path)

    print("Terminé ! Fichier créé :", final_path)


# Exemple d'utilisation :
download_first_20_seconds("https://www.youtube.com/watch?v=XXXXXXXXXXX")