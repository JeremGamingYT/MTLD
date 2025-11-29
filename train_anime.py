"""
Entraînement et génération pour Anime-AI
=======================================

Ce script assemble les composants définis dans `anime_model.py` pour créer un
pipeline complet d'entraînement et de génération.  Il comprend :

- Un dataset qui charge des séquences vidéo d'anime (~20 s), extrait les
  squelettes et prépare les heatmaps et les images.
- Un processus d'entraînement en deux étapes : un pré‑entraînement de l'auto
  encodeur pour apprendre un latent de style, puis un entraînement du
  générateur conditionné par le squelette, avec possibilité d'injecter des
  modules LoRA pour l'adaptation personnalisée.
- Une fonction de génération qui, à partir d'une nouvelle séquence de
  squelettes (ou d'une image fixe pour répéter le squelette), synthétise une
  vidéo d'anime.

Tous les paramètres (chemins, dimensions, nombres d'epochs) peuvent être
ajustés en début de script.  Le code est commenté pour être exécuté dans un
notebook ou en tant que script autonome.
"""

import os
import glob
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from anime_model import SkeletonExtractor, AnimeModel

# -----------------------------------------------------------------------------
# Paramètres globaux
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_DIR = "/kaggle/input/animes-videos"  # dossier contenant des vidéos .mp4 d'environ 20 s
FRAME_SIZE = (256, 256)
NUM_JOINTS = 33  # nombre de keypoints utilisés par MediaPipe
VAE_LATENT_DIM = 64
STYLE_DIM = 128
BATCH_SIZE = 2
NUM_EPOCHS_VAE = 5
NUM_EPOCHS_GEN = 10
LEARNING_RATE = 1e-4

# -----------------------------------------------------------------------------
# Dataset : extraction de frames et de heatmaps
# -----------------------------------------------------------------------------

class AnimeVideoDataset(Dataset):
    def __init__(self, video_dir: str, num_frames: int = 64, frame_size: Tuple[int, int] = (256, 256)):
        self.video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(frame_size),
        ])
        self.skeleton_extractor = SkeletonExtractor(device=DEVICE)

    def __len__(self):
        return len(self.video_paths)

    def _extract_frames(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Charge aléatoirement une fenêtre de frames et extrait les heatmaps."""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # si la vidéo est plus courte que num_frames, on boucle
        start = random.randint(0, max(total_frames - self.num_frames, 0))
        images = []
        heatmaps = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        count = 0
        while count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                # boucle si on atteint la fin
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = self.transform(Image.fromarray(frame_rgb))  # type: ignore
            images.append(img)
            count += 1
        cap.release()
        images_t = torch.stack(images)  # (T, 3, H, W)
        # extraire les keypoints pour chaque frame
        keypoints = self.skeleton_extractor.extract_keypoints(images_t)
        # transformer en heatmaps
        hmaps = self.skeleton_extractor.keypoints_to_heatmaps(keypoints, self.frame_size)
        return images_t, hmaps

    def __getitem__(self, idx):
        vid = self.video_paths[idx]
        frames, hmaps = self._extract_frames(vid)
        return frames, hmaps

# -----------------------------------------------------------------------------
# Fonction d'entraînement
# -----------------------------------------------------------------------------

def train_anime_model():
    dataset = AnimeVideoDataset(VIDEO_DIR, num_frames=32, frame_size=FRAME_SIZE)
    # MediaPipe n'est pas thread‑safe lorsque l'on utilise plusieurs processus.  Le fait de
    # lancer des DataLoader workers en parallèle peut provoquer des erreurs de mutex ou
    # des crashs dans les bibliothèques (cf. logs).  Pour éviter ce problème, nous
    # utilisons un seul worker en définissant num_workers=0.  Cela peut ralentir
    # légèrement le chargement, mais assure la stabilité.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialisation du modèle
    model = AnimeModel(num_joints=NUM_JOINTS, vae_latent_dim=VAE_LATENT_DIM, generator_style_dim=STYLE_DIM)
    model = model.to(DEVICE)

    # Optimiseur distinct pour VAE et générateur
    optimizer_vae = torch.optim.Adam(model.vae.parameters(), lr=LEARNING_RATE)
    optimizer_gen = torch.optim.Adam(
        list(model.style_fc.parameters()) + list(model.generator.parameters()), lr=LEARNING_RATE)

    mse = nn.MSELoss()

    # Étape 1 : entraînement de l'auto‑encodeur de style
    print("--- Entraînement du VAE ---")
    model.train()
    for epoch in range(NUM_EPOCHS_VAE):
        total_loss = 0.0
        for frames, hmaps in dataloader:
            frames = frames.to(DEVICE, dtype=torch.float32)
            recon = model.vae(frames.view(-1, 3, *FRAME_SIZE))
            loss = mse(recon, frames.view(-1, 3, *FRAME_SIZE))
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            total_loss += loss.item()
        print(f"Epoch VAE {epoch+1}/{NUM_EPOCHS_VAE}, loss={total_loss/len(dataloader):.4f}")

    # Étape 2 : entraînement du générateur conditionné
    print("--- Entraînement du générateur conditionné ---")
    model.train()
    for epoch in range(NUM_EPOCHS_GEN):
        total_loss_gen = 0.0
        for frames, hmaps in dataloader:
            frames = frames.to(DEVICE, dtype=torch.float32)
            hmaps = hmaps.to(DEVICE, dtype=torch.float32)
            # shuffle style pour apprendre à généraliser
            recon, gen = model(frames.view(-1, 3, *FRAME_SIZE), hmaps.view(-1, NUM_JOINTS, *FRAME_SIZE))
            # on souhaite que gen soit proche des frames : apprentissage supervisé
            loss_gen = mse(gen, frames.view(-1, 3, *FRAME_SIZE))
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
            total_loss_gen += loss_gen.item()
        print(f"Epoch GEN {epoch+1}/{NUM_EPOCHS_GEN}, loss={total_loss_gen/len(dataloader):.4f}")
    # sauvegarder le modèle
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/anime_model.pth')
    print("Entraînement terminé, modèle enregistré dans checkpoints/anime_model.pth")

# -----------------------------------------------------------------------------
# Génération d'une vidéo d'anime à partir d'une séquence de frames de référence
# -----------------------------------------------------------------------------

def generate_anime_video(input_video: str, output_path: str,
                          model_checkpoint: str = 'checkpoints/anime_model.pth',
                          num_frames: int = 64):
    # charger le modèle
    model = AnimeModel(num_joints=NUM_JOINTS, vae_latent_dim=VAE_LATENT_DIM, generator_style_dim=STYLE_DIM)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    skeleton_extractor = SkeletonExtractor(device=DEVICE)
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_frames = []
    # si moins de frames que demandé, on boucle
    idx = 0
    while len(out_frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
        img = T.Resize(FRAME_SIZE)(img)
        # extraire squelette
        kp = skeleton_extractor.extract_keypoints(img)[0:1]
        hmap = skeleton_extractor.keypoints_to_heatmaps(kp.unsqueeze(0), FRAME_SIZE)
        # échantillonner un style aléatoire ou utiliser la moyenne
        style_vec = torch.randn(1, STYLE_DIM, device=DEVICE)
        with torch.no_grad():
            out_img = model.generate(hmap.to(DEVICE), style_vec)
        out_np = (out_img[0].cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
        out_frames.append(out_np)
        idx += 1
    cap.release()
    # sauvegarde avec imageio
    try:
        import imageio
        imageio.mimwrite(output_path, out_frames, fps=24, macro_block_size=None)
        print(f"Vidéo générée enregistrée dans {output_path}")
    except ImportError:
        print("imageio non installé, impossible de sauvegarder la vidéo.")
    return out_frames

# -----------------------------------------------------------------------------
# Point d'entrée
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description="Entraîner ou générer un modèle Anime-AI")
    parser.add_argument('--train', action='store_true', help='Lance l\'entraînement')
    parser.add_argument('--generate', type=str, default=None, help='Chemin vers une vidéo de référence pour la génération')
    parser.add_argument('--output', type=str, default='generated_anime.mp4', help='Chemin du fichier de sortie généré')
    args = parser.parse_args()
    if args.train:
        train_anime_model()
    elif args.generate:
        generate_anime_video(args.generate, args.output)
    else:
        print("Utilisez --train pour entraîner ou --generate <video> pour générer.")
