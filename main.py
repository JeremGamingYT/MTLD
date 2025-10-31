# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime

Version: 1.2 (Adaptation pour macOS Apple Silicon M3)
Auteur: Votre Expert en IA
Date: 30 octobre 2025
Description:
Ce modèle est spécifiquement conçu pour apprendre une trajectoire séquentielle fixe
(comme un clip d'anime) et la reproduire de manière déterministe à partir d'une seule
image d'amorce. Il remplace l'approche VAE/Transformer par un système Encodeur-Décodeur
déterministe couplé à un générateur de trajectoire latente basé sur un GRU et un
embedding temporel.
Cette version a été adaptée pour une exécution native sur les puces Apple Silicon (M1/M2/M3)
en utilisant le backend Metal Performance Shaders (MPS) de PyTorch.
"""

# --- 1. Importations et Configuration ---

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
from tqdm import tqdm
import math
import shutil
import warnings

# Ignorer les avertissements dépréciés de PyTorch pour une sortie plus propre
warnings.filterwarnings("ignore", category=UserWarning)

# Tenter d'importer LPIPS, guider l'utilisateur si absent
try:
    import lpips
except ImportError:
    print("="*80)
    print("ERREUR: Bibliothèque LPIPS non trouvée.")
    print("Veuillez l'installer pour une évaluation perceptive de la qualité d'image.")
    print("Exécutez la commande suivante dans votre terminal ou une cellule de notebook :")
    print("pip install lpips")
    print("="*80)
    exit()

class Config:
    """
    Configuration centralisée pour le modèle MTLD.
    Modifiez ces valeurs pour adapter l'entraînement à vos besoins.
    """
    # --- Chemins et Données ---
    # [M3 COMPAT] Les chemins ont été modifiés pour être relatifs au script.
    # Assurez-vous de créer un dossier 'dataset/anima-s-dataset' à côté de votre script
    # et de placer vos images dedans.
    DATASET_PATH = "/Users/jeremgaming/.cache/kagglehub/datasets/jeremgaming099/anima-s-dataset/versions/2/test"
    
    # Nombre total d'images dans votre séquence.
    # Le code essaiera de le déduire, mais vous pouvez le forcer ici.
    TOTAL_SEQUENCE_FRAMES = 465 
    
    # Taille des images pour l'entraînement
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    
    # Longueur des sous-séquences utilisées pour chaque étape d'entraînement.
    # Une valeur plus grande capture plus de dynamique temporelle mais consomme plus de VRAM.
    TRAINING_SEQUENCE_LENGTH = 32

    # --- Architecture du Modèle ---
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512

    # --- Paramètres d'Entraînement ---
    EPOCHS = 150
    # [M3 COMPAT] BATCH_SIZE réduit de 4 à 2 pour s'adapter à la mémoire unifiée
    # de 16Go (en visant une utilisation sous 10Go) et éviter les erreurs de mémoire.
    # Si vous rencontrez encore des problèmes, essayez de le passer à 1.
    BATCH_SIZE = 2
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 4e-4
    BETA1 = 0.5
    BETA2 = 0.999

    # --- Pondération des Pertes (Losses) ---
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 10.0
    LAMBDA_LATENT = 150.0
    LAMBDA_ADV = 1.0
    LAMBDA_FM = 10.0

    # --- Environnement et Sauvegarde ---
    # [M3 COMPAT] Détection automatique du device. Priorise le backend 'mps' pour
    # Apple Silicon, puis 'cuda' pour NVIDIA, et enfin 'cpu'.
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    # [M3 COMPAT] NUM_WORKERS mis à 0. L'utilisation de plusieurs workers avec le
    # backend 'mps' peut parfois entraîner des instabilités ou des goulots d'étranglement.
    # 0 est l'option la plus sûre et la plus stable sur macOS.
    NUM_WORKERS = 0
    
    # [M3 COMPAT] Les chemins de sauvegarde sont maintenant relatifs.
    MODEL_SAVE_PATH = "./models_mtld_v1.2/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v1.2/"
    SAVE_EPOCH_INTERVAL = 10

# --- 2. Préparation des Données (logique inchangée) ---

def setup_dummy_dataset(path, num_frames=465, size=256):
    if os.path.exists(path) and len(glob.glob(os.path.join(path, "*.png"))) == num_frames:
        print(f"Jeu de données factice déjà existant dans '{path}'.")
        return
    print(f"Création d'un jeu de données factice de {num_frames} images dans '{path}'...")
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*.png")):
        os.remove(f)
    for i in tqdm(range(num_frames), desc="Génération des images factices"):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        progress = i / (num_frames - 1)
        x = int(progress * (size - 50))
        y = int(progress * (size - 50))
        cv2.rectangle(img, (x, y), (x + 50, y + 50), (255, 100, 100), -1)
        noise = np.random.randint(0, 5, (size, size, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(path, f"frame_{i:04d}.png"))
    print("Jeu de données factice créé avec succès.")

class AnimeFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length, total_frames, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"Aucune image .png trouvée dans {root_dir}. Vérifiez le chemin dans la classe Config.")
        self.num_images = len(self.image_paths)
        if self.num_images != total_frames:
             print(f"AVERTISSEMENT: Config.TOTAL_SEQUENCE_FRAMES={total_frames} mais {self.num_images} images trouvées.")
             self.total_frames = self.num_images
        else:
            self.total_frames = total_frames
    def __len__(self):
        return self.num_images - self.sequence_length + 1
    def __getitem__(self, idx):
        sequence_paths = self.image_paths[idx : idx + self.sequence_length]
        images = [Image.open(p).convert("RGB") for p in sequence_paths]
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
        return images, idx

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = AnimeFrameDataset(
        root_dir=config.DATASET_PATH, 
        sequence_length=config.TRAINING_SEQUENCE_LENGTH,
        total_frames=config.TOTAL_SEQUENCE_FRAMES,
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        # [M3 COMPAT] pin_memory est une optimisation spécifique à CUDA. Elle est
        # désactivée car non pertinente et potentiellement contre-productive sur MPS.
        pin_memory=False, 
        drop_last=True
    )
    return dataloader

# --- 3. Architecture du Modèle MTLD (logique inchangée) ---

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim)
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.decoder_input = nn.Linear(latent_dim, 1024 * 4 * 4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, z):
        is_sequence = z.dim() == 3
        if is_sequence:
            b, s, d = z.shape
            z = z.view(b * s, d)
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 4, 4)
        img = self.model(x)
        if is_sequence:
            _, c, h, w = img.shape
            img = img.view(b, s, c, h, w)
        return img

class LatentTrajectoryGenerator(nn.Module):
    def __init__(self, num_frames, hidden_dim, latent_dim):
        super().__init__()
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.time_embedding = nn.Embedding(num_frames, hidden_dim)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)
    def forward(self, start_indices, max_len):
        batch_size = start_indices.size(0)
        initial_embeddings = self.time_embedding(start_indices)
        h0 = initial_embeddings.unsqueeze(0).repeat(2, 1, 1)
        dummy_input = torch.zeros(batch_size, max_len, 1, device=start_indices.device)
        gru_output, _ = self.gru(dummy_input, h0)
        latent_trajectory = self.head(gru_output)
        return latent_trajectory

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride, 1))]
            if normalize: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128), *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )
    def forward(self, img, extract_features=False):
        features = []
        x = img
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and extract_features:
                features.append(x)
        return x, features[:4]

class MTLD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.IMG_CHANNELS, config.LATENT_DIM)
        self.decoder = Decoder(config.LATENT_DIM, config.IMG_CHANNELS)
        self.trajectory_generator = LatentTrajectoryGenerator(
            config.TOTAL_SEQUENCE_FRAMES, 
            config.GRU_HIDDEN_DIM, 
            config.LATENT_DIM
        )
        self.discriminator = PatchDiscriminator(config.IMG_CHANNELS)

# --- 4. Boucle d'Entraînement (logique inchangée) ---

def train_mtld():
    config = Config()
    print(f"Utilisation du device : {config.DEVICE.upper()}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    # Dé-commentez la ligne suivante si vous voulez générer un jeu de données factice
    # setup_dummy_dataset(config.DATASET_PATH, config.TOTAL_SEQUENCE_FRAMES, config.IMG_SIZE)
    
    dataloader = get_dataloader(config)
    model = MTLD(config).to(config.DEVICE)
    
    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.trajectory_generator.parameters())
    d_params = list(model.discriminator.parameters())
    
    opt_g = optim.Adam(g_params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(d_params, lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    loss_l1 = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    
    print("Début de l'entraînement...")
    for epoch in range(config.EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for i, (real_seq_imgs, start_indices) in enumerate(pbar):
            real_seq_imgs = real_seq_imgs.to(config.DEVICE)
            start_indices = start_indices.to(config.DEVICE)
            b, s, c, h, w = real_seq_imgs.shape
            real_imgs_flat = real_seq_imgs.view(b * s, c, h, w)
            
            # --- Entraînement du Générateur ---
            opt_g.zero_grad()
            with torch.no_grad():
                z_true_flat = model.encoder(real_imgs_flat)
            z_true = z_true_flat.view(b, s, -1)
            
            z_pred = model.trajectory_generator(start_indices, max_len=s)
            fake_seq_imgs = model.decoder(z_pred)
            fake_imgs_flat = fake_seq_imgs.view(b * s, c, h, w)
            
            loss_latent = loss_mse(z_pred, z_true)
            loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
            loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()
            
            fake_d_output, _ = model.discriminator(fake_imgs_flat)
            target_real = torch.ones_like(fake_d_output)
            loss_adv = loss_mse(fake_d_output, target_real)
            
            _, real_features = model.discriminator(real_imgs_flat.detach(), extract_features=True)
            _, fake_features = model.discriminator(fake_imgs_flat, extract_features=True)
            loss_fm = sum(loss_l1(fake_f, real_f.detach()) for real_f, fake_f in zip(real_features, fake_features))
            
            loss_g = (config.LAMBDA_REC_L1 * loss_rec_l1 +
                      config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                      config.LAMBDA_LATENT * loss_latent +
                      config.LAMBDA_ADV * loss_adv +
                      config.LAMBDA_FM * loss_fm)
            
            loss_g.backward()
            opt_g.step()
            
            # --- Entraînement du Discriminateur ---
            opt_d.zero_grad()
            real_d_output, _ = model.discriminator(real_imgs_flat)
            loss_d_real = loss_mse(real_d_output, target_real)
            
            fake_d_output, _ = model.discriminator(fake_imgs_flat.detach())
            target_fake = torch.zeros_like(fake_d_output)
            loss_d_fake = loss_mse(fake_d_output, target_fake)
            
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()
            
            pbar.set_postfix({
                "L_Rec_L1": f"{loss_rec_l1.item():.3f}", "L_Latent": f"{loss_latent.item():.3f}",
                "L_Adv": f"{loss_adv.item():.3f}", "L_D": f"{loss_d.item():.3f}",
            })
            
        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            save_path = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v1.2_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\nModèle sauvegardé : {save_path}")
            
            with torch.no_grad():
                model.eval()
                real_sample = real_seq_imgs[0].unsqueeze(0)
                start_idx_sample = start_indices[0].unsqueeze(0)
                z_pred_sample = model.trajectory_generator(start_idx_sample, max_len=s)
                fake_sample = model.decoder(z_pred_sample)
                comparison = torch.cat([real_sample, fake_sample], dim=0)
                comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                utils.save_image(comparison_flat, grid_path, nrow=s, normalize=True)
                print(f"Image de comparaison sauvegardée : {grid_path}")
                model.train()
                
    print("Entraînement terminé.")

# --- 5. Génération de Séquence (logique inchangée) ---

def generate_sequence(model_path, priming_image_path, priming_frame_index, config):
    print("--- Démarrage de la Génération de Séquence ---")
    if not os.path.exists(model_path):
        print(f"ERREUR : Fichier modèle non trouvé à {model_path}"); return
    if not os.path.exists(priming_image_path):
        print(f"ERREUR : Image d'amorce non trouvée à {priming_image_path}"); return
    if not (0 <= priming_frame_index < config.TOTAL_SEQUENCE_FRAMES):
        print(f"ERREUR : L'index d'amorce {priming_frame_index} est invalide."); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames")
    os.makedirs(output_dir_frames, exist_ok=True)
    
    model = MTLD(config).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        print(f"Modèle chargé avec succès depuis '{model_path}' sur le device '{config.DEVICE.upper()}'.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}"); return

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    num_frames_to_generate = config.TOTAL_SEQUENCE_FRAMES - priming_frame_index
    print(f"Génération de {num_frames_to_generate} images à partir de la trame {priming_frame_index}.")

    with torch.no_grad():
        start_index_tensor = torch.tensor([priming_frame_index], dtype=torch.long, device=config.DEVICE)
        latent_trajectory = model.trajectory_generator(start_index_tensor, max_len=num_frames_to_generate)
        generated_imgs_seq = model.decoder(latent_trajectory).squeeze(0)

    video_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_from_{priming_frame_index}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (config.IMG_SIZE, config.IMG_SIZE))
    pil_images_for_gif = []

    priming_img_pil = Image.open(priming_image_path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    video_writer.write(cv2.cvtColor(np.array(priming_img_pil), cv2.COLOR_RGB2BGR))
    pil_images_for_gif.append(priming_img_pil)
    priming_img_pil.save(os.path.join(output_dir_frames, f"frame_{priming_frame_index:04d}.png"))
    
    for i, img_tensor in enumerate(tqdm(generated_imgs_seq, desc="Sauvegarde des images")):
        frame_idx = priming_frame_index + i + 1
        img_np = np.clip((img_tensor.permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        img_pil.save(os.path.join(output_dir_frames, f"frame_{frame_idx:04d}.png"))
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        pil_images_for_gif.append(img_pil)

    video_writer.release()
    print(f"Vidéo MP4 sauvegardée : {video_path}")
    gif_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_from_{priming_frame_index}.gif")
    pil_images_for_gif[0].save(
        gif_path, save_all=True, append_images=pil_images_for_gif[1:],
        duration=int(1000 / 24), loop=0
    )
    print(f"GIF sauvegardé : {gif_path}")


if __name__ == '__main__':
    # --- Instructions d'Utilisation ---
    # 1. Assurez-vous d'avoir configuré les chemins dans la classe `Config`.
    # 2. Vérifiez que la structure des dossiers (dataset, models) est correcte.
    # 3. Choisissez UN SEUL mode d'exécution ci-dessous en dé-commentant les lignes.

    # --- MODE 1: ENTRAÎNEMENT SUR VOS DONNÉES ---
    print("\n--- MODE ENTRAÎNEMENT ---")
    train_mtld()

    # --- MODE 2: GÉNÉRATION À PARTIR D'UN MODÈLE ENTRAÎNÉ ---
    #print("\n--- MODE GÉNÉRATION ---")
    #config_gen = Config()
    # [M3 COMPAT] Le chemin du modèle est maintenant relatif et pointe vers le dossier de sauvegarde.
    # Assurez-vous que ce fichier existe bien dans ./models_mtld_v1.2/
    #model_file = os.path.join(config_gen.MODEL_SAVE_PATH, "mtld_v1.1_epoch_120.pth") 
    
    # [M3 COMPAT] Le chemin de l'image d'amorce est également relatif.
    # Assurez-vous que cette image existe bien dans votre dossier de dataset.
    #priming_image = os.path.join(config_gen.DATASET_PATH, "frame_0001.png")
    #priming_index = 1 # L'index doit correspondre à l'image (frame_0001.png -> index 1)
    
    #generate_sequence(model_file, priming_image, priming_index, config_gen)