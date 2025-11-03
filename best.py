# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime

Version: 1.8 (Pré-chargement du Dataset en RAM)
Auteur: Votre Expert en IA
Date: 01 novembre 2025
Description:
Cette version introduit une optimisation majeure pour les pipelines de données
limités par les I/O disque. Un nouveau paramètre de configuration,
`PRELOAD_DATASET_IN_RAM`, permet de charger et de prétraiter l'intégralité
du dataset en mémoire vive au démarrage du script.

1.  **Performance Accrue :** En éliminant les accès disque pendant l'entraînement,
    cette approche maximise l'utilisation du GPU et accélère significativement
    la vitesse des époques, surtout avec des disques lents ou des datasets
    composés de nombreux petits fichiers.
2.  **Flexibilité :** L'option est configurable, permettant de la désactiver
    pour les datasets trop volumineux qui ne tiendraient pas en RAM.
3.  **Transparence :** Le script calcule et affiche une estimation de la mémoire
    requise et utilise une barre de progression pour suivre le chargement initial.
"""

# --- 1. Importations et Configuration ---

import os
import glob
import re
import time
from collections import deque
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

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lpips
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque LPIPS non trouvée."); print("!pip install lpips"); print("="*80); exit()

# AVERTISSEMENT: La bibliothèque 'pynvml' est dépréciée.
# Pour supprimer l'avertissement de PyTorch, il est recommandé de la remplacer par 'nvidia-ml-py':
# pip uninstall pynvml
# pip install nvidia-ml-py
try:
    import pynvml
except ImportError:
    print("="*80); print("AVERTISSEMENT: pynvml non trouvé. La surveillance GPU est désactivée."); print("!pip install nvidia-ml-py"); print("="*80)
    pynvml = None

class GPUMonitor:
    def __init__(self, config, device_id=0):
        if not pynvml:
            self.enabled = False
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.config = config
            self.enabled = config.GPU_MONITORING_ENABLED
            self.recent_alerts = deque()
            print(f"GPUMonitor initialisé pour le GPU {pynvml.nvmlDeviceGetName(self.handle)}.")
        except Exception as e:
            print(f"ERREUR: Impossible d'initialiser pynvml. Surveillance GPU désactivée. Erreur : {e}")
            self.enabled = False

    def check(self):
        if not self.enabled:
            return "continue"
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        if temp > self.config.GPU_TEMP_THRESHOLD_C:
            print(f"\n! ALERTE GPU ! Température atteinte : {temp}°C (Seuil: {self.config.GPU_TEMP_THRESHOLD_C}°C)")
            return self._handle_alert()
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
            power_percent = (power_usage / power_limit) * 100
            if power_percent > self.config.GPU_POWER_THRESHOLD_PERCENT:
                print(f"\n! ALERTE GPU ! Consommation atteinte : {power_percent:.1f}% (Seuil: {self.config.GPU_POWER_THRESHOLD_PERCENT}%)")
                return self._handle_alert()
        except pynvml.NVMLError:
            pass
        return "continue"

    def _handle_alert(self):
        current_time = time.time()
        self.recent_alerts.append(current_time)
        while self.recent_alerts and current_time - self.recent_alerts[0] > self.config.GPU_SHUTDOWN_WINDOW_S:
            self.recent_alerts.popleft()
        if len(self.recent_alerts) >= self.config.GPU_SHUTDOWN_THRESHOLD_COUNT:
            print(f"\n!! ALERTE CRITIQUE !! {len(self.recent_alerts)} alertes GPU en {self.config.GPU_SHUTDOWN_WINDOW_S}s. Arrêt d'urgence.")
            return "shutdown"
        else:
            print(f"Mise en pause de l'entraînement pour {self.config.GPU_PAUSE_DURATION_S} secondes...")
            time.sleep(self.config.GPU_PAUSE_DURATION_S)
            return "paused"

    def shutdown(self):
        if self.enabled:
            pynvml.nvmlShutdown()

class Config:
    DATASET_PATH = "/root/.cache/kagglehub/datasets/jeremgaming099/anima-s-dataset/versions/6/animes_dataset"
    # NOUVEAU: Option pour pré-charger le dataset en RAM pour accélérer l'entraînement.
    # Mettre à False si le dataset est trop volumineux pour la RAM disponible.
    PRELOAD_DATASET_IN_RAM = True
    
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 16
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 4e-4
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 10.0
    LAMBDA_LATENT = 150.0
    LAMBDA_ADV = 1.0
    LAMBDA_FM = 10.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    MODEL_SAVE_PATH = "./models_mtld_v1.8/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v1.8/"
    SAVE_EPOCH_INTERVAL = 10
    
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = "./models_mtld_v1.7/mtld_v1.7_checkpoint_epoch_10.pth" 

    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_POWER_THRESHOLD_PERCENT = 95
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300

# --- 2. Préparation des Données (MODIFIÉE pour le pré-chargement en RAM) ---

class AnimeFrameDataset(Dataset):
    """
    Dataset gérant les séquences et capable de pré-charger toutes les images
    en RAM pour des performances maximales.
    """
    def __init__(self, root_dir, sequence_length, transform=None, config=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.config = config
        self.sequences = []
        self.cumulative_lengths = []
        self.preloaded_data = None

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Le répertoire racine du dataset n'a pas été trouvé : {root_dir}")

        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not arc_dirs:
            raise FileNotFoundError(f"Aucun sous-dossier (arc) trouvé dans {root_dir}")

        print(f"Détection de {len(arc_dirs)} arcs potentiels : {arc_dirs}")

        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            image_paths = sorted(
                glob.glob(os.path.join(arc_path, "*.png")),
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
            )
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_possible_sequences = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_possible_sequences
                self.cumulative_lengths.append(total_valid_sequences)
                print(f"  -> Arc '{arc_dir}' validé : {len(image_paths)} images, {num_possible_sequences} séquences possibles.")
            else:
                print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court pour une séquence de {self.sequence_length}).")

        if not self.sequences:
            raise ValueError("Aucun arc valide (suffisamment long) n'a été trouvé dans le dataset.")
        print(f"\nDataset initialisé : {len(self.sequences)} arcs valides, {self.total_sequences()} séquences d'entraînement au total.")

        # NOUVEAU: Logique de pré-chargement
        if self.config and self.config.PRELOAD_DATASET_IN_RAM:
            self._preload_images()

    def _preload_images(self):
        """Charge et transforme toutes les images uniques du dataset en RAM."""
        print("\n--- Pré-chargement du dataset en RAM ---")
        # Créer une liste de tous les chemins d'images uniques pour éviter de charger des doublons
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        num_images = len(all_paths)
        if num_images == 0:
            print("Aucune image à pré-charger.")
            return

        # Calculer l'estimation de la RAM
        c, h, w = self.config.IMG_CHANNELS, self.config.IMG_SIZE, self.config.IMG_SIZE
        bytes_per_tensor = c * h * w * 4  # torch.float32 utilise 4 octets
        total_bytes = num_images * bytes_per_tensor
        total_mb = total_bytes / (1024**2)
        print(f"Chargement de {num_images} images uniques... Estimation RAM : {total_mb:.2f} Mo.")

        self.preloaded_data = {}
        for path in tqdm(all_paths, desc="Pré-chargement des images"):
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    self.preloaded_data[path] = self.transform(img)
            except Exception as e:
                print(f"AVERTISSEMENT: Impossible de charger l'image {path}. Erreur: {e}")
        
        print("--- Pré-chargement terminé ---")


    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
        
    def total_sequences(self):
        return self.__len__()

    def __getitem__(self, idx):
        arc_index = 0
        while idx >= self.cumulative_lengths[arc_index]:
            arc_index += 1
        if arc_index == 0:
            local_start_idx = idx
        else:
            local_start_idx = idx - self.cumulative_lengths[arc_index - 1]
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        
        # MODIFIÉ: Utilise les données pré-chargées si disponibles
        if self.preloaded_data:
            images = [self.preloaded_data[p] for p in sequence_paths if p in self.preloaded_data]
            if len(images) != self.sequence_length:
                 raise RuntimeError(f"Une ou plusieurs images de la séquence index {idx} n'ont pas pu être chargées depuis la RAM.")
        else:
            images_pil = [Image.open(p).convert("RGB") for p in sequence_paths]
            if self.transform:
                images = [self.transform(img) for img in images_pil]
            else:
                images = images_pil
        
        return torch.stack(images)

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # MODIFIÉ: Passe l'objet config au Dataset
    dataset = AnimeFrameDataset(
        root_dir=config.DATASET_PATH, 
        sequence_length=config.TRAINING_SEQUENCE_LENGTH,
        transform=transform,
        config=config
    )
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    return dataloader, dataset.total_sequences()

# --- 3. Architecture du Modèle (inchangée) ---

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
            nn.Linear(1024 * (config.IMG_SIZE // 64) * (config.IMG_SIZE // 64), latent_dim)
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.decoder_input_size = 1024 * (config.IMG_SIZE // 64) * (config.IMG_SIZE // 64)
        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size)
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
        feature_map_size = config.IMG_SIZE // 64
        x = x.view(-1, 1024, feature_map_size, feature_map_size)
        img = self.model(x)
        if is_sequence:
            _, c, h, w = img.shape
            img = img.view(b, s, c, h, w)
        return img

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

class ConditionalLatentTrajectoryGenerator(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)
    def forward(self, z_start, max_len):
        batch_size = z_start.size(0)
        h0_proj = self.z_to_hidden(z_start)
        h0 = h0_proj.unsqueeze(0).repeat(2, 1, 1)
        gru_input = z_start.unsqueeze(1)
        outputs = []
        for _ in range(max_len):
            gru_output, h0 = self.gru(gru_input, h0)
            next_z = self.head(gru_output)
            outputs.append(next_z)
            gru_input = next_z
        latent_trajectory = torch.cat(outputs, dim=1)
        return latent_trajectory

class MTLD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.IMG_CHANNELS, config.LATENT_DIM)
        self.decoder = Decoder(config.LATENT_DIM, config.IMG_CHANNELS)
        self.trajectory_generator = ConditionalLatentTrajectoryGenerator(config.GRU_HIDDEN_DIM, config.LATENT_DIM)
        self.discriminator = PatchDiscriminator(config.IMG_CHANNELS)

# --- 4. Boucle d'Entraînement (inchangée) ---

def save_checkpoint(epoch, model, opt_g, opt_d, scaler_g, scaler_d, config):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
        'scaler_g_state_dict': scaler_g.state_dict(),
        'scaler_d_state_dict': scaler_d.state_dict(),
    }
    filename = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v1.8_checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"\nCheckpoint sauvegardé : {filename}")

def train_mtld():
    config = Config()
    print(f"Utilisation du device : {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    dataloader, total_sequences = get_dataloader(config)
    
    model = MTLD(config).to(config.DEVICE)
    
    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.trajectory_generator.parameters())
    d_params = list(model.discriminator.parameters())
    opt_g = optim.Adam(g_params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(d_params, lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    loss_l1 = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    
    scaler_g = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    scaler_d = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    
    start_epoch = 0
    
    if config.RESUME_TRAINING:
        path = config.CHECKPOINT_TO_RESUME
        if os.path.isfile(path):
            print(f"Reprise de l'entraînement depuis le checkpoint : {path}")
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Checkpoint complet détecté. Chargement du modèle, des optimiseurs et des scalers.")
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_g_state_dict' in checkpoint: opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                if 'optimizer_d_state_dict' in checkpoint: opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                if 'scaler_g_state_dict' in checkpoint: scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
                if 'scaler_d_state_dict' in checkpoint: scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Reprise à partir de l'époque {start_epoch}.")
            else:
                print("Ancien format de poids détecté. Chargement des poids du modèle uniquement.")
                model.load_state_dict(checkpoint)
                start_epoch = 0
                print("Les optimiseurs et scalers sont réinitialisés.")
        else:
            print(f"AVERTISSEMENT: Fichier de checkpoint non trouvé à '{path}'. Démarrage d'un nouvel entraînement.")

    gpu_monitor = GPUMonitor(config)

    print("\nDébut de l'entraînement du modèle conditionnel...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape
                
                priming_img = real_seq_imgs[:, 0, :, :, :]
                future_imgs = real_seq_imgs[:, 1:, :, :, :]
                
                opt_g.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == "cuda")):
                    with torch.no_grad():
                        z_start_true = model.encoder(priming_img)
                        z_future_true = model.encoder(future_imgs.reshape(-1, c, h, w)).view(b, s - 1, -1)
                    z_future_pred = model.trajectory_generator(z_start_true, max_len=s - 1)
                    z_pred_full_seq = torch.cat([z_start_true.unsqueeze(1), z_future_pred], dim=1)
                    fake_seq_imgs = model.decoder(z_pred_full_seq)
                    fake_imgs_flat = fake_seq_imgs.view(b * s, c, h, w)
                    real_imgs_flat = real_seq_imgs.view(b * s, c, h, w)
                    loss_latent = loss_mse(z_future_pred, z_future_true)
                    loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
                    loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()
                    fake_d_output, _ = model.discriminator(fake_imgs_flat)
                    target_real = torch.ones_like(fake_d_output)
                    loss_adv = loss_mse(fake_d_output, target_real)
                    _, real_features = model.discriminator(real_imgs_flat.detach(), extract_features=True)
                    _, fake_features = model.discriminator(fake_imgs_flat, extract_features=True)
                    loss_fm = sum(loss_l1(fake_f, real_f.detach()) for real_f, fake_f in zip(real_features, fake_features))
                    loss_g = (config.LAMBDA_REC_L1 * loss_rec_l1 + config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                              config.LAMBDA_LATENT * loss_latent + config.LAMBDA_ADV * loss_adv + config.LAMBDA_FM * loss_fm)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
                
                opt_d.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == "cuda")):
                    real_d_output, _ = model.discriminator(real_imgs_flat.detach())
                    loss_d_real = loss_mse(real_d_output, target_real)
                    fake_d_output, _ = model.discriminator(fake_imgs_flat.detach())
                    target_fake = torch.zeros_like(fake_d_output)
                    loss_d_fake = loss_mse(fake_d_output, target_fake)
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
                
                pbar.set_postfix({
                    "L_Pred": f"{loss_latent.item():.3f}", "L_Rec": f"{loss_rec_l1.item():.3f}",
                    "L_Adv": f"{loss_adv.item():.3f}", "L_D": f"{loss_d.item():.3f}",
                })
                
                if i > 0 and i % 20 == 0:
                    status = gpu_monitor.check()
                    if status == "shutdown":
                        print("Arrêt d'urgence demandé. Sauvegarde du checkpoint...")
                        save_checkpoint(epoch + 1, model, opt_g, opt_d, scaler_g, scaler_d, config)
                        gpu_monitor.shutdown()
                        exit()

            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, opt_g, opt_d, scaler_g, scaler_d, config)
                with torch.no_grad():
                    model.eval()
                    with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == "cuda")):
                        fake_seq_imgs_eval = model.decoder(z_pred_full_seq)
                    comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_seq_imgs_eval[0].unsqueeze(0)], dim=0)
                    comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                    utils.save_image(comparison_flat, grid_path, nrow=s, normalize=True)
                    print(f"Image de comparaison sauvegardée : {grid_path}")
                    model.train()
    finally:
        gpu_monitor.shutdown()
    print("Entraînement terminé.")

# --- 5. Génération de Séquence (inchangée) ---

def generate_sequence(model_path, priming_image_path, num_frames_to_generate, config):
    print("--- Démarrage de la Génération de Séquence Conditionnelle ---")
    if not os.path.exists(model_path):
        print(f"ERREUR : Fichier modèle non trouvé à {model_path}"); return
    if not os.path.exists(priming_image_path):
        print(f"ERREUR : Image d'amorce non trouvée à {priming_image_path}"); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_conditional")
    os.makedirs(output_dir_frames, exist_ok=True)
    
    gen_config = Config()
    model = MTLD(gen_config).to(gen_config.DEVICE)
    
    print(f"Chargement du modèle depuis : {model_path}")
    checkpoint = torch.load(model_path, map_location=gen_config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Poids du modèle chargés depuis un checkpoint complet.")
    else:
        model.load_state_dict(checkpoint)
        print("Poids du modèle chargés depuis un fichier de poids simple.")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((gen_config.IMG_SIZE, gen_config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(gen_config.DEVICE)

    print(f"Génération de {num_frames_to_generate} images à partir de l'image d'amorce.")

    with torch.no_grad():
        with torch.amp.autocast(device_type=gen_config.DEVICE, dtype=torch.float16, enabled=(gen_config.DEVICE == "cuda")):
            z_start = model.encoder(priming_tensor)
            z_future = model.trajectory_generator(z_start, max_len=num_frames_to_generate - 1)
            z_full_seq = torch.cat([z_start.unsqueeze(1), z_future], dim=1)
            generated_imgs_seq = model.decoder(z_full_seq).squeeze(0)

    video_path = os.path.join(gen_config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (gen_config.IMG_SIZE, gen_config.IMG_SIZE))
    pil_images_for_gif = []

    for i, img_tensor in enumerate(tqdm(generated_imgs_seq, desc="Sauvegarde des images")):
        img_np = np.clip((img_tensor.permute(1, 2, 0).float() * 0.5 + 0.5).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        img_pil.save(os.path.join(output_dir_frames, f"frame_{i:04d}.png"))
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        pil_images_for_gif.append(img_pil)

    video_writer.release()
    print(f"Vidéo MP4 sauvegardée : {video_path}")
    gif_path = os.path.join(gen_config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.gif")
    pil_images_for_gif[0].save(
        gif_path, save_all=True, append_images=pil_images_for_gif[1:],
        duration=int(1000 / 24), loop=0
    )
    print(f"GIF sauvegardé : {gif_path}")


if __name__ == '__main__':
    config = Config()
    
    #print(f"\n--- MODE ENTRAÎNEMENT (v1.8 Pré-chargement RAM) ---")
    #train_mtld()

    # --- MODE 2: GÉNÉRATION CONDITIONNELLE ---
    print("\n--- MODE GÉNÉRATION (v1.8) ---")
    config_gen = Config()
    model_file = "./mtld_v1.8_checkpoint_epoch_100.pth" 
    try:
    #     # On doit instancier un Dataset pour trouver un chemin d'image valide
        dummy_dataset = AnimeFrameDataset(config_gen.DATASET_PATH, 1, config=config_gen)
        priming_image = dummy_dataset.sequences[0][50] 
        print(f"Utilisation de l'image d'amorce : {priming_image}")
    except (IndexError, FileNotFoundError, ValueError) as e:
        print(f"ATTENTION: Impossible de trouver une image d'amorce par défaut : {e}. Spécifiez un chemin valide.")
        priming_image = "frame_0001.png" # REMPLACEZ CECI

    if os.path.exists(priming_image):
        generate_sequence(
            model_path=model_file, 
            priming_image_path="frame_0001.png", 
            num_frames_to_generate=100,
            config=config_gen
        )