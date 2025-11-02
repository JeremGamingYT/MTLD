# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime

Version: 1.9 (Intégration d'un Encodeur Sémantique de type SVG)
Date: 02 novembre 2025
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

# NOUVEAU: Dépendance pour l'encodeur sémantique DINO
try:
    import timm
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque 'timm' non trouvée, requise pour l'encodeur SVG."); print("!pip install timm"); print("="*80); exit()

try:
    import pynvml
except ImportError:
    print("="*80); print("AVERTISSEMENT: pynvml non trouvé. La surveillance GPU est désactivée."); print("!pip install nvidia-ml-py"); print("="*80)
    pynvml = None

class GPUMonitor:
    def __init__(self, config, device_id=0):
        if not pynvml: self.enabled = False; return
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
        if not self.enabled: return "continue"
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
        except pynvml.NVMLError: pass
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
        if self.enabled: pynvml.nvmlShutdown()

class Config:
    # --- Configuration du Modèle et de l'Entraînement ---
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset"
    PRELOAD_DATASET_IN_RAM = True
    
    # NOUVEAU: Choix de l'architecture de l'encodeur
    # Mettre à False pour utiliser l'encodeur CNN original (classe Encoder)
    USE_SVG_ENCODER = True 
    
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 16
    LATENT_DIM = 512 # Augmenté pour accommoder la richesse des features de DINO
    GRU_HIDDEN_DIM = 1024
    
    # NOUVEAU: Dimension pour l'encodeur résiduel (utilisé seulement si USE_SVG_ENCODER=True)
    RESIDUAL_DIM = 128
    
    EPOCHS = 200
    BATCH_SIZE = 2
    LEARNING_RATE_G = 1e-4 # Taux d'apprentissage potentiellement plus faible
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 10.0
    LAMBDA_LATENT = 150.0
    LAMBDA_ADV = 1.0
    LAMBDA_FM = 10.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    MODEL_SAVE_PATH = "./models_mtld_v1.9/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v1.9/"
    SAVE_EPOCH_INTERVAL = 10
    
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = "" 

    # --- Configuration de la Surveillance GPU ---
    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_POWER_THRESHOLD_PERCENT = 95
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300

# --- 2. Préparation des Données (MODIFIÉE pour la normalisation adaptative) ---

class AnimeFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None, config=None):
        self.root_dir, self.sequence_length, self.transform, self.config = root_dir, sequence_length, transform, config
        self.sequences, self.cumulative_lengths, self.preloaded_data = [], [], None
        if not os.path.isdir(root_dir): raise FileNotFoundError(f"Répertoire racine non trouvé : {root_dir}")
        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not arc_dirs: raise FileNotFoundError(f"Aucun sous-dossier (arc) trouvé dans {root_dir}")
        print(f"Détection de {len(arc_dirs)} arcs potentiels : {arc_dirs}")
        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            image_paths = sorted(glob.glob(os.path.join(arc_path, "*.png")), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_possible_sequences = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_possible_sequences
                self.cumulative_lengths.append(total_valid_sequences)
                print(f"  -> Arc '{arc_dir}' validé : {len(image_paths)} images, {num_possible_sequences} séquences.")
            else:
                print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court).")
        if not self.sequences: raise ValueError("Aucun arc valide trouvé.")
        print(f"\nDataset initialisé : {len(self.sequences)} arcs, {self.total_sequences()} séquences au total.")
        if self.config and self.config.PRELOAD_DATASET_IN_RAM: self._preload_images()

    def _preload_images(self):
        print("\n--- Pré-chargement du dataset en RAM ---")
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        num_images = len(all_paths)
        if num_images == 0: print("Aucune image à pré-charger."); return
        c, h, w = self.config.IMG_CHANNELS, self.config.IMG_SIZE, self.config.IMG_SIZE
        total_mb = (num_images * c * h * w * 4) / (1024**2)
        print(f"Chargement de {num_images} images uniques... Estimation RAM : {total_mb:.2f} Mo.")
        self.preloaded_data = {}
        for path in tqdm(all_paths, desc="Pré-chargement des images"):
            try:
                img = Image.open(path).convert("RGB")
                if self.transform: self.preloaded_data[path] = self.transform(img)
            except Exception as e: print(f"AVERTISSEMENT: Impossible de charger {path}. Erreur: {e}")
        print("--- Pré-chargement terminé ---")

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    def total_sequences(self): return self.__len__()
    def __getitem__(self, idx):
        arc_index = next(i for i, total in enumerate(self.cumulative_lengths) if idx < total)
        local_start_idx = idx - (self.cumulative_lengths[arc_index - 1] if arc_index > 0 else 0)
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        if self.preloaded_data:
            images = [self.preloaded_data[p] for p in sequence_paths if p in self.preloaded_data]
            if len(images) != self.sequence_length: raise RuntimeError(f"Images manquantes pour l'index {idx} en RAM.")
        else:
            images = [self.transform(Image.open(p).convert("RGB")) for p in sequence_paths]
        return torch.stack(images)

def get_dataloader(config):
    # MODIFIÉ: La normalisation s'adapte à l'encodeur choisi.
    # DINO a été pré-entraîné sur ImageNet et requiert cette normalisation spécifique.
    if config.USE_SVG_ENCODER:
        print("Utilisation de la normalisation ImageNet pour l'encodeur SVG (DINO).")
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print("Utilisation de la normalisation standard [-1, 1] pour l'encodeur CNN.")
        norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    dataset = AnimeFrameDataset(root_dir=config.DATASET_PATH, sequence_length=config.TRAINING_SEQUENCE_LENGTH, transform=transform, config=config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    return dataloader, dataset.total_sequences()

# --- 3. Architecture du Modèle (AJOUT DE L'ENCODEUR SVG) ---

# CONSERVÉ: L'encodeur CNN original, utilisé si USE_SVG_ENCODER = False
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(1024 * (img_size // 64) * (img_size // 64), latent_dim)
        )
    def forward(self, x):
        return self.model(x)

# NOUVEAU: Encodeur résiduel pour le style, plus petit que l'encodeur original
class ResidualEncoder(nn.Module):
    def __init__(self, in_channels, residual_dim, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 16) * (img_size // 16), residual_dim)
        )
    def forward(self, x):
        return self.model(x)

# NOUVEAU: L'encodeur SVG qui combine DINO (sémantique) et l'encodeur résiduel (style)
class SVGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("Initialisation de l'encodeur SVG...")
        # 1. Charger le backbone sémantique DINO
        try:
            self.dino_encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
            self.dino_dim = self.dino_encoder.embed_dim # Dimension du token [CLS] (384 pour ViT-S/16)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle DINO depuis torch.hub. Vérifiez votre connexion internet. Erreur : {e}")
        
        # 2. Geler les poids de DINO
        print(f"Backbone DINO (ViT-S/16) chargé. Dimension de sortie : {self.dino_dim}. Poids gelés.")
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        self.dino_encoder.eval()

        # 3. Créer l'encodeur résiduel entraînable
        self.residual_encoder = ResidualEncoder(config.IMG_CHANNELS, config.RESIDUAL_DIM, config.IMG_SIZE)
        print(f"Encodeur résiduel créé. Dimension de sortie : {config.RESIDUAL_DIM}.")

        # 4. Créer la tête de projection pour fusionner les deux sources d'information
        self.projection_head = nn.Sequential(
            nn.Linear(self.dino_dim + config.RESIDUAL_DIM, config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM)
        )
        print(f"Tête de projection créée. Dimension finale de l'espace latent : {config.LATENT_DIM}.")

    def forward(self, x):
        # DINO requiert des images de taille 224x224, on redimensionne à la volée
        # Cela évite de changer tout le pipeline de données pour une seule passe
        if x.shape[-1] != 224:
            x_dino = transforms.functional.resize(x, size=(224, 224), antialias=True)
        else:
            x_dino = x

        with torch.no_grad():
            semantic_features = self.dino_encoder(x_dino) # Sortie [B, 384]

        residual_features = self.residual_encoder(x) # Sortie [B, RESIDUAL_DIM]
        
        combined_features = torch.cat([semantic_features, residual_features], dim=1)
        
        latent_vector = self.projection_head(combined_features)
        return latent_vector

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, img_size):
        super().__init__()
        self.feature_map_size = img_size // 64
        self.decoder_input_size = 1024 * self.feature_map_size * self.feature_map_size
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
        if is_sequence: b, s, d = z.shape; z = z.view(b * s, d)
        x = self.decoder_input(z)
        x = x.view(-1, 1024, self.feature_map_size, self.feature_map_size)
        img = self.model(x)
        if is_sequence: _, c, h, w = img.shape; img = img.view(b, s, c, h, w)
        return img

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        def block(i, o, s=2, n=True):
            l = [nn.utils.spectral_norm(nn.Conv2d(i, o, 4, s, 1))]
            if n: l.append(nn.InstanceNorm2d(o))
            l.append(nn.LeakyReLU(0.2, inplace=True))
            return l
        self.model = nn.Sequential(*block(in_channels,64,n=False), *block(64,128), *block(128,256), *block(256,512,s=1), nn.utils.spectral_norm(nn.Conv2d(512,1,4,1,1)))
    def forward(self, img, extract_features=False):
        feats = []; x = img
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and extract_features: feats.append(x)
        return x, feats[:4]

class ConditionalLatentTrajectoryGenerator(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.z_to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)
    def forward(self, z_start, max_len):
        h0 = self.z_to_hidden(z_start).unsqueeze(0).repeat(2, 1, 1)
        gru_input = z_start.unsqueeze(1)
        outputs = []
        for _ in range(max_len):
            gru_output, h0 = self.gru(gru_input, h0)
            next_z = self.head(gru_output)
            outputs.append(next_z)
            gru_input = next_z
        return torch.cat(outputs, dim=1)

class MTLD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # MODIFIÉ: Choix conditionnel de l'encodeur
        if config.USE_SVG_ENCODER:
            self.encoder = SVGEncoder(config)
        else:
            self.encoder = Encoder(config.IMG_CHANNELS, config.LATENT_DIM, config.IMG_SIZE)
        
        self.decoder = Decoder(config.LATENT_DIM, config.IMG_CHANNELS, config.IMG_SIZE)
        self.trajectory_generator = ConditionalLatentTrajectoryGenerator(config.GRU_HIDDEN_DIM, config.LATENT_DIM)
        self.discriminator = PatchDiscriminator(config.IMG_CHANNELS)

# --- 4. Boucle d'Entraînement ---

def save_checkpoint(epoch, model, opt_g, opt_d, scaler_g, scaler_d, config):
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer_g_state_dict': opt_g.state_dict(), 'optimizer_d_state_dict': opt_d.state_dict(),
             'scaler_g_state_dict': scaler_g.state_dict(), 'scaler_d_state_dict': scaler_d.state_dict()}
    filename = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v1.9_checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"\nCheckpoint sauvegardé : {filename}")

def train_mtld():
    config = Config()
    print(f"--- MTLD v1.9 Entraînement ---")
    print(f"Utilisation du device : {config.DEVICE}")
    print(f"Architecture encodeur : {'SVG (DINO + Résiduel)' if config.USE_SVG_ENCODER else 'CNN Original'}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    dataloader, _ = get_dataloader(config)
    model = MTLD(config).to(config.DEVICE)
    
    # MODIFIÉ: Les paramètres du générateur incluent maintenant l'encodeur (partiellement) entraînable
    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.trajectory_generator.parameters())
    d_params = list(model.discriminator.parameters())
    opt_g = optim.Adam(filter(lambda p: p.requires_grad, g_params), lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(d_params, lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    loss_l1, loss_mse = nn.L1Loss(), nn.MSELoss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    scaler_g = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    scaler_d = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    start_epoch = 0
    
    if config.RESUME_TRAINING and os.path.isfile(config.CHECKPOINT_TO_RESUME):
        print(f"Reprise depuis : {config.CHECKPOINT_TO_RESUME}")
        ckpt = torch.load(config.CHECKPOINT_TO_RESUME, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        opt_g.load_state_dict(ckpt['optimizer_g_state_dict'])
        opt_d.load_state_dict(ckpt['optimizer_d_state_dict'])
        scaler_g.load_state_dict(ckpt['scaler_g_state_dict'])
        scaler_d.load_state_dict(ckpt['scaler_d_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Reprise à l'époque {start_epoch}.")

    gpu_monitor = GPUMonitor(config)

    print("\nDébut de l'entraînement...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape
                priming_img, future_imgs = real_seq_imgs[:, 0], real_seq_imgs[:, 1:]
                
                # --- Générateur ---
                opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=True):
                    z_start_true = model.encoder(priming_img)
                    with torch.no_grad():
                        z_future_true = model.encoder(future_imgs.reshape(-1,c,h,w)).view(b,s-1,-1)
                    
                    z_future_pred = model.trajectory_generator(z_start_true, max_len=s-1)
                    z_pred_full = torch.cat([z_start_true.unsqueeze(1), z_future_pred], dim=1)
                    fake_seq = model.decoder(z_pred_full)
                    
                    fake_flat = fake_seq.view(b*s,c,h,w)
                    real_flat = real_seq_imgs.view(b*s,c,h,w)
                    
                    loss_lat = loss_mse(z_future_pred, z_future_true)
                    loss_l1_rec = loss_l1(fake_seq, real_seq_imgs)
                    loss_lpips_rec = loss_lpips_vgg(fake_flat, real_flat).mean()
                    
                    fake_d_out, fake_feats = model.discriminator(fake_flat, extract_features=True)
                    loss_adv = loss_mse(fake_d_out, torch.ones_like(fake_d_out))
                    
                    _, real_feats = model.discriminator(real_flat.detach(), extract_features=True)
                    loss_fm = sum(loss_l1(ff, rf.detach()) for ff, rf in zip(fake_feats, real_feats))
                    
                    loss_g = (config.LAMBDA_LATENT * loss_lat + config.LAMBDA_REC_L1 * loss_l1_rec + 
                              config.LAMBDA_REC_LPIPS * loss_lpips_rec + config.LAMBDA_ADV * loss_adv + 
                              config.LAMBDA_FM * loss_fm)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g); scaler_g.update()

                # --- Discriminateur ---
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=True):
                    loss_d_real = loss_mse(model.discriminator(real_flat.detach())[0], torch.ones_like(fake_d_out))
                    loss_d_fake = loss_mse(model.discriminator(fake_flat.detach())[0], torch.zeros_like(fake_d_out))
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d); scaler_d.update()

                pbar.set_postfix({"L_Lat":f"{loss_lat.item():.3f}","L_Rec":f"{loss_l1_rec.item():.3f}","L_Adv":f"{loss_adv.item():.3f}","L_D":f"{loss_d.item():.3f}"})
                
                if i > 0 and i % 20 == 0:
                    status = gpu_monitor.check()
                    if status == "shutdown":
                        print("Arrêt d'urgence. Sauvegarde..."); save_checkpoint(epoch + 1, model, opt_g, opt_d, scaler_g, scaler_d, config)
                        gpu_monitor.shutdown(); exit()
            
            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, opt_g, opt_d, scaler_g, scaler_d, config)
                with torch.no_grad(), torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=True):
                    fake_eval = model.decoder(z_pred_full)
                    comp = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_eval[0].unsqueeze(0)], dim=0)
                    grid = comp.permute(1,0,2,3,4).reshape(-1,c,h,w)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comp_epoch_{epoch+1}.png")
                    utils.save_image(grid, grid_path, nrow=s, normalize=True)
                    print(f"Image de comparaison sauvegardée : {grid_path}")
    finally:
        gpu_monitor.shutdown()
    print("Entraînement terminé.")

# --- 5. Génération de Séquence ---
# (Aucune modification nécessaire, la logique de chargement de modèle est déjà compatible)

if __name__ == '__main__':
    train_mtld()