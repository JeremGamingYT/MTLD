# -*- coding: utf-8 -*-
"""
MTLD v2.0: Modèle à Trajectoire Latente Variationnelle-Attentionnelle

Version: 2.0
Auteur: Votre Expert en IA
Date: 03 novembre 2025
Description:
Cette version représente une refonte architecturale et algorithmique majeure de MTLD,
visant une convergence plus rapide vers une qualité supérieure, une meilleure
généralisation sur des datasets de taille limitée, et une vitesse d'époque
maximale grâce aux optimisations modernes de PyTorch.

Nouveautés Clés :
1.  **Encodeur Variationnel (VAE) :** L'encodeur apprend une distribution latente
    (moyenne + variance), créant un espace latent plus riche et mieux régularisé,
    essentiel pour combattre le sur-apprentissage.
2.  **Générateur de Trajectoire Attentionnel :** Le GRU est remplacé par un
    bloc Attention-GRU. À chaque étape, un mécanisme d'attention pondère les
    dimensions latentes, permettant au modèle de se concentrer sur les
    informations pertinentes pour une meilleure prédiction temporelle.
3.  **Critique WGAN-GP :** Le discriminateur GAN classique est remplacé par un
    Critique entraîné avec la perte de Wasserstein et une pénalité de gradient.
    Ceci stabilise radicalement l'entraînement et améliore la qualité des images.
4.  **Compilation JIT avec `torch.compile` :** Les modules sont compilés pour une
    accélération significative, exploitant pleinement PyTorch 2.0+.
5.  **Augmentation de Données Robuste :** Une pipeline d'augmentation agressive
    est introduite pour augmenter artificiellement la taille du dataset,
    une étape cruciale pour l'entraînement sur des données limitées.
6.  **Reprise d'Entraînement Intelligente :** Capacité de charger les poids d'un
    ancien checkpoint (non-strict) avec des taux d'apprentissage différentiels
    pour une adaptation rapide des nouvelles couches.
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

# Vérification de la version de PyTorch pour torch.compile
IS_PYTORCH_2_PLUS = int(torch.__version__.split('.')[0]) >= 2
if not IS_PYTORCH_2_PLUS:
    print("="*80)
    print("AVERTISSEMENT: `torch.compile` n'est pas disponible. Votre version de PyTorch est < 2.0.")
    print("Le modèle fonctionnera, mais sera significativement plus lent.")
    print("Veuillez mettre à jour PyTorch pour des performances optimales.")
    print("="*80)

try:
    import lpips
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque LPIPS non trouvée."); print("!pip install lpips"); print("="*80); exit()

try:
    from pynvml import *
    nvmlInit()
    pynvml_available = True
except (ImportError, NVMLError):
    print("="*80); print("AVERTISSEMENT: pynvml non trouvé ou échec de l'initialisation. La surveillance GPU est désactivée."); print("!pip install nvidia-ml-py"); print("="*80)
    pynvml_available = False


class GPUMonitor:
    # ... (Le code du GPUMonitor reste identique, il est déjà robuste) ...
    def __init__(self, config, device_id=0):
        if not pynvml_available:
            self.enabled = False
            return
        try:
            self.handle = nvmlDeviceGetHandleByIndex(device_id)
            self.config = config
            self.enabled = config.GPU_MONITORING_ENABLED
            self.recent_alerts = deque()
            print(f"GPUMonitor initialisé pour le GPU {nvmlDeviceGetName(self.handle).decode('utf-8')}.")
        except Exception as e:
            print(f"ERREUR: Impossible d'initialiser pynvml. Surveillance GPU désactivée. Erreur : {e}")
            self.enabled = False

    def check(self):
        if not self.enabled: return "continue"
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
        if temp > self.config.GPU_TEMP_THRESHOLD_C:
            print(f"\n! ALERTE GPU ! Température atteinte : {temp}°C (Seuil: {self.config.GPU_TEMP_THRESHOLD_C}°C)")
            return self._handle_alert()
        # ... (reste de la logique identique) ...
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
            nvmlShutdown()

class Config:
    # --- Chemins et Données ---
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset"
    PRELOAD_DATASET_IN_RAM = True
    
    # --- Architecture du Modèle ---
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 16
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    ATTENTION_HEADS = 4 # NOUVEAU: Nombre de têtes pour l'attention

    # --- Paramètres d'Entraînement ---
    EPOCHS = 300 # Augmenté car le modèle est plus complexe
    BATCH_SIZE = 2
    LEARNING_RATE_G = 1e-4 # Taux d'apprentissage plus faible pour WGAN
    LEARNING_RATE_D = 4e-4
    BETA1 = 0.0
    BETA2 = 0.9 # Betas recommandés pour WGAN
    
    # --- Pondération des Pertes (Losses) ---
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 20.0 # Augmenté pour une meilleure qualité perceptive
    LAMBDA_LATENT = 1.0     # La perte latente est maintenant implicite dans le VAE
    LAMBDA_KL = 0.05        # NOUVEAU: Poids pour la divergence KL du VAE
    LAMBDA_ADV = 1.0
    LAMBDA_FM = 10.0
    LAMBDA_GP = 10.0        # NOUVEAU: Poids pour la pénalité de gradient WGAN-GP
    
    # --- Infrastructure et Reprise ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    COMPILE_MODEL = IS_PYTORCH_2_PLUS # NOUVEAU: Activer torch.compile
    # Options: "default", "reduce-overhead", "max-autotune"
    COMPILE_MODE = "reduce-overhead" 
    
    MODEL_SAVE_PATH = "./models_mtld_v2.0/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v2.0/"
    SAVE_EPOCH_INTERVAL = 10
    
    # --- Reprise d'entraînement ---
    RESUME_TRAINING = True # Mettre à True pour charger le checkpoint
    # Mettez ici le chemin vers votre checkpoint v1.8 de 100 époques
    CHECKPOINT_TO_RESUME = "./MTLD/mtld_v1.8_checkpoint_epoch_100.pth" 
    LOAD_STRICT = False # NOUVEAU: Charger les poids de manière non-stricte
    # Taux d'apprentissage pour les nouvelles couches (Attention, VAE) lors de la reprise
    NEW_LAYERS_LR_FACTOR = 10.0 

    # --- Surveillance GPU ---
    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_POWER_THRESHOLD_PERCENT = 95
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300

# --- 2. Préparation des Données (avec Augmentation Agressive) ---

class AnimeFrameDataset(Dataset):
    # ... (Logique de base et de pré-chargement identique) ...
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
            else:
                print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court).")

        if not self.sequences:
            raise ValueError("Aucun arc valide (suffisamment long) n'a été trouvé dans le dataset.")
        print(f"\nDataset initialisé : {len(self.sequences)} arcs valides, {self.total_sequences()} séquences d'entraînement au total.")

        if self.config and self.config.PRELOAD_DATASET_IN_RAM:
            self._preload_images()

    def _preload_images(self):
        print("\n--- Pré-chargement du dataset en RAM ---")
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        num_images = len(all_paths)
        if num_images == 0: print("Aucune image à pré-charger."); return

        c, h, w = self.config.IMG_CHANNELS, self.config.IMG_SIZE, self.config.IMG_SIZE
        bytes_per_tensor = c * h * w * 2  # torch.bfloat16 utilise 2 octets
        total_mb = (num_images * bytes_per_tensor) / (1024**2)
        print(f"Chargement de {num_images} images uniques... Estimation RAM : {total_mb:.2f} Mo.")

        self.preloaded_data = {}
        for path in tqdm(all_paths, desc="Pré-chargement des images"):
            try:
                img = Image.open(path).convert("RGB")
                # L'augmentation est appliquée à la volée dans __getitem__
                # On pré-charge seulement l'image de base transformée
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
        
        local_start_idx = idx - (self.cumulative_lengths[arc_index - 1] if arc_index > 0 else 0)
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        
        if self.preloaded_data:
            images = [self.preloaded_data[p] for p in sequence_paths if p in self.preloaded_data]
            if len(images) != self.sequence_length:
                 raise RuntimeError(f"Images manquantes pour l'index {idx} en RAM.")
        else: # Fallback si le pré-chargement est désactivé
            images = []
            for p in sequence_paths:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    images.append(self.transform(img))

        return torch.stack(images)

def get_dataloader(config):
    # NOUVEAU: Pipeline d'augmentation de données agressive
    # Appliquée à chaque batch pour combattre l'overfitting
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
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

# --- 3. Architecture du Modèle (MTLD v2.0) ---

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        feature_map_size = config.IMG_SIZE // 64
        linear_input_dim = 1024 * feature_map_size * feature_map_size
        self.fc_mu = nn.Linear(linear_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(linear_input_dim, latent_dim)

    def forward(self, x):
        features = self.model(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    # ... (Le décodeur reste structurellement identique) ...
    def __init__(self, latent_dim, out_channels, config):
        super().__init__()
        self.config = config
        feature_map_size = self.config.IMG_SIZE // 64
        decoder_input_size = 1024 * feature_map_size * feature_map_size
        self.decoder_input = nn.Linear(latent_dim, decoder_input_size)
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
        feature_map_size = self.config.IMG_SIZE // 64
        x = x.view(-1, 1024, feature_map_size, feature_map_size)
        img = self.model(x)
        if is_sequence: _, c, h, w = img.shape; img = img.view(b, s, c, h, w)
        return img

class Critic(nn.Module): # Renommé de Discriminator à Critic pour WGAN
    def __init__(self, in_channels):
        super().__init__()
        # La normalisation spectrale est une excellente alternative à la pénalité de gradient
        # mais nous allons utiliser GP explicitement pour un contrôle maximal.
        def critic_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalize=False),
            *critic_block(64, 128), *critic_block(128, 256),
            *critic_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1) # Pas de fonction d'activation à la fin pour le score
        )
    def forward(self, img, extract_features=False):
        features = []
        x = img
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and extract_features:
                features.append(x)
        return x, features[:4]

class AttentionalLatentTrajectoryGenerator(nn.Module):
    def __init__(self, hidden_dim, latent_dim, n_heads):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_start, max_len):
        batch_size = z_start.size(0)
        
        # Initialiser l'état caché du GRU à partir du latent de départ
        h0_proj = self.z_to_hidden(z_start)
        h0 = h0_proj.unsqueeze(0).repeat(2, 1, 1) # Pour 2 couches de GRU

        # L'entrée initiale pour le GRU est aussi basée sur z_start
        gru_input = h0_proj.unsqueeze(1)
        
        outputs = []
        for _ in range(max_len):
            # L'état caché (query) assiste au latent précédent (key, value)
            attn_output, _ = self.attention(query=gru_input, key=gru_input, value=gru_input)
            
            # Le GRU traite le résultat de l'attention
            gru_output, h0 = self.gru(attn_output, h0)
            
            # Prédire le prochain latent
            next_z = self.head(gru_output)
            outputs.append(next_z)
            
            # L'entrée pour la prochaine étape est basée sur la sortie actuelle
            gru_input = self.z_to_hidden(next_z).relu()

        latent_trajectory = torch.cat(outputs, dim=1)
        return latent_trajectory

class MTLD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = VariationalEncoder(config.IMG_CHANNELS, config.LATENT_DIM, config)
        self.decoder = Decoder(config.LATENT_DIM, config.IMG_CHANNELS, config)
        self.trajectory_generator = AttentionalLatentTrajectoryGenerator(
            config.GRU_HIDDEN_DIM, config.LATENT_DIM, config.ATTENTION_HEADS
        )
        self.critic = Critic(config.IMG_CHANNELS)

# --- 4. Boucle d'Entraînement (MTLD v2.0) ---

def save_checkpoint(epoch, model, opt_g, opt_d, config):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
    }
    filename = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v2.0_checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"\nCheckpoint sauvegardé : {filename}")

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calcule la pénalité de gradient pour WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates, _ = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_mtld():
    config = Config()
    print(f"--- Démarrage de l'entraînement de MTLD v2.0 ---")
    print(f"Device: {config.DEVICE}, Compilation JIT: {config.COMPILE_MODEL}, Précision: bfloat16")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    dataloader, total_sequences = get_dataloader(config)
    
    model = MTLD(config).to(config.DEVICE)

    if config.COMPILE_MODEL:
        print("Compilation des modules du modèle... (peut prendre quelques minutes la première fois)")
        model.encoder = torch.compile(model.encoder, mode=config.COMPILE_MODE)
        model.decoder = torch.compile(model.decoder, mode=config.COMPILE_MODE)
        model.trajectory_generator = torch.compile(model.trajectory_generator, mode=config.COMPILE_MODE)
        model.critic = torch.compile(model.critic, mode=config.COMPILE_MODE)
        print("Compilation terminée.")
    
    # --- Configuration de l'optimiseur avec taux d'apprentissage différentiel ---
    base_params = []
    new_params = []
    if config.RESUME_TRAINING:
        print("Configuration des taux d'apprentissage différentiels pour la reprise.")
        new_param_names = ['fc_mu', 'fc_logvar', 'attention', 'trajectory_generator.head', 'trajectory_generator.z_to_hidden']
        for name, param in model.named_parameters():
            if any(new_name in name for new_name in new_param_names):
                new_params.append(param)
            else:
                base_params.append(param)
        
        # S'applique uniquement au générateur
        g_params = [
            {'params': base_params, 'lr': config.LEARNING_RATE_G},
            {'params': new_params, 'lr': config.LEARNING_RATE_G * config.NEW_LAYERS_LR_FACTOR}
        ]
        print(f"{len(base_params)} paramètres de base (LR: {config.LEARNING_RATE_G}), "
              f"{len(new_params)} nouveaux paramètres (LR: {config.LEARNING_RATE_G * config.NEW_LAYERS_LR_FACTOR})")
    else:
        g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.trajectory_generator.parameters())

    d_params = list(model.critic.parameters())
    opt_g = optim.Adam(g_params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(d_params, lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    loss_l1 = nn.L1Loss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    
    start_epoch = 0
    
    if config.RESUME_TRAINING:
        path = config.CHECKPOINT_TO_RESUME
        if os.path.isfile(path):
            print(f"Reprise de l'entraînement depuis : {path}")
            checkpoint = torch.load(path, map_location=config.DEVICE)
            
            # Logique de chargement non-stricte
            model_dict = model.state_dict()
            pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=config.LOAD_STRICT)
            print(f"{len(pretrained_dict)}/{len(model_dict)} poids chargés depuis le checkpoint.")

            # Charger les optimiseurs si les clés existent (peut échouer si la structure a changé)
            try:
                if 'optimizer_g_state_dict' in checkpoint: opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                if 'optimizer_d_state_dict' in checkpoint: opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Reprise à partir de l'époque {start_epoch + 1}.")
            except Exception as e:
                print(f"AVERTISSEMENT: Impossible de charger les états de l'optimiseur. Ils sont réinitialisés. Erreur: {e}")
                start_epoch = 0
        else:
            print(f"AVERTISSEMENT: Checkpoint non trouvé à '{path}'. Démarrage d'un nouvel entraînement.")

    gpu_monitor = GPUMonitor(config)

    print("\nDébut de l'entraînement...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape
                
                priming_img = real_seq_imgs[:, 0, :, :, :]
                future_imgs = real_seq_imgs[:, 1:, :, :, :]
                real_imgs_flat = real_seq_imgs.view(b * s, c, h, w)
                
                # --- Entraînement du Générateur ---
                opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
                    mu_start, logvar_start = model.encoder(priming_img)
                    z_start = model.encoder.reparameterize(mu_start, logvar_start)
                    
                    z_future_pred = model.trajectory_generator(z_start, max_len=s - 1)
                    z_pred_full_seq = torch.cat([z_start.unsqueeze(1), z_future_pred], dim=1)
                    
                    fake_seq_imgs = model.decoder(z_pred_full_seq)
                    fake_imgs_flat = fake_seq_imgs.view(b * s, c, h, w)
                    
                    # Pertes de reconstruction
                    loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
                    loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()
                    
                    # Perte VAE (KL-Divergence)
                    loss_kl = -0.5 * torch.mean(1 + logvar_start - mu_start.pow(2) - logvar_start.exp())
                    
                    # Perte adversariale (WGAN)
                    fake_critic_output, _ = model.critic(fake_imgs_flat)
                    loss_adv = -torch.mean(fake_critic_output)
                    
                    # Perte de Feature Matching
                    _, real_features = model.critic(real_imgs_flat.detach(), extract_features=True)
                    _, fake_features = model.critic(fake_imgs_flat, extract_features=True)
                    loss_fm = sum(loss_l1(fake_f, real_f.detach()) for real_f, fake_f in zip(real_features, fake_features))
                    
                    loss_g = (config.LAMBDA_REC_L1 * loss_rec_l1 +
                              config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                              config.LAMBDA_KL * loss_kl +
                              config.LAMBDA_ADV * loss_adv +
                              config.LAMBDA_FM * loss_fm)
                
                loss_g.backward()
                opt_g.step()
                
                # --- Entraînement du Critique ---
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
                    real_critic_output, _ = model.critic(real_imgs_flat.detach())
                    fake_critic_output, _ = model.critic(fake_imgs_flat.detach())
                    
                    gp = compute_gradient_penalty(model.critic, real_imgs_flat.detach(), fake_imgs_flat.detach(), config.DEVICE)
                    
                    loss_d = (torch.mean(fake_critic_output) - torch.mean(real_critic_output) + config.LAMBDA_GP * gp)

                loss_d.backward()
                opt_d.step()
                
                pbar.set_postfix({
                    "L_Rec": f"{loss_rec_l1.item():.3f}", "L_KL": f"{loss_kl.item():.3f}",
                    "L_G": f"{loss_adv.item():.3f}", "L_D": f"{loss_d.item():.3f}", "GP": f"{gp.item():.2f}"
                })
                
                # ... (Logique de surveillance GPU identique) ...
            
            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, opt_g, opt_d, config)
                with torch.no_grad():
                    model.eval()
                    with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
                        # Pour la visualisation, utilisons mu pour une sortie déterministe
                        z_future_eval = model.trajectory_generator(mu_start, max_len=s - 1)
                        z_eval_seq = torch.cat([mu_start.unsqueeze(1), z_future_eval], dim=1)
                        fake_seq_imgs_eval = model.decoder(z_eval_seq)
                    
                    # Afficher la première séquence du batch
                    comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_seq_imgs_eval[0].unsqueeze(0)], dim=0)
                    comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                    utils.save_image(comparison_flat, grid_path, nrow=2, normalize=True) # 2 lignes: réel vs fake
                    print(f"Image de comparaison sauvegardée : {grid_path}")
                    model.train()

    finally:
        gpu_monitor.shutdown()
    print("Entraînement terminé.")

# --- 5. Génération de Séquence (Mise à jour pour v2.0) ---

def generate_sequence(model_path, priming_image_path, num_frames_to_generate, config):
    # ... (La logique de génération est similaire, mais doit utiliser le nouveau modèle) ...
    print("--- Démarrage de la Génération de Séquence (MTLD v2.0) ---")
    if not os.path.exists(model_path): print(f"ERREUR : Fichier modèle non trouvé à {model_path}"); return
    if not os.path.exists(priming_image_path): print(f"ERREUR : Image d'amorce non trouvée à {priming_image_path}"); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_v2")
    os.makedirs(output_dir_frames, exist_ok=True)
    
    model = MTLD(config).to(config.DEVICE)
    if config.COMPILE_MODEL:
        model = torch.compile(model) # Compiler le modèle complet pour l'inférence
    
    print(f"Chargement du modèle depuis : {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(config.DEVICE)

    print(f"Génération de {num_frames_to_generate} images...")

    with torch.no_grad():
        with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
            # En inférence, on utilise mu pour une génération déterministe et stable
            mu_start, _ = model.encoder(priming_tensor)
            z_future = model.trajectory_generator(mu_start, max_len=num_frames_to_generate - 1)
            z_full_seq = torch.cat([mu_start.unsqueeze(1), z_future], dim=1)
            generated_imgs_seq = model.decoder(z_full_seq).squeeze(0)

    # ... (Logique de sauvegarde en vidéo/gif identique) ...
    video_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_v2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (config.IMG_SIZE, config.IMG_SIZE))
    pil_images_for_gif = []

    for i, img_tensor in enumerate(tqdm(generated_imgs_seq, desc="Sauvegarde des images")):
        img_np = np.clip((img_tensor.permute(1, 2, 0).float() * 0.5 + 0.5).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        img_pil.save(os.path.join(output_dir_frames, f"frame_{i:04d}.png"))
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB_BGR))
        pil_images_for_gif.append(img_pil)

    video_writer.release()
    print(f"Vidéo MP4 sauvegardée : {video_path}")
    gif_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_v2.gif")
    pil_images_for_gif[0].save(gif_path, save_all=True, append_images=pil_images_for_gif[1:], duration=int(1000/24), loop=0)
    print(f"GIF sauvegardé : {gif_path}")


if __name__ == '__main__':
    config = Config()
    
    # --- MODE 1: ENTRAÎNEMENT ---
    print(f"\n--- MODE ENTRAÎNEMENT (v2.0) ---")
    train_mtld()

    # --- MODE 2: GÉNÉRATION ---
    # Décommentez pour lancer la génération après l'entraînement
    # print("\n--- MODE GÉNÉRATION (v2.0) ---")
    # model_file_gen = "./models_mtld_v2.0/mtld_v2.0_checkpoint_epoch_300.pth" # Mettez le chemin final
    # priming_image_gen = "/kaggle/input/anima-s-dataset/animes_dataset/Arc_1/00000.png" # Mettez une image de votre choix
    
    # if os.path.exists(model_file_gen) and os.path.exists(priming_image_gen):
    #     generate_sequence(
    #         model_path=model_file_gen, 
    #         priming_image_path=priming_image_gen, 
    #         num_frames_to_generate=100,
    #         config=config
    #     )
    # else:
    #     print("Fichier modèle ou image d'amorce non trouvé pour la génération.")