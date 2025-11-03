# -*- coding: utf-8 -*-
"""
VAE-Flow: Modèle Variationnel à Dynamique Latente par Flots Normalisants
pour la Génération Séquentielle d'Anime

Version: 2.1 (Correction de la Distribution de Base)
Auteur: Votre Expert en IA
Date: 03 novembre 2025
Description:
Cette version corrige une instabilité numérique liée à la distribution de base
`torch.distributions.MultivariateNormal`.

CORRECTION CLÉ :
- Remplacement de `MultivariateNormal` par une distribution `torch.distributions.Normal`
  factorisée. C'est mathématiquement équivalent pour une distribution de base
  gaussienne avec une covariance identité, mais numériquement plus stable et
  robuste dans PyTorch.
- Les calculs de log-probabilité dans la boucle d'entraînement sont adaptés en
  ajoutant `.sum(dim=-1)` pour agréger les probabilités des dimensions latentes
  indépendantes.
- La méthode d'échantillonnage dans la fonction de génération est mise à jour.

Cette modification résout l'erreur `ValueError` lors du calcul de la log-vraisemblance
et améliore la stabilité générale de l'entraînement.
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

try:
    import pynvml
except ImportError:
    print("="*80); print("AVERTISSEMENT: pynvml non trouvé. La surveillance GPU est désactivée."); print("!pip install nvidia-ml-py"); print("="*80)
    pynvml = None

class GPUMonitor: # Conservé tel quel
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
        return "continue"
    def _handle_alert(self):
        current_time = time.time()
        self.recent_alerts.append(current_time)
        while self.recent_alerts and current_time - self.recent_alerts[0] > self.config.GPU_SHUTDOWN_WINDOW_S: self.recent_alerts.popleft()
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
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset"
    PRELOAD_DATASET_IN_RAM = True
    
    IMG_SIZE = 128
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 12
    LATENT_DIM = 256
    
    NUM_FLOW_BLOCKS = 6
    COUPLING_HIDDEN_DIM = 512

    EPOCHS = 150
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    BETA1 = 0.9
    BETA2 = 0.999
    
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 10.0
    LAMBDA_PRIOR = 1.0
    LAMBDA_DYNAMICS = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    MODEL_SAVE_PATH = "./models_vaeflow_v2.1/"
    OUTPUT_SAVE_PATH = "./outputs_vaeflow_v2.1/"
    SAVE_EPOCH_INTERVAL = 10
    
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = ""

    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300

# --- 2. Préparation des Données (Conservé tel quel) ---

class AnimeFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None, config=None):
        self.root_dir = root_dir; self.sequence_length = sequence_length; self.transform = transform; self.config = config
        self.sequences = []; self.cumulative_lengths = []; self.preloaded_data = None
        if not os.path.isdir(root_dir): raise FileNotFoundError(f"Répertoire racine non trouvé : {root_dir}")
        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not arc_dirs: raise FileNotFoundError(f"Aucun sous-dossier trouvé dans {root_dir}")
        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            image_paths = sorted(glob.glob(os.path.join(arc_path, "*.png")), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_possible_sequences = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_possible_sequences
                self.cumulative_lengths.append(total_valid_sequences)
        if not self.sequences: raise ValueError("Aucun arc valide trouvé.")
        if self.config and self.config.PRELOAD_DATASET_IN_RAM: self._preload_images()
    def _preload_images(self):
        print("\n--- Pré-chargement du dataset en RAM ---")
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        num_images = len(all_paths)
        if num_images == 0: return
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
    def __len__(self): return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    def total_sequences(self): return self.__len__()
    def __getitem__(self, idx):
        arc_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_start_idx = idx - (self.cumulative_lengths[arc_index-1] if arc_index > 0 else 0)
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        if self.preloaded_data:
            images = [self.preloaded_data[p] for p in sequence_paths if p in self.preloaded_data]
            if len(images) != self.sequence_length: raise RuntimeError(f"Images manquantes pour l'index {idx}")
        else:
            images = [self.transform(Image.open(p).convert("RGB")) for p in sequence_paths]
        return torch.stack(images)

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = AnimeFrameDataset(root_dir=config.DATASET_PATH, sequence_length=config.TRAINING_SEQUENCE_LENGTH, transform=transform, config=config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    return dataloader, dataset.total_sequences()

# --- 3. Architecture du Modèle (VAE-Flow) ---

class ResidualCouplingBlock(nn.Module):
    def __init__(self, dim, hidden_dim, condition_dim=None):
        super().__init__()
        self.dim = dim
        self.d_half = dim // 2
        self.d_other = dim - self.d_half
        input_net_dim = self.d_half + (condition_dim if condition_dim is not None else 0)
        self.s_t_network = nn.Sequential(
            nn.Linear(input_net_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.d_other * 2)
        )
    def forward(self, x, condition=None):
        x_a, x_b = x.chunk(2, dim=-1)
        net_input = torch.cat([x_a, condition], dim=-1) if condition is not None else x_a
        s_t = self.s_t_network(net_input)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)
        y_b = x_b * torch.exp(s) + t
        y = torch.cat([x_a, y_b], dim=-1)
        log_det_J = s.sum(dim=-1)
        return y, log_det_J
    def inverse(self, y, condition=None):
        y_a, y_b = y.chunk(2, dim=-1)
        net_input = torch.cat([y_a, condition], dim=-1) if condition is not None else y_a
        s_t = self.s_t_network(net_input)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)
        x_b = (y_b - t) * torch.exp(-s)
        x = torch.cat([y_a, x_b], dim=-1)
        log_det_J = -s.sum(dim=-1)
        return x, log_det_J

class NormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim, n_blocks, condition_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualCouplingBlock(dim, hidden_dim, condition_dim) for _ in range(n_blocks)])
        self.permutations = [torch.randperm(dim) for _ in range(n_blocks)]
    def forward(self, x, condition=None):
        log_det_J_total = 0
        for i, block in enumerate(self.blocks):
            p = self.permutations[i].to(x.device)
            x = x[:, p]
            x, log_det_J = block(x, condition)
            log_det_J_total += log_det_J
        return x, log_det_J_total
    def inverse(self, u, condition=None):
        log_det_J_total = 0
        for i, block in reversed(list(enumerate(self.blocks))):
            u, log_det_J = block.inverse(u, condition)
            p = self.permutations[i].to(u.device)
            inv_p = torch.argsort(p)
            u = u[:, inv_p]
            log_det_J_total += log_det_J
        return u, log_det_J_total

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        feature_map_size = img_size // 16
        self.fc_mu = nn.Linear(512 * feature_map_size * feature_map_size, latent_dim)
        self.fc_logvar = nn.Linear(512 * feature_map_size * feature_map_size, latent_dim)
    def forward(self, x):
        h = self.model(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, img_size):
        super().__init__()
        feature_map_size = img_size // 16
        self.decoder_input_size = 512 * feature_map_size * feature_map_size
        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh(),
        )
        self.feature_map_size = feature_map_size
    def forward(self, z):
        is_sequence = z.dim() == 3
        if is_sequence: b, s, d = z.shape; z = z.view(b * s, d)
        x = self.decoder_input(z)
        x = x.view(-1, 512, self.feature_map_size, self.feature_map_size)
        img = self.model(x)
        if is_sequence: _, c, h, w = img.shape; img = img.view(b, s, c, h, w)
        return img

class VAEFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = VariationalEncoder(config.IMG_CHANNELS, config.LATENT_DIM, config.IMG_SIZE)
        self.decoder = Decoder(config.LATENT_DIM, config.IMG_CHANNELS, config.IMG_SIZE)
        self.prior_flow = NormalizingFlow(config.LATENT_DIM, config.COUPLING_HIDDEN_DIM, config.NUM_FLOW_BLOCKS)
        self.dynamics_flow = NormalizingFlow(config.LATENT_DIM, config.COUPLING_HIDDEN_DIM, config.NUM_FLOW_BLOCKS, condition_dim=config.LATENT_DIM)

        # ### CORRECTION ###
        # Remplacement de MultivariateNormal par une distribution Normale factorisée
        # pour une meilleure stabilité numérique.
        self.register_buffer('base_dist_mean', torch.zeros(config.LATENT_DIM))
        self.register_buffer('base_dist_var', torch.ones(config.LATENT_DIM))

    @property
    def base_dist(self):
        return torch.distributions.Normal(self.base_dist_mean, self.base_dist_var)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, seq_imgs):
        b, s, c, h, w = seq_imgs.shape
        seq_imgs_flat = seq_imgs.view(b * s, c, h, w)
        mu, logvar = self.encoder(seq_imgs_flat)
        z_seq = self.reparameterize(mu, logvar).view(b, s, -1)
        recon_seq_imgs = self.decoder(z_seq)
        return recon_seq_imgs, z_seq, mu.view(b,s,-1), logvar.view(b,s,-1)

# --- 4. Boucle d'Entraînement ---

def save_checkpoint(epoch, model, optimizer, scaler, config):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    filename = os.path.join(config.MODEL_SAVE_PATH, f"vaeflow_v2.1_checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"\nCheckpoint sauvegardé : {filename}")

def train_vaeflow():
    config = Config()
    print(f"Utilisation du device : {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    dataloader, _ = get_dataloader(config)
    
    model = VAEFlow(config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    
    loss_l1 = nn.L1Loss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    
    scaler = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    start_epoch = 0
    
    if config.RESUME_TRAINING and os.path.isfile(config.CHECKPOINT_TO_RESUME):
        print(f"Reprise depuis : {config.CHECKPOINT_TO_RESUME}")
        checkpoint = torch.load(config.CHECKPOINT_TO_RESUME, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Reprise à l'époque {start_epoch}.")
    
    gpu_monitor = GPUMonitor(config)

    print("\nDébut de l'entraînement du modèle VAE-Flow...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == "cuda")):
                    recon_seq_imgs, z_seq, _, _ = model(real_seq_imgs)

                    loss_rec_l1 = loss_l1(recon_seq_imgs, real_seq_imgs)
                    loss_rec_lpips = loss_lpips_vgg(recon_seq_imgs.view(b*s, c, h, w), real_seq_imgs.view(b*s, c, h, w)).mean()
                    loss_reconstruction = config.LAMBDA_REC_L1 * loss_rec_l1 + config.LAMBDA_REC_LPIPS * loss_rec_lpips

                    z_0 = z_seq[:, 0, :]
                    u_0, log_det_J_prior = model.prior_flow.forward(z_0)
                    # ### CORRECTION ### : On somme les log-probabilités sur la dimension latente
                    log_p_z0 = model.base_dist.log_prob(u_0).sum(dim=-1) + log_det_J_prior
                    loss_prior = -log_p_z0.mean()

                    z_t = z_seq[:, :-1, :].reshape((s-1)*b, -1)
                    z_t_plus_1 = z_seq[:, 1:, :].reshape((s-1)*b, -1)
                    u_t_plus_1, log_det_J_dyn = model.dynamics_flow.forward(z_t_plus_1, condition=z_t)
                    # ### CORRECTION ### : On somme les log-probabilités sur la dimension latente
                    log_p_zt1_given_zt = model.base_dist.log_prob(u_t_plus_1).sum(dim=-1) + log_det_J_dyn
                    loss_dynamics = -log_p_zt1_given_zt.mean()

                    total_loss = loss_reconstruction + config.LAMBDA_PRIOR * loss_prior + config.LAMBDA_DYNAMICS * loss_dynamics

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_postfix({
                    "L_Rec": f"{loss_reconstruction.item():.3f}",
                    "L_Prior": f"{loss_prior.item():.3f}",
                    "L_Dyn": f"{loss_dynamics.item():.3f}",
                })
                
                if i > 0 and i % 50 == 0:
                    status = gpu_monitor.check()
                    if status == "shutdown":
                        print("Arrêt d'urgence. Sauvegarde...")
                        save_checkpoint(epoch + 1, model, optimizer, scaler, config)
                        gpu_monitor.shutdown()
                        exit()

            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, optimizer, scaler, config)
                with torch.no_grad():
                    model.eval()
                    comparison = torch.cat([real_seq_imgs[0], recon_seq_imgs[0]], dim=0)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                    utils.save_image(comparison, grid_path, nrow=s, normalize=True)
                    print(f"Image de comparaison sauvegardée : {grid_path}")
                    model.train()
    finally:
        gpu_monitor.shutdown()
    print("Entraînement terminé.")

# --- 5. Génération de Séquence ---

def generate_sequence_vaeflow(model_path, priming_image_path, num_frames_to_generate, config):
    print("--- Démarrage de la Génération de Séquence (VAE-Flow) ---")
    if not os.path.exists(model_path): print(f"ERREUR : Modèle non trouvé {model_path}"); return
    if not os.path.exists(priming_image_path): print(f"ERREUR : Image d'amorce non trouvée {priming_image_path}"); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_vaeflow")
    os.makedirs(output_dir_frames, exist_ok=True)
    
    model = VAEFlow(config).to(config.DEVICE)
    print(f"Chargement du modèle depuis : {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(config.DEVICE)

    print(f"Génération de {num_frames_to_generate} images...")
    
    generated_z = []
    with torch.no_grad():
        with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=(config.DEVICE == "cuda")):
            mu, logvar = model.encoder(priming_tensor)
            z_t = model.reparameterize(mu, logvar)
            generated_z.append(z_t)

            for _ in range(num_frames_to_generate - 1):
                # ### CORRECTION ### : Échantillonnage adapté à la nouvelle distribution
                # On échantillonne un batch de 1 vecteur de la distribution de base
                u_t = model.base_dist.sample(sample_shape=(1,))
                z_t, _ = model.dynamics_flow.inverse(u_t, condition=z_t)
                generated_z.append(z_t)
            
            z_full_seq = torch.cat(generated_z, dim=0).unsqueeze(0)
            generated_imgs_seq = model.decoder(z_full_seq).squeeze(0)

    video_path = os.path.join(config.OUTPUT_SAVE_PATH, "generated_sequence_vaeflow.mp4")
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
    gif_path = os.path.join(config.OUTPUT_SAVE_PATH, "generated_sequence_vaeflow.gif")
    pil_images_for_gif[0].save(gif_path, save_all=True, append_images=pil_images_for_gif[1:], duration=int(1000/24), loop=0)
    print(f"GIF sauvegardé : {gif_path}")


if __name__ == '__main__':
    config = Config()
    
    # --- Option 1: Entraîner le nouveau modèle VAE-Flow ---
    print("\n--- MODE ENTRAÎNEMENT (v2.1 VAE-Flow) ---")
    train_vaeflow()

    # --- Option 2: Générer avec le nouveau modèle VAE-Flow ---
    # print("\n--- MODE GÉNÉRATION (v2.1 VAE-Flow) ---")
    # model_file = "./models_vaeflow_v2.1/vaeflow_v2.1_checkpoint_epoch_150.pth" 
    # priming_image = "chemin/vers/votre/image.png" # REMPLACEZ CECI
    
    # if os.path.exists(model_file) and os.path.exists(priming_image):
    #     generate_sequence_vaeflow(
    #         model_path=model_file, 
    #         priming_image_path=priming_image, 
    #         num_frames_to_generate=100,
    #         config=config
    #     )
    # else:
    #     print("Veuillez spécifier un modèle entraîné et une image d'amorce valides.")