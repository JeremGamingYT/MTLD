
# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime
Variant sans GAN : TFC-NCD (Teacher-Free Consistency + Noise-Conditioned Decoder) + RPC (Ranked Patch Contrast)

Version: 2.0 (TFC-NCD + RPC, Pré-chargement RAM conservé)
Date: 10 novembre 2025
Auteur: Co-penseur IA

Résumé:
- Suppression du discriminateur et des pertes adversariales.
- Ajout d'un décodeur conditionné par bruit (NoiseConditionedDecoder) avec modulation FiLM.
- Ajout d'une perte de cohérence multi-bruit (consistency loss) pour stabiliser et accélérer l'entraînement.
- Ajout d'une perte de "ranked patch contrast" (RPC) via un projecteur perceptuel léger.
- Conservation du pipeline dataset + préchargement RAM + monitoring GPU + LPIPS.
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lpips
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque LPIPS non trouvée."); print("!pip install lpips"); print("="*80); raise

# AVERTISSEMENT: La bibliothèque 'pynvml' est dépréciée.
# Pour supprimer l'avertissement de PyTorch, il est recommandé de la remplacer par 'nvidia-ml-py':
# pip uninstall pynvml
# pip install nvidia-ml-py
try:
    import pynvml
except ImportError:
    print("="*80); print("AVERTISSEMENT: pynvml non trouvé. La surveillance GPU est désactivée."); print("!pip install nvidia-ml-py"); print("="*80)
    pynvml = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        except Exception:
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
    # Données
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset"
    PRELOAD_DATASET_IN_RAM = True

    # Image
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 16

    # Latents / modèle
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    NOISE_EMB_DIM = 32

    # Entraînement
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE_G = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    DEVICE = get_device()
    NUM_WORKERS = 4

    # Pondérations de pertes
    LAMBDA_REC_L1 = 25.0
    LAMBDA_REC_LPIPS = 5.0
    LAMBDA_LATENT = 150.0
    LAMBDA_CONSIST = 20.0
    LAMBDA_RPC = 5.0

    # RPC
    RPC_MARGIN = 0.4
    RPC_PATCH_SIZE = 32
    RPC_PROJ_DIM = 128

    # Logs / sauvegarde
    MODEL_SAVE_PATH = "./models_mtld_v2.0/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v2.0/"
    SAVE_EPOCH_INTERVAL = 10

    # Reprise
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = "./models_mtld_v2.0/mtld_v2.0_checkpoint_epoch_10.pth"

    # Monitoring GPU
    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_POWER_THRESHOLD_PERCENT = 95
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300


# --- 2. Préparation des Données (pré-chargement RAM) ---

class AnimeFrameDataset(Dataset):
    """
    Dataset gérant les séquences et capable de pré-charger toutes les images
    en RAM pour des performances maximales.
    """
    def __init__(self, root_dir, sequence_length, transform=None, config: Config=None):
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
                print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court pour {self.sequence_length}).")

        if not self.sequences:
            raise ValueError("Aucun arc valide (suffisamment long) n'a été trouvé dans le dataset.")
        print(f"\nDataset initialisé : {len(self.sequences)} arcs valides, {self.total_sequences()} séquences d'entraînement au total.")

        if self.config and self.config.PRELOAD_DATASET_IN_RAM:
            self._preload_images()

    def _preload_images(self):
        """Charge et transforme toutes les images uniques du dataset en RAM."""
        print("\n--- Pré-chargement du dataset en RAM ---")
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        num_images = len(all_paths)
        if num_images == 0:
            print("Aucune image à pré-charger."); return

        c, h, w = self.config.IMG_CHANNELS, self.config.IMG_SIZE, self.config.IMG_SIZE
        bytes_per_tensor = c * h * w * 4  # float32
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
        local_start_idx = idx if arc_index == 0 else idx - self.cumulative_lengths[arc_index - 1]
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]

        if self.preloaded_data:
            images = [self.preloaded_data[p] for p in sequence_paths if p in self.preloaded_data]
            if len(images) != self.sequence_length:
                raise RuntimeError(f"Des images manquent en RAM pour la séquence index {idx}.")
        else:
            images_pil = [Image.open(p).convert("RGB") for p in sequence_paths]
            images = [self.transform(img) for img in images_pil] if self.transform else images_pil

        return torch.stack(images)


def get_dataloader(config: Config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
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


# --- 3. Architecture du Modèle (Encoder, Trajectoire, NCD, Projecteur) ---

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        fm = img_size // 64
        self.fc = nn.Linear(1024 * fm * fm, latent_dim)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)


class NoiseConditionedDecoder(nn.Module):
    """
    Décodeur transpose conv modulé par un code bruit via FiLM (gamma,beta) par bloc.
    """
    def __init__(self, latent_dim: int, out_channels: int, noise_dim: int, img_size: int):
        super().__init__()
        self.img_size = img_size
        fm = img_size // 64
        self.decoder_input_size = 1024 * fm * fm
        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size)

        # Embedding bruit -> FiLM params (gamma,beta) pour chaque bloc
        channels = [1024, 512, 256, 128, 64]
        self.channels = channels
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, 2 * sum(channels))  # 2*(sum C): gammas et betas concaténés
        )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c)
            )
        self.up1 = up(1024, 1024)
        self.up2 = up(1024, 512)
        self.up3 = up(512, 256)
        self.up4 = up(256, 128)
        self.up5 = up(128, 64)
        self.out = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    @staticmethod
    def film(x, gamma, beta):
        return torch.relu_(gamma * x + beta)

    def forward(self, z, noise_code):
        # z: (B,S,D) ou (B,D), noise_code conforme
        seq = (z.dim() == 3)
        if seq:
            b, s, d = z.shape
            z = z.view(b*s, d)
            noise_code = noise_code.view(b*s, -1)
        x = self.decoder_input(z)
        fm = self.img_size // 64
        x = x.view(-1, 1024, fm, fm)

        gb = self.noise_mlp(noise_code)  # (B*, 2*sum(C))
        splits = torch.split(gb, [2*c for c in self.channels], dim=1)
        def gb_split(block, c):
            g = block[:, :c].view(-1, c, 1, 1)
            b = block[:, c:2*c].view(-1, c, 1, 1)
            return g, b

        g1,b1 = gb_split(splits[0], 1024)
        g2,b2 = gb_split(splits[1], 512)
        g3,b3 = gb_split(splits[2], 256)
        g4,b4 = gb_split(splits[3], 128)
        g5,b5 = gb_split(splits[4], 64)

        x = self.up1(x); x = self.film(x, g1, b1)
        x = self.up2(x); x = self.film(x, g2, b2)
        x = self.up3(x); x = self.film(x, g3, b3)
        x = self.up4(x); x = self.film(x, g4, b4)
        x = self.up5(x); x = self.film(x, g5, b5)
        img = torch.tanh(self.out(x))

        if seq:
            img = img.view(b, s, -1, self.img_size, self.img_size)
        return img


class PerceptualProjector(nn.Module):
    """
    Projecteur perceptuel léger pour embeddings de patches.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        # x en [0,1]
        f = self.net(x).flatten(1)
        e = self.head(f)
        return nn.functional.normalize(e, dim=1)


class ConditionalLatentTrajectoryGenerator(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.IMG_CHANNELS, config.LATENT_DIM, config.IMG_SIZE)
        self.decoder = NoiseConditionedDecoder(config.LATENT_DIM, config.IMG_CHANNELS, config.NOISE_EMB_DIM, config.IMG_SIZE)
        self.trajectory_generator = ConditionalLatentTrajectoryGenerator(config.GRU_HIDDEN_DIM, config.LATENT_DIM)
        self.proj = PerceptualProjector(config.RPC_PROJ_DIM)


# --- 4. Pertes (consistency & ranked patch contrast) ---

def sample_noise_code(shape, device):
    # shape: (B,S,noise_dim) ou (B,noise_dim)
    return torch.randn(*shape, device=device)

def degrade(imgs):
    # imgs: (B,S,C,H,W) ou (B,C,H,W) in [-1,1]
    is_seq = (imgs.dim()==5)
    x = imgs if not is_seq else imgs.reshape(-1, *imgs.shape[2:])
    with torch.no_grad():
        x = (x+1)/2  # -> [0,1]
        # flou moyen léger + bruit gaussien
        k = 3
        x_blur = nn.AvgPool2d(k, stride=1, padding=k//2)(x)
        noise = 0.03*torch.randn_like(x)
        x = torch.clamp(x_blur + noise, 0, 1)
        x = x*2 - 1
    return x if not is_seq else x.view_as(imgs)

def consistency_loss(decoder: NoiseConditionedDecoder, z_seq, noise_dim, device):
    # deux bruits -> mêmes images
    if z_seq.dim()==2:
        b = z_seq.size(0)
        n1 = sample_noise_code((b, noise_dim), device)
        n2 = sample_noise_code((b, noise_dim), device)
    else:
        b,s,_ = z_seq.shape
        n1 = sample_noise_code((b,s, noise_dim), device)
        n2 = sample_noise_code((b,s, noise_dim), device)
    y1 = decoder(z_seq, n1)
    y2 = decoder(z_seq, n2)
    return nn.functional.l1_loss(y1, y2)

def ranked_patch_contrast_loss(projector: PerceptualProjector, real_imgs, fake_imgs, margin: float, patch_size: int):
    # real_imgs, fake_imgs: (B,S,C,H,W) in [-1,1]
    b,s,c,h,w = fake_imgs.shape
    real = real_imgs.reshape(b*s, c, h, w)
    fake = fake_imgs.reshape(b*s, c, h, w)
    deg  = degrade(real)

    ps = patch_size
    def crop(x):
        H,W = x.shape[-2:]
        ys = torch.randint(0, H-ps+1, (x.size(0),), device=x.device)
        xs = torch.randint(0, W-ps+1, (x.size(0),), device=x.device)
        idx = torch.arange(x.size(0), device=x.device)
        return x[idx,:, ys:ys+ps, xs:xs+ps]

    r = crop(real); f = crop(fake); d = crop(deg)

    # Projecteur travaille en [0,1]
    er = projector((r+1)/2)
    ef = projector((f+1)/2)
    ed = projector((d+1)/2)

    # distance cos ~ 1 - cos_sim
    def dist(a,b): return 1 - (a*b).sum(-1)

    # 1) real vs fake: dist(real, real) < dist(real, fake) + margin  => forcer fake à se rapprocher de real
    loss1 = torch.relu(-(dist(er,er) - dist(er,ef)) + margin).mean()
    # 2) fake vs degraded: dist(fake, real) < dist(fake, degraded) + margin
    loss2 = torch.relu(-(dist(ef,er) - dist(ef,ed)) + margin).mean()
    return 0.5*(loss1+loss2)


# --- 5. Boucle d'Entraînement (sans GAN) ---

def save_checkpoint(epoch, model, opt_g, scaler, config: Config):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    filename = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v2.0_checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)
    print(f"\nCheckpoint sauvegardé : {filename}")

def train_mtld():
    config = Config()
    print(f"Utilisation du device : {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)

    dataloader, total_sequences = get_dataloader(config)

    model = MTLD(config).to(config.DEVICE)

    params = list(model.encoder.parameters()) + \
             list(model.decoder.parameters()) + \
             list(model.trajectory_generator.parameters()) + \
             list(model.proj.parameters())
    opt_g = optim.Adam(params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))

    loss_l1 = nn.L1Loss()
    loss_mse = nn.MSELoss()
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

    scaler = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))

    start_epoch = 0
    if config.RESUME_TRAINING:
        path = config.CHECKPOINT_TO_RESUME
        if os.path.isfile(path):
            print(f"Reprise de l'entraînement depuis le checkpoint : {path}")
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_g_state_dict' in checkpoint: opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Reprise à partir de l'époque {start_epoch}.")
            else:
                print("Ancien format de poids détecté. Chargement des poids du modèle uniquement.")
                model.load_state_dict(checkpoint); start_epoch = 0

    gpu_monitor = GPUMonitor(config)

    print("\nDébut de l'entraînement (TFC-NCD + RPC)...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape

                priming_img = real_seq_imgs[:, 0, :, :, :]
                future_imgs = real_seq_imgs[:, 1:, :, :, :]

                opt_g.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=("cuda" if config.DEVICE=="cuda" else "cpu"),
                                        dtype=torch.float16 if config.DEVICE=="cuda" else torch.bfloat16,
                                        enabled=True):
                    # Encodeurs (avec stop-grad sur cibles)
                    with torch.no_grad():
                        z_start_true = model.encoder(priming_img)
                        z_future_true = model.encoder(future_imgs.reshape(-1, c, h, w)).view(b, s - 1, -1)
                    # Trajectoire latente prédite
                    z_future_pred = model.trajectory_generator(z_start_true, max_len=s - 1)
                    z_pred_full_seq = torch.cat([z_start_true.unsqueeze(1), z_future_pred], dim=1)

                    # Décodage conditionné par bruit
                    noise_code = sample_noise_code((b, s, config.NOISE_EMB_DIM), config.DEVICE)
                    fake_seq_imgs = model.decoder(z_pred_full_seq, noise_code)

                    # Pertes de reconstruction
                    loss_latent = loss_mse(z_future_pred, z_future_true)
                    loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
                    fake_imgs_flat = fake_seq_imgs.view(b*s, c, h, w)
                    real_imgs_flat = real_seq_imgs.view(b*s, c, h, w)
                    loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()

                    # Pertes TFC & RPC
                    loss_consist = consistency_loss(model.decoder, z_pred_full_seq, config.NOISE_EMB_DIM, config.DEVICE)
                    loss_rpc = ranked_patch_contrast_loss(model.proj, real_seq_imgs, fake_seq_imgs, config.RPC_MARGIN, config.RPC_PATCH_SIZE)

                    loss_g = (
                        config.LAMBDA_REC_L1 * loss_rec_l1 +
                        config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                        config.LAMBDA_LATENT * loss_latent +
                        config.LAMBDA_CONSIST * loss_consist +
                        config.LAMBDA_RPC * loss_rpc
                    )

                scaler.scale(loss_g).backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt_g)
                scaler.update()

                pbar.set_postfix({
                    "L_Pred": f"{loss_latent.item():.3f}",
                    "L_Rec": f"{loss_rec_l1.item():.3f}",
                    "L_Cst": f"{loss_consist.item():.3f}",
                    "L_RPC": f"{loss_rpc.item():.3f}",
                })

                if i > 0 and i % 20 == 0:
                    status = gpu_monitor.check()
                    if status == "shutdown":
                        print("Arrêt d'urgence demandé. Sauvegarde du checkpoint...")
                        save_checkpoint(epoch + 1, model, opt_g, scaler, config)
                        gpu_monitor.shutdown()
                        raise SystemExit

            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, opt_g, scaler, config)
                with torch.no_grad():
                    model.eval()
                    with torch.amp.autocast(device_type=("cuda" if config.DEVICE=="cuda" else "cpu"),
                                            dtype=torch.float16 if config.DEVICE=="cuda" else torch.bfloat16,
                                            enabled=True):
                        # réutilise dernier z pour visualisation
                        noise_eval = sample_noise_code((b, s, config.NOISE_EMB_DIM), config.DEVICE)
                        fake_seq_imgs_eval = model.decoder(z_pred_full_seq, noise_eval)
                    comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_seq_imgs_eval[0].unsqueeze(0)], dim=0)
                    comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                    utils.save_image(comparison_flat, grid_path, nrow=s, normalize=True)
                    print(f"Image de comparaison sauvegardée : {grid_path}")
                    model.train()

    finally:
        gpu_monitor.shutdown()
    print("Entraînement terminé.")


# --- 6. Génération de Séquence ---

def generate_sequence(model_path, priming_image_path, num_frames_to_generate, config: Config):
    print("--- Démarrage de la Génération de Séquence (TFC-NCD) ---")
    if not os.path.exists(model_path):
        print(f"ERREUR : Fichier modèle non trouvé à {model_path}"); return
    if not os.path.exists(priming_image_path):
        print(f"ERREUR : Image d'amorce non trouvée à {priming_image_path}"); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_conditional")
    os.makedirs(output_dir_frames, exist_ok=True)

    gen_config = config
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
        with torch.amp.autocast(device_type=("cuda" if gen_config.DEVICE=="cuda" else "cpu"),
                                dtype=torch.float16 if gen_config.DEVICE=="cuda" else torch.bfloat16,
                                enabled=True):
            z_start = model.encoder(priming_tensor)
            z_future = model.trajectory_generator(z_start, max_len=num_frames_to_generate - 1)
            z_full_seq = torch.cat([z_start.unsqueeze(1), z_future], dim=1)
            noise_code = sample_noise_code((1, num_frames_to_generate, gen_config.NOISE_EMB_DIM), gen_config.DEVICE)
            generated_imgs_seq = model.decoder(z_full_seq, noise_code).squeeze(0)

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


# --- 7. Exécution (décommente le mode voulu) ---
if __name__ == '__main__':
    cfg = Config()
    # --- MODE ENTRAÎNEMENT ---
    print(f"\n--- MODE ENTRAÎNEMENT (v2.0 TFC-NCD + RPC) ---")
    train_mtld()

    # --- MODE GÉNÉRATION ---
    #print("\n--- MODE GÉNÉRATION (v2.0) ---")
    # Exemple: on cherche une image d'amorce depuis le dataset
    #try:
    #    dataset_tmp = AnimeFrameDataset(cfg.DATASET_PATH, 1, transform=transforms.Compose([
    #        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    #        transforms.ToTensor(),
    #        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #    ]), config=cfg)
    #    priming_image = dataset_tmp.sequences[0][0] 
    #    print(f"Utilisation de l'image d'amorce : {priming_image}")
    #except Exception as e:
    #    print(f"ATTENTION: Impossible de trouver une image d'amorce par défaut : {e}. Spécifiez un chemin valide.")
    #    priming_image = "frame_0001.png"  # à remplacer

    #model_file = os.path.join(cfg.MODEL_SAVE_PATH, "mtld_v2.0_checkpoint_epoch_100.pth")
    #if os.path.exists(priming_image) and os.path.exists(model_file):
    #    generate_sequence(
    #        model_path=model_file,
    #        priming_image_path=priming_image,
    #        num_frames_to_generate=100,
    #        config=cfg
    #    )
    #else:
    #    print("Note: modèle ou image d'amorce manquants pour la génération de démo.")
