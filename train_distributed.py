# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime

Version: 1.6 (Entraînement Distribué Multi-GPU avec DDP)
Date: 01 novembre 2025
"""

# --- 1. Importations et Configuration ---

import os
import glob
import re
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

# Importations pour l'entraînement distribué (DDP)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lpips
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque LPIPS non trouvée."); print("!pip install lpips"); print("="*80); exit()

class Config:
    # MODIFIÉ: Le chemin pointe maintenant vers le dossier parent contenant les arcs
    DATASET_PATH = "/root/.cache/kagglehub/datasets/jeremgaming099/anima-s-dataset/versions/4/animes_dataset" # Exemple: /path/to/your/dataset/
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 32
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    EPOCHS = 200
    # Le BATCH_SIZE est maintenant par GPU. La taille de batch globale sera BATCH_SIZE * N_GPUS
    BATCH_SIZE = 4 
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 4e-4
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_REC_L1 = 100.0
    LAMBDA_REC_LPIPS = 10.0
    LAMBDA_LATENT = 150.0
    LAMBDA_ADV = 1.0
    LAMBDA_FM = 10.0
    # Le DEVICE sera géré par le processus DDP
    NUM_WORKERS = 2 # Nombre de workers par processus GPU
    MODEL_SAVE_PATH = "./models_mtld_v1.6_ddp/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v1.6_ddp/"
    SAVE_EPOCH_INTERVAL = 10
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = ""

# --- Fonctions pour la gestion du processus distribué (DDP) ---

def setup_ddp(rank, world_size):
    """Initialise le groupe de processus distribué."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialise le groupe de processus. 'nccl' est le backend recommandé pour les GPU NVIDIA.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Nettoie et détruit le groupe de processus."""
    dist.destroy_process_group()


# --- 2. Préparation des Données (MODIFIÉE pour DDP) ---

class AnimeFrameDataset(Dataset):
    """
    Dataset repensé pour gérer une structure de dossiers où chaque sous-dossier
    est une séquence temporelle distincte (un "arc"). Inchangé par rapport à v1.5.
    """
    def __init__(self, root_dir, sequence_length, transform=None, is_master=False):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []
        self.cumulative_lengths = []

        if not os.path.isdir(root_dir):
            if is_master:
                raise FileNotFoundError(f"Le répertoire racine du dataset n'a pas été trouvé : {root_dir}")
            return

        # 1. Identifier les sous-dossiers (arcs)
        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not arc_dirs:
            if is_master:
                raise FileNotFoundError(f"Aucun sous-dossier (arc) trouvé dans {root_dir}")
            return
        
        if is_master: print(f"Détection de {len(arc_dirs)} arcs potentiels : {arc_dirs}")

        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            
            # 2. Collecter et trier les images pour chaque arc
            image_paths = sorted(
                glob.glob(os.path.join(arc_path, "*.png")),
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
            )

            # 3. Ne conserver que les arcs suffisamment longs
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_possible_sequences = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_possible_sequences
                self.cumulative_lengths.append(total_valid_sequences)
                if is_master: print(f"  -> Arc '{arc_dir}' validé : {len(image_paths)} images, {num_possible_sequences} séquences possibles.")
            else:
                if is_master: print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court pour une séquence de {self.sequence_length}).")

        if not self.sequences and is_master:
            raise ValueError("Aucun arc valide (suffisamment long) n'a été trouvé dans le dataset.")
            
        if is_master and self.sequences:
            print(f"\nDataset initialisé : {len(self.sequences)} arcs valides, {self.total_sequences()} séquences d'entraînement au total.")

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
        
    def total_sequences(self):
        return self.__len__()

    def __getitem__(self, idx):
        arc_index = 0
        while idx >= self.cumulative_lengths[arc_index]: arc_index += 1
        
        local_start_idx = idx if arc_index == 0 else idx - self.cumulative_lengths[arc_index - 1]
            
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        
        images = [Image.open(p).convert("RGB") for p in sequence_paths]
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
            
        return images

def get_dataloader(config, rank, world_size):
    """Crée un DataLoader avec un DistributedSampler pour DDP."""
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = AnimeFrameDataset(
        root_dir=config.DATASET_PATH, 
        sequence_length=config.TRAINING_SEQUENCE_LENGTH,
        transform=transform,
        is_master=(rank == 0) # Seul le processus maître affiche les logs de scan
    )
    
    # Le Sampler distribué s'assure que chaque GPU voit une partie unique du dataset.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True # Le shuffle est géré ici
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, # shuffle=False car le sampler s'en occupe
        num_workers=config.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True,
        sampler=sampler
    )
    return dataloader, sampler, dataset.total_sequences()

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
        self.z_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
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
        self.trajectory_generator = ConditionalLatentTrajectoryGenerator(
            config.GRU_HIDDEN_DIM, 
            config.LATENT_DIM
        )
        self.discriminator = PatchDiscriminator(config.IMG_CHANNELS)

# --- 4. Boucle d'Entraînement (MODIFIÉE pour DDP) ---

def main_worker(rank, world_size, config):
    """
    Fonction principale d'entraînement exécutée par chaque processus GPU.
    `rank` est l'identifiant du GPU (0, 1, ...).
    `world_size` est le nombre total de GPU.
    """
    print(f"Lancement du worker sur le rank {rank}.")
    setup_ddp(rank, world_size)
    
    # Seul le processus maître (rank 0) crée les dossiers
    if rank == 0:
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    dataloader, sampler, total_sequences = get_dataloader(config, rank, world_size)
    
    # Assigne le modèle au GPU spécifique à ce processus
    model = MTLD(config).to(rank)
    # Enveloppe le modèle avec DDP
    model = DDP(model, device_ids=[rank])
    
    # Accès aux sous-modules via `model.module`
    g_params = list(model.module.encoder.parameters()) + list(model.module.decoder.parameters()) + list(model.module.trajectory_generator.parameters())
    d_params = list(model.module.discriminator.parameters())
    opt_g = optim.Adam(g_params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(d_params, lr=config.LEARNING_RATE_D, betas=(config.BETA1, config.BETA2))
    
    loss_l1 = nn.L1Loss().to(rank)
    loss_mse = nn.MSELoss().to(rank)
    loss_lpips_vgg = lpips.LPIPS(net='vgg').to(rank)
    
    start_epoch = 0
    
    if rank == 0:
        print("Début de l'entraînement distribué du modèle...")

    for epoch in range(start_epoch, config.EPOCHS):
        # Le sampler doit connaître l'époque actuelle pour un shuffle correct
        sampler.set_epoch(epoch)
        
        # tqdm n'est affiché que par le processus maître
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", disable=(rank != 0))
        
        for i, real_seq_imgs in enumerate(pbar):
            # Les données sont déjà sur le bon GPU grâce à pin_memory et au worker
            real_seq_imgs = real_seq_imgs.to(rank, non_blocking=True)
            b, s, c, h, w = real_seq_imgs.shape
            
            priming_img = real_seq_imgs[:, 0, :, :, :]
            future_imgs = real_seq_imgs[:, 1:, :, :, :]
            
            opt_g.zero_grad()
            
            # Utilisation de model.module pour accéder aux méthodes de la classe MTLD
            with torch.no_grad():
                z_start_true = model.module.encoder(priming_img)
                z_future_true = model.module.encoder(future_imgs.reshape(-1, c, h, w)).view(b, s - 1, -1)

            z_future_pred = model.module.trajectory_generator(z_start_true, max_len=s - 1)
            
            z_pred_full_seq = torch.cat([z_start_true.unsqueeze(1), z_future_pred], dim=1)
            fake_seq_imgs = model.module.decoder(z_pred_full_seq)

            fake_imgs_flat = fake_seq_imgs.view(b * s, c, h, w)
            real_imgs_flat = real_seq_imgs.view(b * s, c, h, w)
            
            loss_latent = loss_mse(z_future_pred, z_future_true)
            loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
            loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()

            fake_d_output, _ = model.module.discriminator(fake_imgs_flat)
            target_real = torch.ones_like(fake_d_output)
            loss_adv = loss_mse(fake_d_output, target_real)
            _, real_features = model.module.discriminator(real_imgs_flat.detach(), extract_features=True)
            _, fake_features = model.module.discriminator(fake_imgs_flat, extract_features=True)
            loss_fm = sum(loss_l1(fake_f, real_f.detach()) for real_f, fake_f in zip(real_features, fake_features))
            
            loss_g = (config.LAMBDA_REC_L1 * loss_rec_l1 +
                      config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                      config.LAMBDA_LATENT * loss_latent +
                      config.LAMBDA_ADV * loss_adv +
                      config.LAMBDA_FM * loss_fm)
            loss_g.backward() # DDP s'occupe de la synchronisation des gradients
            opt_g.step()
            
            opt_d.zero_grad()
            real_d_output, _ = model.module.discriminator(real_imgs_flat)
            loss_d_real = loss_mse(real_d_output, target_real)
            fake_d_output, _ = model.module.discriminator(fake_imgs_flat.detach())
            target_fake = torch.zeros_like(fake_d_output)
            loss_d_fake = loss_mse(fake_d_output, target_fake)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()
            
            if rank == 0:
                pbar.set_postfix({
                    "L_Pred": f"{loss_latent.item():.3f}", "L_Rec": f"{loss_rec_l1.item():.3f}",
                    "L_Adv": f"{loss_adv.item():.3f}", "L_D": f"{loss_d.item():.3f}",
                })
        
        # Seul le processus maître sauvegarde le modèle et les images
        if rank == 0 and (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            save_path = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v1.6_epoch_{epoch+1}.pth")
            # On sauvegarde le state_dict du modèle original, pas du wrapper DDP
            torch.save(model.module.state_dict(), save_path)
            print(f"\nModèle sauvegardé : {save_path}")
            
            with torch.no_grad():
                model.eval()
                comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_seq_imgs[0].unsqueeze(0)], dim=0)
                comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                utils.save_image(comparison_flat, grid_path, nrow=s, normalize=True)
                print(f"Image de comparaison sauvegardée : {grid_path}")
                model.train()
                
    if rank == 0:
        print("Entraînement terminé.")
    
    cleanup_ddp()


# --- 5. Génération de Séquence (inchangée, à exécuter en mode mono-GPU) ---

def generate_sequence(model_path, priming_image_path, num_frames_to_generate, config):
    # La génération n'a pas besoin d'être distribuée, elle s'exécute sur un seul GPU.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("--- Démarrage de la Génération de Séquence Conditionnelle ---")
    if not os.path.exists(model_path):
        print(f"ERREUR : Fichier modèle non trouvé à {model_path}"); return
    if not os.path.exists(priming_image_path):
        print(f"ERREUR : Image d'amorce non trouvée à {priming_image_path}"); return

    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_conditional")
    os.makedirs(output_dir_frames, exist_ok=True)
    
    model = MTLD(config).to(DEVICE)
    # Le modèle a été sauvegardé sans le wrapper DDP, donc on le charge directement.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Modèle chargé avec succès.")

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(DEVICE)

    print(f"Génération de {num_frames_to_generate} images à partir de l'image d'amorce.")

    with torch.no_grad():
        z_start = model.encoder(priming_tensor)
        z_future = model.trajectory_generator(z_start, max_len=num_frames_to_generate - 1)
        z_full_seq = torch.cat([z_start.unsqueeze(1), z_future], dim=1)
        generated_imgs_seq = model.decoder(z_full_seq).squeeze(0)

    video_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (config.IMG_SIZE, config.IMG_SIZE))
    pil_images_for_gif = []

    for i, img_tensor in enumerate(tqdm(generated_imgs_seq, desc="Sauvegarde des images")):
        img_np = np.clip((img_tensor.permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        img_pil.save(os.path.join(output_dir_frames, f"frame_{i:04d}.png"))
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        pil_images_for_gif.append(img_pil)

    video_writer.release()
    print(f"Vidéo MP4 sauvegardée : {video_path}")
    gif_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.gif")
    pil_images_for_gif[0].save(
        gif_path, save_all=True, append_images=pil_images_for_gif[1:],
        duration=int(1000 / 24), loop=0
    )
    print(f"GIF sauvegardé : {gif_path}")


if __name__ == '__main__':
    config = Config()
    
    # --- MODE 1: ENTRAÎNEMENT DU MODÈLE (DISTRIBUÉ) ---
    #print("\n--- MODE ENTRAÎNEMENT (v1.6 Distribué Multi-GPU) ---")
    
    #world_size = torch.cuda.device_count()
    #if world_size > 1:
    #    print(f"Détection de {world_size} GPUs. Lancement de l'entraînement DDP.")
    ##    # Lance la fonction main_worker sur 'world_size' processus
    #    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)
    #elif world_size == 1:
    #    print("Un seul GPU détecté. Lancement en mode mono-processus.")
    #    # Pour un seul GPU, on peut appeler directement le worker sans mp.spawn
    #    main_worker(0, 1, config)
    #else:
    #    print("ERREUR: Aucun GPU détecté. L'entraînement sur CPU n'est pas supporté pour ce script.")

    # --- MODE 2: GÉNÉRATION CONDITIONNELLE (NON-DISTRIBUÉE) ---
    # Décommentez cette section pour exécuter la génération après l'entraînement.
    # La génération se fait sur un seul GPU.
    
    print("\n--- MODE GÉNÉRATION (v1.6) ---")
    config_gen = Config()
    model_file = os.path.join("mtld_epoch_20.pth")
    
    try:
        dummy_dataset = AnimeFrameDataset(config_gen.DATASET_PATH, 1, is_master=True)
        if dummy_dataset.sequences and len(dummy_dataset.sequences[0]) > 50:
            priming_image = dummy_dataset.sequences[0][50] 
            print(f"Utilisation de l'image d'amorce : {priming_image}")
        else:
            raise IndexError # Force le bloc except
    except (IndexError, FileNotFoundError, ValueError) as e:
        print(f"ATTENTION: Impossible de trouver une image d'amorce par défaut ({e}). Spécifiez un chemin valide.")
        priming_image = "animes_dataset/imouto_umaru_chan_opening/frame_0014.png" # REMPLACEZ CECI

    if os.path.exists(priming_image):
        generate_sequence(
            model_path=model_file, 
            priming_image_path=priming_image, 
            num_frames_to_generate=100,
            config=config_gen
        )