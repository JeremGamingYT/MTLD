# -*- coding: utf-8 -*-
"""
HALT: Hierarchical Autoregressive Latent Transformer
Version: 1.0
Date: 02 novembre 2025
Description:
Un modèle de génération de séquences d'images (anime) basé sur une architecture hiérarchique.
Il combine trois composants principaux :
1. Un encodeur discret (VQ-VAE) pour transformer les images en grilles de tokens sémantiques.
2. Un Transformer de trajectoire latente pour prédire l'évolution temporelle de ces grilles de tokens.
3. Un Transformer de pixels, conditionnel et autorégressif, pour synthétiser des images de haute
   fidélité à partir des grilles de tokens prédites.

Ce modèle vise une cohérence temporelle parfaite et une qualité d'image maximale en séparant
la planification sémantique (trajectoire) du rendu détaillé (génération de pixels).
"""

# --- 1. Importations et Configuration ---
import os
import glob
import re
import time
import math
import warnings
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import lpips

warnings.filterwarnings("ignore", category=UserWarning)

class Config:
    # --- Chemins et Données ---
    # !! MODIFIER ICI le chemin vers votre dataset d'images !!
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset"
    # Chemin où les tokens discrets seront sauvegardés après la Phase B
    TOKEN_DATASET_PATH = "./animes_dataset_tokens/"
    
    # --- Paramètres de l'Image ---
    IMG_SIZE = 128  # Réduit à 128 pour la faisabilité. Augmenter à 256 pour une qualité maximale.
    IMG_CHANNELS = 3

    # --- Paramètres du VQ-VAE (Phase A) ---
    VQ_LATENT_DIM = 256         # Dimension des vecteurs dans le codebook
    VQ_NUM_CODEBOOK_VECTORS = 8192 # Taille du dictionnaire de tokens visuels
    VQ_COMMITMENT_COST = 0.25   # Poids de la perte de "commitment"
    VQ_DECODER_CHANNELS = 64
    VQ_DOWNSAMPLE_LEVELS = 3    # 128 -> 64 -> 32 -> 16. Grid de 16x16
    
    # --- Paramètres des Transformers (Phase C) ---
    T_EMBED_DIM = 512           # Dimension d'embedding pour les Transformers
    T_NUM_HEADS = 8             # Nombre de têtes d'attention
    T_NUM_LAYERS = 8            # Nombre de blocs Transformer
    T_DROPOUT = 0.1

    # --- Paramètres d'Entraînement ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEQUENCE_LENGTH = 16        # Nombre d'images par séquence
    
    # Phase A: Entraînement du VQ-VAE
    VQ_EPOCHS = 50
    VQ_BATCH_SIZE = 32
    VQ_LEARNING_RATE = 3e-4

    # Phase C: Entraînement des Transformers HALT
    HALT_EPOCHS = 100
    HALT_BATCH_SIZE = 4 # Doit être plus petit à cause de la mémoire
    HALT_LEARNING_RATE = 1e-4

    # --- Sorties et Sauvegardes ---
    MODEL_SAVE_PATH = "./models_halt_v1/"
    OUTPUT_SAVE_PATH = "./outputs_halt_v1/"
    SAVE_EPOCH_INTERVAL = 5
    
    # --- Inférence ---
    GENERATION_PRIMING_IMAGE = "frame_0001.png" # Remplacer par un chemin valide
    GENERATION_NUM_FRAMES = 64


# --- 2. Préparation des Données ---

# Dataset pour les images brutes (Phase A)
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            paths = sorted(glob.glob(os.path.join(arc_path, "*.png")), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            self.image_paths.extend(paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Dataset pour les séquences de tokens (Phase C)
class TokenSequenceDataset(Dataset):
    def __init__(self, token_dir, sequence_length, img_transform):
        self.token_dir = token_dir
        self.sequence_length = sequence_length
        self.img_transform = img_transform
        self.sequences = []
        
        arc_dirs = sorted([d for d in os.listdir(token_dir) if os.path.isdir(os.path.join(token_dir, d))])
        for arc_dir in arc_dirs:
            arc_path = os.path.join(token_dir, arc_dir)
            token_files = sorted(
                glob.glob(os.path.join(arc_path, "*.pt")),
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
            )
            
            if len(token_files) >= self.sequence_length:
                # Créer des séquences qui se chevauchent
                for i in range(len(token_files) - self.sequence_length + 1):
                    seq_files = token_files[i : i + self.sequence_length]
                    self.sequences.append(seq_files)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        token_paths = self.sequences[idx]
        
        token_tensors = [torch.load(p, map_location='cpu') for p in token_paths]
        
        # Charger l'image correspondante pour le PixelTransformer
        # On prend la première image de la séquence comme exemple
        first_token_path = token_paths[0]
        img_path = first_token_path.replace(self.token_dir, Config.DATASET_PATH).replace('.pt', '.png')
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image)
        
        # Discrétiser les pixels de l'image en entiers de 0 à 255
        pixel_values = (image_tensor * 255).long()

        return torch.stack(token_tensors), pixel_values


# --- 3. Composant 1: Vector Quantized Encoder (VQ-VAE) ---

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: (B, D, H, W) -> (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Distances
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_flat, self.embedding.weight.t()))
        
        # Trouver les indices les plus proches
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Quantifier
        quantized_flat = self.embedding(encoding_indices).view(z_flat.shape)
        
        # Calcul des pertes (avec Straight-Through Estimator)
        e_latent_loss = F.mse_loss(quantized_flat.detach(), z_flat)
        q_latent_loss = F.mse_loss(quantized_flat, z_flat.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Copier les gradients
        quantized = z + (quantized_flat.view_as(z) - z).detach()
        
        # Reshape les indices
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])
        
        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        encoder_layers = []
        in_c = config.IMG_CHANNELS
        # Downsampling
        for i in range(config.VQ_DOWNSAMPLE_LEVELS):
            out_c = config.VQ_DECODER_CHANNELS * (2**i)
            encoder_layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_c = out_c
        
        # Add a conv to get to VQ_LATENT_DIM
        encoder_layers.append(nn.Conv2d(in_c, config.VQ_LATENT_DIM, kernel_size=3, padding=1))
        
        # Add residual blocks
        for _ in range(2):
            encoder_layers.append(ResidualBlock(config.VQ_LATENT_DIM, config.VQ_LATENT_DIM))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(config.VQ_NUM_CODEBOOK_VECTORS, config.VQ_LATENT_DIM, config.VQ_COMMITMENT_COST)
        
        # Decoder
        decoder_layers = []
        
        # Start with a conv layer to prepare for residual blocks
        decoder_layers.append(nn.Conv2d(config.VQ_LATENT_DIM, in_c, kernel_size=3, padding=1))
        
        # Add residual blocks
        for _ in range(2):
            decoder_layers.append(ResidualBlock(in_c, in_c))
            
        # Upsampling
        for i in reversed(range(config.VQ_DOWNSAMPLE_LEVELS)):
            out_c = config.VQ_DECODER_CHANNELS * (2**i)
            decoder_layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())
            in_c = out_c
            
        # Final conv to get back to image channels
        decoder_layers.append(nn.Conv2d(in_c, config.IMG_CHANNELS, kernel_size=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices
        
    def encode(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.quantizer(z_e)
        return indices

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        super().__init__()
        
        layers = [nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --- 4. Composant 2: Latent Trajectory Transformer ---

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len_t, max_len_h, max_len_w):
        super().__init__()
        self.d_model = d_model
        pe_t = torch.zeros(max_len_t, d_model)
        pe_h = torch.zeros(max_len_h, d_model)
        pe_w = torch.zeros(max_len_w, d_model)

        position_t = torch.arange(0, max_len_t, dtype=torch.float).unsqueeze(1)
        position_h = torch.arange(0, max_len_h, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_len_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe_t[:, 0::2] = torch.sin(position_t * div_term)
        pe_t[:, 1::2] = torch.cos(position_t * div_term)
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)

        self.register_buffer('pe_t', pe_t)
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)

    def forward(self, t, h, w):
        # t, h, w are tensors of indices
        return self.pe_t[t] + self.pe_h[h] + self.pe_w[w]

class LatentTrajectoryTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        grid_size = config.IMG_SIZE // (2**config.VQ_DOWNSAMPLE_LEVELS)
        
        self.token_embedding = nn.Embedding(config.VQ_NUM_CODEBOOK_VECTORS, config.T_EMBED_DIM)
        self.pos_embedding = PositionalEncoding3D(config.T_EMBED_DIM, config.SEQUENCE_LENGTH, grid_size, grid_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.T_EMBED_DIM, nhead=config.T_NUM_HEADS, 
            dim_feedforward=4*config.T_EMBED_DIM, dropout=config.T_DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.T_NUM_LAYERS)
        
        self.to_logits = nn.Linear(config.T_EMBED_DIM, config.VQ_NUM_CODEBOOK_VECTORS)
        self.grid_size = grid_size

    def forward(self, z_indices):
        # z_indices: (B, S, H, W)
        B, S, H, W = z_indices.shape
        
        # Flatten and create positions
        z_flat = z_indices.view(B, -1) # (B, S*H*W)
        
        t_pos = torch.arange(S, device=z_indices.device).repeat_interleave(H*W).unsqueeze(0).repeat(B, 1)
        h_pos = torch.arange(H, device=z_indices.device).repeat_interleave(W).repeat(S).unsqueeze(0).repeat(B, 1)
        w_pos = torch.arange(W, device=z_indices.device).repeat(S*H).unsqueeze(0).repeat(B, 1)

        # Shift input for autoregressive prediction
        z_input = z_flat[:, :-1]
        pos_input = self.pos_embedding(t_pos[:, :-1], h_pos[:, :-1], w_pos[:, :-1])
        
        x = self.token_embedding(z_input) + pos_input
        
        # Causal mask
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer forward
        output = self.transformer(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)
        
        logits = self.to_logits(output) # (B, S*H*W - 1, VQ_NUM_VECTORS)
        
        return logits, z_flat[:, 1:] # Return logits and target


# --- 5. Composant 3: Pixel Transformer Decoder ---

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(height * width, d_model)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, n):
        return self.pe[:n, :]

class PixelTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        grid_size = config.IMG_SIZE // (2**config.VQ_DOWNSAMPLE_LEVELS)
        
        # Pixel embeddings
        self.pixel_embedding = nn.Embedding(256 * config.IMG_CHANNELS, config.T_EMBED_DIM)
        
        # Positional encoding for pixels
        self.pixel_pos_embedding = PositionalEncoding2D(config.T_EMBED_DIM, config.IMG_SIZE, config.IMG_SIZE)

        # Latent grid conditioning
        self.latent_embedding = nn.Embedding(config.VQ_NUM_CODEBOOK_VECTORS, config.T_EMBED_DIM)
        self.latent_pos_embedding = PositionalEncoding2D(config.T_EMBED_DIM, grid_size, grid_size)
        
        # Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.T_EMBED_DIM, nhead=config.T_NUM_HEADS,
            dim_feedforward=4 * config.T_EMBED_DIM, dropout=config.T_DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.T_NUM_LAYERS)
        
        # Output head
        self.to_logits = nn.Linear(config.T_EMBED_DIM, 256 * config.IMG_CHANNELS)

    def forward(self, z_indices, pixels):
        # z_indices: (B, H, W) - a single latent grid
        # pixels: (B, C, H_img, W_img) - the target image
        B, C, H_img, W_img = pixels.shape
        
        # Prepare latent conditioning (memory)
        z_flat = z_indices.view(B, -1)
        latent_embed = self.latent_embedding(z_flat)
        latent_pos = self.latent_pos_embedding(z_flat.shape[1]).unsqueeze(0)
        memory = latent_embed + latent_pos
        
        # Prepare pixel sequence (target)
        # Combine channels into a single dimension for embedding
        pixel_flat = pixels.permute(0, 2, 3, 1).contiguous().view(B, H_img * W_img, C)
        # Create a single integer representation for each pixel's RGB values
        pixel_indices = pixel_flat[..., 0] * (256**2) + pixel_flat[..., 1] * 256 + pixel_flat[..., 2]
        # This is a simplification. A better approach would be to embed each channel separately.
        # Let's do that instead for better quality.
        
        # Re-approach: Embed each channel
        pixel_flat_shifted = F.pad(pixel_flat, (0, 0, 1, 0), value=0)[:, :-1, :] # Shift for autoregression
        
        r_embed = self.pixel_embedding(pixel_flat_shifted[..., 0] + 0*256)
        g_embed = self.pixel_embedding(pixel_flat_shifted[..., 1] + 1*256)
        b_embed = self.pixel_embedding(pixel_flat_shifted[..., 2] + 2*256)
        
        pixel_embed = r_embed + g_embed + b_embed
        
        pixel_pos = self.pixel_pos_embedding(H_img * W_img).unsqueeze(0)
        tgt = pixel_embed + pixel_pos
        
        # Causal mask for pixels
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward
        output = self.transformer(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        
        # Get logits
        logits = self.to_logits(output) # (B, H*W, 256 * 3)
        
        # Separate logits for each channel
        r_logits, g_logits, b_logits = torch.chunk(logits, 3, dim=-1)
        
        # Targets
        r_target, g_target, b_target = pixel_flat.unbind(-1)
        
        return (r_logits, g_logits, b_logits), (r_target, g_target, b_target)


# --- 6. Orchestration et Boucles d'Entraînement ---

def train_vqvae(config):
    print("--- Phase A: Entraînement du VQ-GAN (avec démarrage progressif) ---")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)

    # --- Modèles ---
    model = VQVAE(config).to(config.DEVICE)
    discriminator = PatchDiscriminator(input_channels=config.IMG_CHANNELS).to(config.DEVICE)
    perceptual_loss = lpips.LPIPS(net='vgg').to(config.DEVICE)

    # --- Optimiseurs ---
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.VQ_LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.VQ_LEARNING_RATE, betas=(0.5, 0.999))

    # --- Données ---
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageDataset(root_dir=config.DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.VQ_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    # --- Paramètres de l'entraînement GAN ---
    gan_weight = 0.1
    disc_start_epoch = 5 # Démarrer l'entraînement du discriminateur après cette époque

    for epoch in range(config.VQ_EPOCHS):
        pbar = tqdm(dataloader, desc=f"VQ-GAN Epoch {epoch+1}/{config.VQ_EPOCHS}")
        total_recon_loss, total_perceptual_loss, total_g_loss, total_d_loss = 0, 0, 0, 0

        # Déterminer si le discriminateur doit être actif
        is_disc_active = epoch >= disc_start_epoch
        current_gan_weight = gan_weight if is_disc_active else 0.0

        for images in pbar:
            images = images.to(config.DEVICE)

            with torch.cuda.amp.autocast():
                recon_images, vq_loss, _ = model(images)

            # --- Entraînement du Générateur (VQ-VAE) ---
            optimizer_g.zero_grad()
            with torch.cuda.amp.autocast():
                # Perte de reconstruction L1
                recon_loss_l1 = F.l1_loss(recon_images, images)
                
                # Perte perceptuelle
                perceptual_loss_val = perceptual_loss(recon_images, images).mean()

                # Perte adversariale (uniquement si active)
                if is_disc_active:
                    logits_fake = discriminator(recon_images)
                    g_loss = -torch.mean(logits_fake)
                else:
                    g_loss = torch.tensor(0.0, device=config.DEVICE)

                # Perte totale du générateur
                loss_g = recon_loss_l1 + perceptual_loss_val + vq_loss + current_gan_weight * g_loss
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # --- Entraînement du Discriminateur (uniquement si actif) ---
            d_loss = torch.tensor(0.0, device=config.DEVICE)
            if is_disc_active:
                optimizer_d.zero_grad()
                with torch.cuda.amp.autocast():
                    logits_real = discriminator(images)
                    logits_fake_detached = discriminator(recon_images.detach())

                    # Perte du discriminateur (Hinge Loss)
                    d_loss_real = torch.mean(F.relu(1. - logits_real))
                    d_loss_fake = torch.mean(F.relu(1. + logits_fake_detached))
                    d_loss = (d_loss_real + d_loss_fake) / 2

                scaler_d.scale(d_loss).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

            # --- Logging ---
            total_recon_loss += recon_loss_l1.item()
            total_perceptual_loss += perceptual_loss_val.item()
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            pbar.set_postfix({
                "Recon L1": f"{recon_loss_l1.item():.4f}",
                "Perceptual": f"{perceptual_loss_val.item():.4f}",
                "G Loss": f"{g_loss.item():.4f}",
                "D Loss": f"{d_loss.item():.4f}"
            })

        avg_recon = total_recon_loss / len(dataloader)
        avg_perceptual = total_perceptual_loss / len(dataloader)
        avg_g = total_g_loss / len(dataloader)
        avg_d = total_d_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Losses -> Recon: {avg_recon:.4f}, Perceptual: {avg_perceptual:.4f}, G: {avg_g:.4f}, D: {avg_d:.4f}")

        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"vqgan_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Modèle VQ-GAN sauvegardé : {model_path}")
            
            with torch.no_grad():
                comparison = torch.cat([images[:4], recon_images[:4]])
                grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"vqgan_recon_{epoch+1}.png")
                utils.save_image(comparison, grid_path, nrow=4, normalize=True, value_range=(-1, 1))

def convert_dataset_to_tokens(config):
    print("\n--- Phase B: Conversion du Dataset en Tokens ---")
    os.makedirs(config.TOKEN_DATASET_PATH, exist_ok=True)
    
    vqgan_path = os.path.join(config.MODEL_SAVE_PATH, f"vqgan_epoch_{config.VQ_EPOCHS}.pth")
    if not os.path.exists(vqgan_path):
        print(f"ERREUR: Modèle VQ-GAN entraîné non trouvé à {vqgan_path}. Lancez d'abord l'entraînement.")
        return
        
    model = VQVAE(config).to(config.DEVICE)
    model.load_state_dict(torch.load(vqgan_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageDataset(root_dir=config.DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.VQ_BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        idx = 0
        for images in tqdm(dataloader, desc="Conversion des images en tokens"):
            images = images.to(config.DEVICE)
            indices = model.encode(images).cpu() # (B, H, W)
            
            for i in range(indices.shape[0]):
                original_path = dataset.image_paths[idx]
                
                # Recréer la structure de dossiers
                relative_path = os.path.relpath(original_path, config.DATASET_PATH)
                token_save_path = os.path.join(config.TOKEN_DATASET_PATH, relative_path)
                token_save_path = os.path.splitext(token_save_path)[0] + ".pt"
                os.makedirs(os.path.dirname(token_save_path), exist_ok=True)
                
                torch.save(indices[i], token_save_path)
                idx += 1
    print("Conversion terminée.")

def train_halt(config):
    print("\n--- Phase C: Entraînement des Transformers HALT ---")
    if not os.path.exists(config.TOKEN_DATASET_PATH):
        print(f"ERREUR: Dossier de tokens non trouvé. Lancez d'abord la conversion.")
        return

    img_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = TokenSequenceDataset(config.TOKEN_DATASET_PATH, config.SEQUENCE_LENGTH, img_transform)
    dataloader = DataLoader(dataset, batch_size=config.HALT_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    traj_transformer = LatentTrajectoryTransformer(config).to(config.DEVICE)
    pixel_transformer = PixelTransformerDecoder(config).to(config.DEVICE)
    
    params = list(traj_transformer.parameters()) + list(pixel_transformer.parameters())
    optimizer = torch.optim.Adam(params, lr=config.HALT_LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.HALT_EPOCHS):
        pbar = tqdm(dataloader, desc=f"HALT Epoch {epoch+1}/{config.HALT_EPOCHS}")
        
        for token_seqs, pixel_values in pbar:
            token_seqs = token_seqs.to(config.DEVICE) # (B, S, H, W)
            pixel_values = pixel_values.to(config.DEVICE) # (B, C, H_img, W_img)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(config.DEVICE == "cuda")):
                # --- Perte du Trajectory Transformer ---
                traj_logits, traj_targets = traj_transformer(token_seqs)
                loss_traj = criterion(traj_logits.view(-1, config.VQ_NUM_CODEBOOK_VECTORS), traj_targets.view(-1))
                
                # --- Perte du Pixel Transformer ---
                # On entraîne sur la première image de la séquence pour l'exemple
                z_indices_cond = token_seqs[:, 0, :, :]
                (r_logits, g_logits, b_logits), (r_target, g_target, b_target) = pixel_transformer(z_indices_cond, pixel_values)
                
                loss_r = criterion(r_logits.view(-1, 256), r_target.view(-1))
                loss_g = criterion(g_logits.view(-1, 256), g_target.view(-1))
                loss_b = criterion(b_logits.view(-1, 256), b_target.view(-1))
                loss_pixel = (loss_r + loss_g + loss_b) / 3.0
                
                # --- Perte totale ---
                loss = loss_traj + loss_pixel

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({
                "Loss Traj": f"{loss_traj.item():.4f}",
                "Loss Pixel": f"{loss_pixel.item():.4f}"
            })

        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            torch.save(traj_transformer.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f"traj_transformer_epoch_{epoch+1}.pth"))
            torch.save(pixel_transformer.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f"pixel_transformer_epoch_{epoch+1}.pth"))
            print(f"Modèles HALT sauvegardés pour l'époque {epoch+1}")


# --- 7. Inférence et Génération ---

@torch.no_grad()
def generate_sequence(config):
    print("\n--- Phase D: Génération de Séquence avec HALT ---")
    # Charger les modèles
    vqvae = VQVAE(config).to(config.DEVICE)
    traj_transformer = LatentTrajectoryTransformer(config).to(config.DEVICE)
    pixel_transformer = PixelTransformerDecoder(config).to(config.DEVICE)

    vqvae.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, f"vqgan_epoch_{config.VQ_EPOCHS}.pth")))
    traj_transformer.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, f"traj_transformer_epoch_{config.HALT_EPOCHS}.pth")))
    pixel_transformer.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, f"pixel_transformer_epoch_{config.HALT_EPOCHS}.pth")))

    vqvae.eval(); traj_transformer.eval(); pixel_transformer.eval()
    
    # Charger l'image d'amorce
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    priming_img = Image.open(config.GENERATION_PRIMING_IMAGE).convert("RGB")
    priming_tensor = transform(priming_img).unsqueeze(0).to(config.DEVICE)

    # --- 1. Générer la trajectoire latente ---
    print("Étape 1/2 : Génération de la trajectoire latente...")
    z_grid_0 = vqvae.encode(priming_tensor) # (1, H, W)
    
    grid_h, grid_w = z_grid_0.shape[-2:]
    num_tokens_per_frame = grid_h * grid_w

    latent_tokens = z_grid_0.view(1, -1)

    total_tokens_to_generate = config.GENERATION_NUM_FRAMES * num_tokens_per_frame
    pbar_traj = tqdm(range(latent_tokens.shape[1], total_tokens_to_generate), desc="Génération de la trajectoire")

    for i in pbar_traj:
        context_tokens = latent_tokens
        
        # Limiter le contexte pour éviter une charge de calcul trop importante
        max_context_tokens = (config.SEQUENCE_LENGTH - 1) * num_tokens_per_frame
        if context_tokens.shape[1] > max_context_tokens:
            context_tokens = context_tokens[:, -max_context_tokens:]

        # Le modèle attend des trames complètes, nous devons donc paver la séquence
        num_tokens_in_context = context_tokens.shape[1]
        frames_to_feed = math.ceil(num_tokens_in_context / num_tokens_per_frame)
        
        padded_len = frames_to_feed * num_tokens_per_frame
        padded_tokens = F.pad(context_tokens, (0, padded_len - num_tokens_in_context), value=0)
        
        input_seq_4d = padded_tokens.view(1, frames_to_feed, grid_h, grid_w)
        
        # Prédire le prochain token
        logits, _ = traj_transformer(input_seq_4d)
        
        # Obtenir le logit pour le token que nous voulons prédire
        logit_index = num_tokens_in_context - 1
        next_token_logits = logits[:, logit_index, :]
        
        # Échantillonner
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        
        # Ajouter le token généré à notre séquence
        latent_tokens = torch.cat([latent_tokens, next_token], dim=1)

    latent_trajectory = latent_tokens.view(config.GENERATION_NUM_FRAMES, grid_h, grid_w)
    print("Trajectoire latente générée.")

    # --- 2. Rendu des images pixel par pixel ---
    print("Étape 2/2 : Rendu des images à partir de la trajectoire...")
    generated_frames = []
    
    for t in range(config.GENERATION_NUM_FRAMES):
        z_grid_t = latent_trajectory[t].unsqueeze(0) # (1, H, W)
        print(f"  Rendu de l'image {t+1}/{config.GENERATION_NUM_FRAMES}...")
        
        # Préparer la mémoire pour le PixelTransformer
        z_flat = z_grid_t.view(1, -1)
        latent_embed = pixel_transformer.latent_embedding(z_flat)
        latent_pos = pixel_transformer.latent_pos_embedding(z_flat.shape[1]).unsqueeze(0)
        memory = latent_embed + latent_pos
        
        pixels = torch.zeros(1, config.IMG_SIZE, config.IMG_SIZE, 3, dtype=torch.long).to(config.DEVICE)
        
        pbar_pixels = tqdm(range(config.IMG_SIZE * config.IMG_SIZE), desc="  Génération des pixels")
        for i in pbar_pixels:
            row, col = i // config.IMG_SIZE, i % config.IMG_SIZE
            
            # Préparer l'input pour le transformer
            pixel_flat_so_far = pixels.view(1, -1, 3)
            
            r_embed = pixel_transformer.pixel_embedding(pixel_flat_so_far[..., 0] + 0*256)
            g_embed = pixel_transformer.pixel_embedding(pixel_flat_so_far[..., 1] + 1*256)
            b_embed = pixel_transformer.pixel_embedding(pixel_flat_so_far[..., 2] + 2*256)
            
            pixel_embed = r_embed + g_embed + b_embed
            pixel_pos = pixel_transformer.pixel_pos_embedding(pixel_embed.shape[1]).unsqueeze(0)
            tgt = pixel_embed + pixel_pos
            
            # Causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Prédire le prochain pixel
            output = pixel_transformer.transformer(tgt, memory, tgt_mask=tgt_mask)
            logits = pixel_transformer.to_logits(output[:, i, :]) # Prendre le logit pour le pixel courant
            
            r_logits, g_logits, b_logits = torch.chunk(logits, 3, dim=-1)
            
            # Échantillonner les valeurs RGB
            r_val = torch.multinomial(F.softmax(r_logits, dim=-1), 1).squeeze(-1)
            g_val = torch.multinomial(F.softmax(g_logits, dim=-1), 1).squeeze(-1)
            b_val = torch.multinomial(F.softmax(b_logits, dim=-1), 1).squeeze(-1)
            
            pixels[0, row, col, 0] = r_val
            pixels[0, row, col, 1] = g_val
            pixels[0, row, col, 2] = b_val
            
        generated_frames.append(pixels.cpu().numpy().astype(np.uint8))

    # --- 3. Sauvegarder la vidéo ---
    video_path = os.path.join(config.OUTPUT_SAVE_PATH, "generated_halt_sequence.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (config.IMG_SIZE, config.IMG_SIZE))
    for frame in generated_frames:
        video_writer.write(cv2.cvtColor(frame.squeeze(0), cv2.COLOR_RGB2BGR))
    video_writer.release()
    print(f"Séquence générée et sauvegardée en vidéo : {video_path}")


# --- 8. Point d'Entrée Principal ---

if __name__ == '__main__':
    config = Config()
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    
    print("="*80)
    print("Modèle HALT: Hierarchical Autoregressive Latent Transformer")
    print(f"Utilisation du device : {config.DEVICE}")
    print("="*80)
    
    # --- Instructions d'utilisation ---
    # Décommentez UNE SEULE des actions suivantes à la fois.
    # L'ordre correct est : train_vqvae -> convert_dataset_to_tokens -> train_halt -> generate_sequence
    
    # ACTION 1: Entraîner le VQ-VAE pour apprendre le dictionnaire de tokens visuels.
    train_vqvae(config)
    
    # ACTION 2: Convertir toutes les images du dataset en fichiers de tokens.
    # convert_dataset_to_tokens(config)
    
    # ACTION 3: Entraîner les deux transformers (trajectoire et pixels) sur les tokens.
    # train_halt(config)
    
    # ACTION 4: Générer une nouvelle séquence d'animation à partir d'une image.
    # Assurez-vous que les modèles sont complètement entraînés avant de lancer ceci.
    # La génération est TRÈS LENTE à cause de la nature autorégressive pixel par pixel.
    # generate_sequence(config)
    
    #print("\nScript terminé. Modifiez la section `if __name__ == '__main__':` pour exécuter une action.")