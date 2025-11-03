# -*- coding: utf-8 -*-
# ==============================================================================
#                            IMPORTATIONS GLOBALES
# ==============================================================================
import os
import shutil
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import imageio
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModel
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
#                       CLASSE DE CONFIGURATION CENTRALE
# ==============================================================================
class Config:
    """Configuration centralisée pour le modèle, l'entraînement et les données."""
    # --- Chemins et Données ---
    # !! MODIFIEZ CETTE LIGNE pour pointer vers votre dataset !!
    DATASET_PATH = "/kaggle/input/anima-s-dataset/animes_dataset" 
    OUTPUT_PATH = "./clemo_diff_outputs/"
    MODEL_NAME = "CLeMo-Diff_v1.0"
    
    # --- Paramètres du Dataset ---
    IMG_SIZE = 256
    SEQUENCE_LENGTH = 16 # Nombre d'images par séquence vidéo

    # --- Hyperparamètres du Modèle ---
    LATENT_APPEARANCE_DIM = 768 # Doit correspondre à la sortie de CLIP ViT-L/14
    LATENT_MOTION_DIM = 64     # Dimension pour chaque vecteur de mouvement
    
    # --- Hyperparamètres d'Entraînement ---
    EPOCHS = 10 # Réduit pour une exécution de démonstration rapide
    BATCH_SIZE = 1 # Les modèles de diffusion sont gourmands en VRAM
    LEARNING_RATE = 1e-5
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_WEIGHT_DECAY = 1e-6
    ADAM_EPSILON = 1e-08
    
    # --- Poids des Fonctions de Coût ---
    MOTION_VAE_KL_WEIGHT = 1e-6

    # --- Configuration de la Diffusion ---
    NUM_TRAIN_TIMESTEPS = 1000
    
    # --- Divers ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    SAVE_EPOCH_INTERVAL = 5
    
    # --- Modèles Pré-entraînés (de Hugging Face) ---
    VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
    UNET_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


# ==============================================================================
#                      GÉNÉRATION D'UN DATASET FACTICE
# ==============================================================================
def create_dummy_dataset(base_path, num_arcs=5, frames_per_arc=30, size=256):
    """Génère un dataset vidéo factice pour les tests."""
    if os.path.exists(base_path):
        print(f"Le dataset factice '{base_path}' existe déjà.")
        return
    print(f"Création d'un dataset factice à '{base_path}'...")
    for i in range(num_arcs):
        arc_path = os.path.join(base_path, f"Arc_{i+1}")
        os.makedirs(arc_path, exist_ok=True)
        start_color = np.random.randint(0, 256, 3, dtype=np.uint8)
        for j in range(frames_per_arc):
            img = np.zeros((size, size, 3), dtype=np.uint8)
            frame_color = np.clip(start_color + np.array([i*2, j, -j//2]), 0, 255).astype(np.uint8)
            img[:, :] = frame_color
            x_pos = int((size / 2) + np.sin(j / frames_per_arc * 2 * np.pi) * (size / 4))
            y_pos = int((size / 2) + np.cos(j / frames_per_arc * 2 * np.pi) * (size / 4))
            img[y_pos-20:y_pos+20, x_pos-20:x_pos+20] = [255, 255, 255]
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(arc_path, f"{j:05d}.png"))
    print("Dataset factice créé.")


# ==============================================================================
#                         PIPELINE DE PRÉPARATION DES DONNÉES
# ==============================================================================
class AnimeVideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length, img_size):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.video_paths = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.sequences = self._create_sequences()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _create_sequences(self):
        sequences = []
        for video_path in self.video_paths:
            frames = sorted(glob.glob(os.path.join(video_path, "*.png")))
            if len(frames) >= self.sequence_length:
                for i in range(len(frames) - self.sequence_length + 1):
                    sequences.append(frames[i : i + self.sequence_length])
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        reference_image_path = frame_paths[0]
        try:
            reference_image = Image.open(reference_image_path).convert("RGB")
            video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
            reference_tensor = self.transform(reference_image)
            video_tensor = torch.stack([self.transform(img) for img in video_frames]).permute(1, 0, 2, 3)
            return {"reference_image": reference_tensor, "video_sequence": video_tensor}
        except Exception as e:
            print(f"Erreur de chargement à l'index {idx} (path: {reference_image_path}): {e}")
            return self.__getitem__((idx + 1) % len(self))

def get_dataloader(config):
    dataset = AnimeVideoDataset(root_dir=config.DATASET_PATH, sequence_length=config.SEQUENCE_LENGTH, img_size=config.IMG_SIZE)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)


# ==============================================================================
#                        ARCHITECTURE DU MODÈLE CLeMo-Diff
# ==============================================================================
class AppearanceEncoder(nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        self.vision_tower = CLIPVisionModel.from_pretrained(model_id).to(device)
        self.vision_tower.requires_grad_(False)
        self.output_dim = self.vision_tower.config.hidden_size

    def forward(self, reference_image):
        return self.vision_tower(pixel_values=reference_image, return_dict=True).pooler_output

class MotionEncoder3D(nn.Module):
    def __init__(self, motion_dim, sequence_length, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, sequence_length, img_size, img_size)
            linear_input_size = self.model(dummy_input).shape[1]
        self.head = nn.Linear(linear_input_size, sequence_length * motion_dim)
        self.motion_dim = motion_dim
        self.sequence_length = sequence_length

    def forward(self, video_sequence):
        features = self.model(video_sequence)
        return self.head(features).view(-1, self.sequence_length, self.motion_dim)

class MotionPriorVAE(nn.Module):
    def __init__(self, motion_dim, sequence_length, hidden_dim=256):
        super().__init__()
        self.sequence_length, self.motion_dim = sequence_length, motion_dim
        flat_dim = motion_dim * sequence_length
        self.encoder = nn.Sequential(nn.Linear(flat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_logvar = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, flat_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, z_motion):
        h = self.encoder(z_motion.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z).view(-1, self.sequence_length, self.motion_dim)

    def forward(self, z_motion):
        mu, logvar = self.encode(z_motion)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)

class CLeMoDiff(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vae = AutoencoderKL.from_pretrained(config.VAE_MODEL_ID)
        self.unet = UNet2DConditionModel.from_pretrained(config.UNET_MODEL_ID, subfolder="unet")
        self.appearance_encoder = AppearanceEncoder(config.CLIP_MODEL_ID, config.DEVICE)
        self.motion_encoder = MotionEncoder3D(config.LATENT_MOTION_DIM, config.SEQUENCE_LENGTH, config.IMG_SIZE)
        self.motion_prior = MotionPriorVAE(config.LATENT_MOTION_DIM, config.SEQUENCE_LENGTH)
        self.vae.requires_grad_(False)
        self.appearance_encoder.requires_grad_(False)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.NUM_TRAIN_TIMESTEPS, beta_schedule="squaredcos_cap_v2")
        unet_cross_dim = self.unet.config.cross_attention_dim
        self.appearance_proj = nn.Linear(self.appearance_encoder.output_dim, unet_cross_dim)
        self.motion_proj = nn.Linear(config.LATENT_MOTION_DIM, unet_cross_dim)

    def forward(self, batch):
        ref_img, video_seq = batch["reference_image"], batch["video_sequence"]
        b, c, t, h, w = video_seq.shape
        video_seq_flat = video_seq.permute(0, 2, 1, 3, 4).flatten(0, 1)
        with torch.no_grad():
            latents = self.vae.encode(video_seq_flat).latent_dist.sample() * self.vae.config.scaling_factor
        latents = latents.unflatten(0, (b, t))
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (b,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        with torch.no_grad():
            z_app_raw = self.appearance_encoder(ref_img)
        z_app = self.appearance_proj(z_app_raw).unsqueeze(1)
        z_motion = self.motion_encoder(video_seq)
        z_motion_proj = self.motion_proj(z_motion)
        conditioning = torch.cat([z_app, z_motion_proj], dim=1)
        recon_z_motion, mu, logvar = self.motion_prior(z_motion.detach())
        loss_motion_recon = F.mse_loss(recon_z_motion, z_motion)
        loss_motion_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_motion_vae = loss_motion_recon + self.config.MOTION_VAE_KL_WEIGHT * loss_motion_kl
        noise_pred = torch.stack([self.unet(noisy_latents[:, i], timesteps, conditioning).sample for i in range(t)], dim=1)
        loss_diffusion = F.mse_loss(noise_pred, noise)
        return loss_diffusion, loss_motion_vae


# ==============================================================================
#                             BOUCLE D'ENTRAÎNEMENT
# ==============================================================================
def train_clemo_diff(config):
    print("Initialisation de l'entraînement...")
    dataloader = get_dataloader(config)
    model = CLeMoDiff(config).to(config.DEVICE)
    trainable_params = list(model.unet.parameters()) + list(model.motion_encoder.parameters()) + \
                       list(model.motion_prior.parameters()) + list(model.appearance_proj.parameters()) + \
                       list(model.motion_proj.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                            weight_decay=config.ADAM_WEIGHT_DECAY, eps=config.ADAM_EPSILON)
    
    for epoch in range(config.EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for batch in progress_bar:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            loss_diffusion, loss_motion_vae = model(batch)
            total_loss = loss_diffusion + loss_motion_vae
            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"L_diff": f"{loss_diffusion.item():.4f}", "L_vae": f"{loss_motion_vae.item():.4f}"})
        
        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            save_path = os.path.join(config.OUTPUT_PATH, 'models', f"{config.MODEL_NAME}_epoch_{epoch+1}.pth")
            torch.save({k: v.state_dict() for k, v in {
                'unet': model.unet, 'motion_encoder': model.motion_encoder, 'motion_prior': model.motion_prior,
                'appearance_proj': model.appearance_proj, 'motion_proj': model.motion_proj
            }.items()}, save_path)
            print(f"\nModèle sauvegardé à : {save_path}")
    print("Entraînement terminé.")
    return model


# ==============================================================================
#                         FONCTION DE GÉNÉRATION (INFÉRENCE)
# ==============================================================================
@torch.no_grad()
def generate_animation(model, input_image_path, num_frames, num_inference_steps=50):
    model.eval()
    config = model.config
    device = config.DEVICE
    print(f"Génération d'une animation de {num_frames} frames à partir de '{input_image_path}'...")
    input_image = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    z_app_raw = model.appearance_encoder(input_tensor)
    z_app = model.appearance_proj(z_app_raw).unsqueeze(1)
    z_motion = model.motion_prior.sample(1, device)
    z_motion_proj = model.motion_proj(z_motion)
    if num_frames != config.SEQUENCE_LENGTH:
        z_motion_proj = F.interpolate(z_motion_proj.transpose(1, 2), size=num_frames, mode='linear').transpose(1, 2)
    conditioning = torch.cat([z_app.repeat(1, num_frames, 1), z_motion_proj], dim=-1) # Correction: conditionnement par frame
    
    model.noise_scheduler.set_timesteps(num_inference_steps)
    h, w = config.IMG_SIZE // 8, config.IMG_SIZE // 8
    latents = torch.randn((1, num_frames, model.unet.config.in_channels, h, w), device=device)

    for t in tqdm(model.noise_scheduler.timesteps, desc="Dénaturation"):
        noise_pred_list = []
        for i in range(num_frames):
            # Créer un conditionnement spécifique pour la frame `i`
            frame_conditioning = torch.cat([z_app, z_motion_proj[:, i:i+1, :]], dim=1)
            noise_pred_i = model.unet(latents[:, i], t, frame_conditioning).sample
            noise_pred_list.append(noise_pred_i)
        
        latents = model.noise_scheduler.step(torch.stack(noise_pred_list, dim=1), t, latents).prev_sample

    latents = 1 / model.vae.config.scaling_factor * latents
    video_tensor = model.vae.decode(latents.permute(0, 2, 1, 3, 4)).sample
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in video_tensor[0].permute(1, 0, 2, 3)]
    
    base_filename = os.path.basename(input_image_path).split('.')[0]
    output_gif_path = os.path.join(config.OUTPUT_PATH, 'results', f"{base_filename}_animation.gif")
    output_mp4_path = os.path.join(config.OUTPUT_PATH, 'results', f"{base_filename}_animation.mp4")
    imageio.mimsave(output_gif_path, frames, fps=12)
    print(f"Animation GIF sauvegardée à : {output_gif_path}")
    imageio.mimsave(output_mp4_path, frames, fps=12)
    print(f"Animation MP4 sauvegardée à : {output_mp4_path}")


# ==============================================================================
#                        BLOC D'EXÉCUTION PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Initialisation ---
    print("Démarrage du pipeline CLeMo-Diff...")
    config = Config()
    print(f"Configuration chargée. Utilisation du device : {config.DEVICE}")
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_PATH, 'models'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_PATH, 'results'), exist_ok=True)

    # --- 2. Préparation des données ---
    create_dummy_dataset(config.DATASET_PATH, size=config.IMG_SIZE)

    # --- 3. Entraînement du modèle ---
    # NOTE: L'entraînement est coûteux. Sur un vrai dataset, cela prendra des heures/jours.
    # Le modèle retourné est l'état final après N époques.
    trained_model = train_clemo_diff(config)

    # --- 4. Génération d'une animation ---
    # Créer une image d'entrée factice pour le test de généralisation
    dummy_input_img = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    dummy_input_img[:, :] = [0, 0, 128] # Fond bleu
    dummy_input_img[100:150, 100:150] = [255, 0, 0] # Carré rouge au centre
    dummy_input_img_path = os.path.join(config.OUTPUT_PATH, "dummy_input_OOD.png")
    Image.fromarray(dummy_input_img).save(dummy_input_img_path)

    # Lancer la génération avec le modèle fraîchement entraîné
    generate_animation(
        model=trained_model,
        input_image_path=dummy_input_img_path,
        num_frames=32,
        num_inference_steps=50
    )
    
    print("Pipeline CLeMo-Diff terminé.")