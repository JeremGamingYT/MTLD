# -*- coding: utf-8 -*-
"""
MTLD: Modèle à Trajectoire Latente Déterministe pour la Restitution Séquentielle d'Anime

Version: 1.5 (Gestion de Données Multi-Séquences)
Auteur: Votre Expert en IA
Date: 01 novembre 2025
Description:
Cette version adapte le pipeline de données pour gérer une structure de dataset
contenant plusieurs séquences temporelles indépendantes dans des sous-dossiers
distincts (par exemple, différents arcs d'anime). La classe AnimeFrameDataset
a été repensée pour garantir que les séquences d'entraînement sont extraites
en respectant les frontières de chaque arc, préservant ainsi la cohérence
temporelle, ce qui est essentiel pour la qualité de l'apprentissage du modèle.
L'architecture du modèle et la logique d'entraînement restent inchangées.
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
# AJUSTEMENT: Importation des outils pour l'entraînement en précision mixte (AMP)
# RAISON: Accélère l'entraînement et réduit l'utilisation de la VRAM sur les cartes NVIDIA récentes.
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lpips
except ImportError:
    print("="*80); print("ERREUR: Bibliothèque LPIPS non trouvée."); print("!pip install lpips"); print("="*80); exit()

class Config:
    # MODIFIÉ: Le chemin pointe maintenant vers le dossier parent contenant les arcs
    DATASET_PATH = "animes_dataset" # Exemple: /path/to/your/dataset/
    
    # AJUSTEMENT: Réduction de la taille de l'image de 256x256 à 128x128.
    # RAISON: Réduit considérablement l'utilisation de la VRAM (d'un facteur 4 pour les activations).
    # C'est l'ajustement le plus important pour une carte avec 10 Go de VRAM.
    # NOTE: Cela réduira la résolution des images générées, mais garantit que l'entraînement est possible.
    IMG_SIZE = 128
    
    IMG_CHANNELS = 3
    
    # AJUSTEMENT: Réduction de la longueur de la séquence de 32 à 16.
    # RAISON: Divise par deux la mémoire nécessaire pour traiter une séquence,
    # ce qui allège la charge sur la VRAM et le modèle récurrent (GRU).
    TRAINING_SEQUENCE_LENGTH = 16
    
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    EPOCHS = 200
    
    # AJUSTEMENT: Réduction de la taille du lot (batch size) de 4 à 2.
    # RAISON: Diminue directement la quantité de données traitées simultanément par le GPU.
    # C'est un paramètre crucial pour contrôler l'utilisation de la VRAM.
    # NOTE: Si vous rencontrez toujours des erreurs "out of memory", essayez de le passer à 1.
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
    
    # AJUSTEMENT: NUM_WORKERS à 2 est un choix sûr pour un PC de bureau.
    # RAISON: Utilise des processus CPU supplémentaires pour charger les données sans surcharger le système.
    # Avec 12 Go de RAM disponible, c'est une valeur prudente qui n'entraînera pas de surconsommation de RAM.
    NUM_WORKERS = 2
    
    MODEL_SAVE_PATH = "./models_mtld_v1.5/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v1.5/"
    SAVE_EPOCH_INTERVAL = 10
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = ""

# --- 2. Préparation des Données (inchangée) ---

class AnimeFrameDataset(Dataset):
    """
    Dataset repensé pour gérer une structure de dossiers où chaque sous-dossier
    est une séquence temporelle distincte (un "arc").
    """
    def __init__(self, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []
        self.cumulative_lengths = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Le répertoire racine du dataset n'a pas été trouvé : {root_dir}")

        # 1. Identifier les sous-dossiers (arcs)
        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not arc_dirs:
            raise FileNotFoundError(f"Aucun sous-dossier (arc) trouvé dans {root_dir}")

        print(f"Détection de {len(arc_dirs)} arcs potentiels : {arc_dirs}")

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
                print(f"  -> Arc '{arc_dir}' validé : {len(image_paths)} images, {num_possible_sequences} séquences possibles.")
            else:
                print(f"  -> Arc '{arc_dir}' ignoré : {len(image_paths)} images (trop court pour une séquence de {self.sequence_length}).")

        if not self.sequences:
            raise ValueError("Aucun arc valide (suffisamment long) n'a été trouvé dans le dataset.")
            
        print(f"\nDataset initialisé : {len(self.sequences)} arcs valides, {self.total_sequences()} séquences d'entraînement au total.")

    def __len__(self):
        # La longueur totale est le nombre total de séquences possibles sur tous les arcs
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
        
    def total_sequences(self):
        return self.__len__()

    def __getitem__(self, idx):
        # 1. Déterminer à quel arc l'index global `idx` appartient
        # `bisect_right` trouverait l'insertion point, qui est l'index de l'arc
        arc_index = 0
        while idx >= self.cumulative_lengths[arc_index]:
            arc_index += 1
        
        # 2. Calculer l'index de départ local à l'intérieur de cet arc
        if arc_index == 0:
            local_start_idx = idx
        else:
            local_start_idx = idx - self.cumulative_lengths[arc_index - 1]
            
        # 3. Extraire la séquence de chemins d'images de l'arc correct
        sequence_paths = self.sequences[arc_index][local_start_idx : local_start_idx + self.sequence_length]
        
        # 4. Charger et transformer les images
        images = [Image.open(p).convert("RGB") for p in sequence_paths]
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
            
        return images

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = AnimeFrameDataset(
        root_dir=config.DATASET_PATH, 
        sequence_length=config.TRAINING_SEQUENCE_LENGTH,
        transform=transform
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
            # AJUSTEMENT: La taille de l'entrée de cette couche dépend de IMG_SIZE.
            # Pour IMG_SIZE=128, la carte de features est 2x2. Pour 256, elle est 4x4.
            # L'architecture actuelle s'adapte en changeant le Linear suivant.
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            # AJUSTEMENT: Calcul automatique de la taille pour la couche linéaire.
            # Cela rend le code robuste aux changements de IMG_SIZE.
            nn.Linear(1024 * (config.IMG_SIZE // 64) * (config.IMG_SIZE // 64), latent_dim)
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        # AJUSTEMENT: La sortie de cette couche doit correspondre à l'entrée de la suivante.
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
        # AJUSTEMENT: La taille du reshape doit correspondre à la nouvelle IMG_SIZE
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

# --- 4. Boucle d'Entraînement (AJUSTÉE pour la performance) ---

def train_mtld():
    config = Config() # Utilisation de l'instance de configuration locale
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
    
    # AJUSTEMENT: Initialisation des GradScalers pour la précision mixte.
    # RAISON: Gère la mise à l'échelle des gradients pour éviter les problèmes de "underflow"
    # lors de l'utilisation de Tensors float16, ce qui est essentiel pour l'AMP.
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    
    start_epoch = 0
    
    print("Début de l'entraînement du modèle conditionnel...")
    for epoch in range(start_epoch, config.EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for i, real_seq_imgs in enumerate(pbar):
            real_seq_imgs = real_seq_imgs.to(config.DEVICE)
            b, s, c, h, w = real_seq_imgs.shape
            
            priming_img = real_seq_imgs[:, 0, :, :, :]
            future_imgs = real_seq_imgs[:, 1:, :, :, :]
            
            # --- Entraînement du Générateur ---
            opt_g.zero_grad()
            
            # AJUSTEMENT: Utilisation de autocast pour le forward pass du générateur.
            # RAISON: Exécute les opérations dans un format plus efficace (float16)
            # pour réduire l'utilisation de la VRAM et augmenter la vitesse.
            with autocast(enabled=(config.DEVICE == "cuda")):
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
                
                # Le calcul de la perte FM nécessite des forward pass séparés
                _, real_features = model.discriminator(real_imgs_flat.detach(), extract_features=True)
                _, fake_features = model.discriminator(fake_imgs_flat, extract_features=True)
                loss_fm = sum(loss_l1(fake_f, real_f.detach()) for real_f, fake_f in zip(real_features, fake_features))
                
                loss_g = (config.LAMBDA_REC_L1 * loss_rec_l1 +
                          config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                          config.LAMBDA_LATENT * loss_latent +
                          config.LAMBDA_ADV * loss_adv +
                          config.LAMBDA_FM * loss_fm)

            # AJUSTEMENT: Mise à l'échelle de la perte et rétropropagation avec le scaler.
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            
            # --- Entraînement du Discriminateur ---
            opt_d.zero_grad()
            
            # AJUSTEMENT: Utilisation de autocast pour le forward pass du discriminateur.
            with autocast(enabled=(config.DEVICE == "cuda")):
                real_d_output, _ = model.discriminator(real_imgs_flat.detach())
                loss_d_real = loss_mse(real_d_output, target_real)
                
                fake_d_output, _ = model.discriminator(fake_imgs_flat.detach())
                target_fake = torch.zeros_like(fake_d_output)
                loss_d_fake = loss_mse(fake_d_output, target_fake)
                
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                
            # AJUSTEMENT: Mise à l'échelle de la perte et rétropropagation avec le scaler.
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            
            pbar.set_postfix({
                "L_Pred": f"{loss_latent.item():.3f}", "L_Rec": f"{loss_rec_l1.item():.3f}",
                "L_Adv": f"{loss_adv.item():.3f}", "L_D": f"{loss_d.item():.3f}",
            })
            
        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            save_path = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v1.5_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\nModèle sauvegardé : {save_path}")
            
            with torch.no_grad():
                model.eval()
                # La génération d'images peut se faire en pleine précision pour la qualité
                with autocast(enabled=(config.DEVICE == "cuda")):
                    fake_seq_imgs_eval = model.decoder(z_pred_full_seq)
                comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), fake_seq_imgs_eval[0].unsqueeze(0)], dim=0)
                comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                utils.save_image(comparison_flat, grid_path, nrow=s, normalize=True)
                print(f"Image de comparaison sauvegardée : {grid_path}")
                model.train()
                
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
    
    # Utiliser une nouvelle instance de Config pour la génération
    gen_config = Config()
    
    model = MTLD(gen_config).to(gen_config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=gen_config.DEVICE))
    model.eval()
    print("Modèle chargé avec succès.")

    transform = transforms.Compose([
        transforms.Resize((gen_config.IMG_SIZE, gen_config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(gen_config.DEVICE)

    print(f"Génération de {num_frames_to_generate} images à partir de l'image d'amorce.")

    with torch.no_grad():
        # L'inférence peut aussi bénéficier de l'autocast pour la vitesse
        with autocast(enabled=(gen_config.DEVICE == "cuda")):
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
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB_BGR))
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
    # Initialise la configuration globale utilisée par les modèles
    config = Config()

    # Placez votre base de données (avec les sous-dossiers d'arcs) dans le
    # dossier spécifié par Config.DATASET_PATH.
    
    # --- MODE 1: ENTRAÎNEMENT DU MODÈLE ---
    print("\n--- MODE ENTRAÎNEMENT (v1.5 Multi-Séquences, AJUSTÉ POUR 10Go VRAM) ---")
    train_mtld()

    # --- MODE 2: GÉNÉRATION CONDITIONNELLE ---
    # Décommentez cette section pour lancer la génération après l'entraînement.
    # print("\n--- MODE GÉNÉRATION (v1.5 Multi-Séquences) ---")
    # config_gen = Config()
    # model_file = "./models_mtld_v1.5/mtld_v1.5_epoch_200.pth" 
    
    # # L'image d'amorce peut provenir de n'importe où.
    # # Par exemple, la 50ème image du premier arc trouvé par le dataloader.
    # # Note: Cette partie est juste un exemple, vous devez adapter les chemins.
    # try:
    #     dummy_dataset = AnimeFrameDataset(config_gen.DATASET_PATH, 1)
    #     priming_image = dummy_dataset.sequences[0][50] 
    #     print(f"Utilisation de l'image d'amorce : {priming_image}")
    # except (IndexError, FileNotFoundError) as e:
    #     print(f"ATTENTION: Impossible de trouver une image d'amorce par défaut : {e}. Spécifiez un chemin valide.")
    #     priming_image = "/path/to/your/image.png" # REMPLACEZ CECI

    # if os.path.exists(priming_image):
    #     generate_sequence(
    #         model_path=model_file, 
    #         priming_image_path=priming_image, 
    #         num_frames_to_generate=100,
    #         config=config_gen
    #     )