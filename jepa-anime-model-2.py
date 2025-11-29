import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import glob
from einops import rearrange, repeat
from tqdm import tqdm
import math

# Pour le chargement vidéo ultra-rapide
try:
    from decord import VideoReader, cpu
except ImportError:
    print("ERREUR CRITIQUE: Installez decord ! (pip install decord)")
    exit()

# ==========================================
# 1. CONFIGURATION "RESEARCH GRADE"
# ==========================================
CONFIG = {
    "img_size": 128,          
    "num_frames": 16,         
    "tubelet_size": (2, 16, 16), # (t, h, w) -> Patch 3D
    "embed_dim": 512,         
    "depth": 8,               # Profondeur Encodeur
    "predictor_depth": 4,     # Profondeur Prédicteur
    "num_heads": 8,           
    "mlp_ratio": 4.0,
    "mask_ratio": 0.60,       # Masquage agressif
    "ema_decay": 0.996,       
    "batch_size": 8,          # Optimisé
    "lr": 1.5e-4,             # Légèrement augmenté pour la convergence initiale
    "min_lr": 1e-6,
    "weight_decay": 0.05,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. DATASET OPTIMISÉ (DECORD)
# ==========================================
class AnimeVideoDataset(Dataset):
    def __init__(self, folder_path, num_frames=16, img_size=128, step=1):
        # Recherche récursive pour trouver les vidéos partout
        # On cherche .mp4, .MP4, .mkv, .avi pour être sûr
        extensions = ['*.mp4', '*.MP4', '*.mkv', '*.avi']
        self.files = []
        for ext in extensions:
            self.files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
            
        self.num_frames = num_frames
        self.img_size = img_size
        self.step = step 
        
        # VÉRIFICATION CRITIQUE
        if len(self.files) == 0:
            print(f"❌ ERREUR: Aucune vidéo trouvée dans {folder_path}")
            print("   -> Vérifiez le chemin '/kaggle/input/...'")
            print("   -> Vérifiez que les fichiers sont bien des .mp4")
        else:
            print(f"✅ Dataset chargé : {len(self.files)} vidéos trouvées.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            vr = VideoReader(path, ctx=cpu(0), width=self.img_size, height=self.img_size)
            total_frames = len(vr)
            seq_len = self.num_frames * self.step
            
            if total_frames < seq_len:
                indices = np.arange(total_frames)
                indices = np.pad(indices, (0, seq_len - total_frames), mode='wrap')
            else:
                start = np.random.randint(0, total_frames - seq_len)
                indices = np.arange(start, start + seq_len, self.step)
            
            video = vr.get_batch(indices).asnumpy() 
            video = torch.from_numpy(video).float() / 255.0
            video = (video * 2) - 1 
            video = video.permute(3, 0, 1, 2) 
            return video
        except Exception as e:
            # En cas d'erreur de lecture, on retourne un tenseur vide pour ne pas crasher
            # Le dataloader gérera mal le None, donc on renvoie du noir
            print(f"⚠️ Erreur lecture {os.path.basename(path)}: {e}")
            return torch.zeros((3, self.num_frames, self.img_size, self.img_size))

# ==========================================
# 3. BRIQUES FONDAMENTALES (FACTORIZED ATTENTION)
# ==========================================
class FactorizedAttention(nn.Module):
    """
    Attention Spatiale et Temporelle découplée.
    Utilise Flash Attention (F.scaled_dot_product_attention) pour la vitesse.
    """
    def __init__(self, dim, num_heads=8, drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention optimisée
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., frames_per_group=8):
        super().__init__()
        self.frames = frames_per_group
        
        # Temporal
        self.norm_temp = nn.LayerNorm(dim)
        self.attn_temp = FactorizedAttention(dim, num_heads, drop)
        
        # Spatial
        self.norm_spat = nn.LayerNorm(dim)
        self.attn_spat = FactorizedAttention(dim, num_heads, drop)
        
        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, T, S):
        B, N, D = x.shape
        
        # 1. Temporal Attention
        # (B, T*S, D) -> (B*S, T, D)
        # On regroupe par position spatiale (S) pour faire attention sur le temps (T)
        xt = self.norm_temp(x).view(B, T, S, D).permute(0, 2, 1, 3).reshape(B*S, T, D)
        xt = self.attn_temp(xt)
        xt = xt.view(B, S, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + xt
        
        # 2. Spatial Attention
        # (B, T*S, D) -> (B*T, S, D)
        # On regroupe par frame (T) pour faire attention sur l'espace (S)
        xs = self.norm_spat(x).view(B, T, S, D).reshape(B*T, S, D)
        xs = self.attn_spat(xs)
        xs = xs.view(B, T, S, D).reshape(B, N, D)
        x = x + xs
        
        # 3. MLP
        x = x + self.mlp(self.norm_mlp(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_tokens_t, num_tokens_s, dim, depth, heads, mlp_ratio):
        super().__init__()
        self.num_tokens_t = num_tokens_t
        self.num_tokens_s = num_tokens_s
        self.layers = nn.ModuleList([
            SpatioTemporalBlock(dim, heads, mlp_ratio, frames_per_group=num_tokens_t)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights) # Initialisation robuste

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.num_tokens_t, self.num_tokens_s)
        return self.norm(x)

# ==========================================
# 4. COEUR DU SYSTÈME : V-JEPA
# ==========================================
class AnimeJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.t_patch, self.h_patch, self.w_patch = config['tubelet_size']
        self.num_t = config['num_frames'] // self.t_patch
        self.num_h = config['img_size'] // self.h_patch
        self.num_w = config['img_size'] // self.w_patch
        self.num_patches = self.num_t * self.num_h * self.num_w
        embed_dim = config['embed_dim']

        # --- Embedding 3D (Tubelet) ---
        self.patch_embed = nn.Conv3d(
            3, embed_dim, 
            kernel_size=config['tubelet_size'], 
            stride=config['tubelet_size']
        )
        self.norm_embed = nn.LayerNorm(embed_dim)
        
        # Positional Embedding (T, H, W) aplatis
        # Utilisation de learnable embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # --- Context Encoder ---
        self.context_encoder = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['depth'], config['num_heads'], config['mlp_ratio']
        )

        # --- Target Encoder (EMA) ---
        self.target_encoder = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['depth'], config['num_heads'], config['mlp_ratio']
        )
        # Copie des poids initiaux
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- Predictor ---
        self.predictor = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['predictor_depth'], config['num_heads'], config['mlp_ratio']
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # --- Decoder (Auxiliaire) ---
        # Sortie: pixels aplatis du patch (C * pt * ph * pw)
        patch_pixels = 3 * self.t_patch * self.h_patch * self.w_patch
        
        self.decoder_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_pixels),
            nn.Tanh() # Normalisation [-1, 1] des pixels
        )

    def patchify(self, video):
        # (B, C, T, H, W) -> (B, N, D)
        x = self.patch_embed(video) 
        x = x.flatten(2).transpose(1, 2) 
        x = self.norm_embed(x)
        return x

    def get_pixel_targets(self, video):
        # Prépare la vérité terrain des pixels pour la loss auxiliaire
        targets = rearrange(
            video, 
            'b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)', 
            pt=self.t_patch, ph=self.h_patch, pw=self.w_patch
        )
        return targets

    def forward_target(self, video):
        # Target Encoder voit tout (sans masque)
        x = self.patchify(video)
        x = x + self.pos_embed
        x = self.target_encoder(x)
        return x

    def forward(self, video):
        B = video.shape[0]
        
        # 1. Calcul des cibles (Target) avec EMA
        with torch.no_grad():
            target_features = self.forward_target(video)
            target_features = target_features.detach()

        # 2. Génération du Masque
        # Stratégie: Random Masking (peut être remplacé par Tube/Block masking plus tard)
        num_masked = int(self.config['mask_ratio'] * self.num_patches)
        rand_indices = torch.rand(B, self.num_patches, device=video.device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_masked] # Indices cachés
        
        # 3. Context Encoder
        input_tokens = self.patchify(video)
        
        # On ajoute l'embedding de position AVANT le masquage pour le contexte
        # C'est crucial : l'encodeur doit savoir où sont les patches visibles
        input_tokens = input_tokens + self.pos_embed
        
        # On crée un clone pour l'entrée contextuelle
        context_input = input_tokens.clone()
        
        # Masquage: On remplace par 0 ou on supprime. 
        # Pour JEPA standard, on laisse souvent le token mais "vide" (0) ou on le drop.
        # Ici on met à 0 pour garder la structure tensorielle simple.
        batch_range = torch.arange(B, device=video.device)[:, None]
        context_input[batch_range, mask_indices] = 0 
        
        context_features = self.context_encoder(context_input)
        
        # 4. Predictor
        # C'est ici l'étape CRITIQUE : Le predictor doit savoir OÙ prédire.
        predictor_input = context_features.clone()
        
        # On récupère les embeddings de position correspondant aux zones masquées
        # pos_embed: (1, N, D) -> expand -> (B, N, D)
        pos_embed_expanded = self.pos_embed.expand(B, -1, -1)
        pos_at_masks = pos_embed_expanded[batch_range, mask_indices]
        
        # Au lieu de juste mettre le mask_token, on met mask_token + pos_embedding
        # Cela informe le Prédicteur : "Devine ce qu'il y a à la position (x,y,t)"
        predictor_input[batch_range, mask_indices] = self.mask_token + pos_at_masks
        
        predicted_latents = self.predictor(predictor_input)

        # 5. Decoder Auxiliaire (Reconstruction Pixels)
        # On reconstruit à partir des latents prédits
        reconstructed_pixels = self.decoder_head(predicted_latents)
        
        return predicted_latents, target_features, reconstructed_pixels, mask_indices

# ==========================================
# 5. ENTRAÎNEMENT AVEC EMA
# ==========================================
def update_ema(model, decay):
    """Mise à jour exponentielle des poids Target <- Context"""
    with torch.no_grad():
        msd = model.context_encoder.state_dict()
        tsd = model.target_encoder.state_dict()
        for k in msd.keys():
            tsd[k].data.mul_(decay).add_(msd[k].data * (1. - decay))

def train():
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Définition du chemin (Kaggle vs Local)
    if os.path.exists("/kaggle/input/animes-videos/"):
        dataset_path = "/kaggle/input/animes-videos/"
    else:
        dataset_path = "./mes_animes" # Fallback local
        os.makedirs(dataset_path, exist_ok=True)

    # 2. Chargement Dataset
    dataset = AnimeVideoDataset(folder_path=dataset_path, 
                                num_frames=CONFIG['num_frames'], 
                                img_size=CONFIG['img_size'])
    
    if len(dataset) == 0:
        print("⛔ ARRÊT : Dataset vide. Ajoutez des fichiers .mp4 dans le dossier.")
        return

    # 3. Sécurité Batch Size
    # Si on a moins de vidéos que le batch size, on réduit le batch size
    effective_batch_size = min(CONFIG['batch_size'], len(dataset))
    
    # Si le dataset est tout petit, on désactive le drop_last pour ne pas perdre de données
    use_drop_last = True if len(dataset) > effective_batch_size else False

    print(f"ℹ️ Config: {len(dataset)} vidéos | Batch: {effective_batch_size} | Drop Last: {use_drop_last}")

    dataloader = DataLoader(dataset, 
                            batch_size=effective_batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            drop_last=use_drop_last)

    # Si malgré tout le dataloader est vide (ex: 0 batchs formés), on arrête
    if len(dataloader) == 0:
        print("⛔ ERREUR : Le DataLoader est vide (len=0). Vérifiez que batch_size <= nombre de vidéos.")
        return

    model = AnimeJEPA(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.95))
    scaler = GradScaler() 

    print(f"🚀 Démarrage Entraînement JEPA Anime sur {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        loop = tqdm(dataloader, desc=f"Ep {epoch+1}/{CONFIG['epochs']}")
        loss_avg_jepa = 0
        loss_avg_recon = 0
        
        # Compteur pour la moyenne manuelle
        steps = 0
        
        for i, video in enumerate(loop):
            video = video.to(CONFIG['device'], non_blocking=True)
            
            with autocast(): 
                pred_latents, target_latents, recon_pixels, mask_idx = model(video)
                
                # Loss JEPA
                B, N, D = pred_latents.shape
                batch_range = torch.arange(B, device=video.device)[:, None]
                pred_masked = pred_latents[batch_range, mask_idx]
                target_masked = target_latents[batch_range, mask_idx]
                
                loss_latent = F.smooth_l1_loss(pred_masked, target_masked, beta=1.0)
                
                # Loss Recon
                target_pixels = model.get_pixel_targets(video)
                loss_recon = F.mse_loss(recon_pixels, target_pixels)
                
                loss = 1.0 * loss_latent + 0.5 * loss_recon
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            update_ema(model, CONFIG['ema_decay'])
            
            loss_avg_jepa += loss_latent.item()
            loss_avg_recon += loss_recon.item()
            steps += 1
            
            if i % 10 == 0:
                loop.set_postfix(jepa=loss_latent.item(), recon=loss_recon.item())

        # Sauvegarde (Calcul de moyenne sécurisé)
        if steps > 0:
            avg_jepa = loss_avg_jepa / steps
        else:
            avg_jepa = 0.0

        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG
            }, f"checkpoints/jepa_anime_ep{epoch+1}.pt")
            print(f"✅ Modèle sauvegardé (loss jepa: {avg_jepa:.4f})")

if __name__ == "__main__":
    train()