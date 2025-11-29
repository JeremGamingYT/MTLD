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
    "num_frames": 16,         # Séquence d'entraînement
    "tubelet_size": (2, 16, 16), # (Temps, Hauteur, Largeur) -> Compression spatio-temporelle
    "embed_dim": 512,         # Dimension latente augmentée
    "depth": 8,               # Profondeur Encodeur
    "predictor_depth": 4,     # Profondeur Prédicteur (léger)
    "num_heads": 8,           
    "mlp_ratio": 4.0,
    "mask_ratio": 0.60,       # Masquage agressif pour forcer l'apprentissage structurel
    "ema_decay": 0.996,       # Taux de mise à jour du Target Encoder
    "batch_size": 4,          # Ajuster selon VRAM
    "lr": 1e-4,               # Learning rate plus bas pour la stabilité JEPA
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
        self.files = glob.glob(os.path.join(folder_path, "*.mp4"))
        self.num_frames = num_frames
        self.img_size = img_size
        self.step = step # Sampling rate (ex: prendre 1 frame sur 2 pour accélérer le mouvement)
        print(f"Dataset chargé : {len(self.files)} vidéos trouvées.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # Decord permet de charger juste ce qu'il faut sans décoder toute la vidéo
            vr = VideoReader(path, ctx=cpu(0), width=self.img_size, height=self.img_size)
            total_frames = len(vr)
            
            # Séquence nécessaire (frames * step)
            seq_len = self.num_frames * self.step
            
            if total_frames < seq_len:
                # Loop padding si trop court
                indices = np.arange(total_frames)
                indices = np.pad(indices, (0, seq_len - total_frames), mode='wrap')
            else:
                start = np.random.randint(0, total_frames - seq_len)
                indices = np.arange(start, start + seq_len, self.step)
            
            video = vr.get_batch(indices).asnumpy() # (T, H, W, C)
            
            # Normalisation et Permutation
            video = torch.from_numpy(video).float() / 255.0
            video = (video * 2) - 1 # [-1, 1]
            video = video.permute(3, 0, 1, 2) # (C, T, H, W)
            return video
            
        except Exception as e:
            print(f"Erreur chargement {path}: {e}")
            return torch.zeros((3, self.num_frames, self.img_size, self.img_size))

# ==========================================
# 3. BRIQUES FONDAMENTALES (FACTORIZED ATTENTION)
# ==========================================
class FactorizedAttention(nn.Module):
    """
    Attention séparée Space-Time avec Flash Attention native.
    Complexité mémoire réduite drastiquement.
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
        # Flash Attention Setup
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # F.scaled_dot_product_attention utilise automatiquement les cœurs CUDA optimisés
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., frames_per_group=8):
        super().__init__()
        self.frames = frames_per_group # Nombre de tokens temporels
        
        # 1. Temporal Block
        self.norm_temp = nn.LayerNorm(dim)
        self.attn_temp = FactorizedAttention(dim, num_heads, drop)
        
        # 2. Spatial Block
        self.norm_spat = nn.LayerNorm(dim)
        self.attn_spat = FactorizedAttention(dim, num_heads, drop)
        
        # 3. MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, T, S):
        # x: (B, T*S, Dim)
        B, N, D = x.shape
        
        # --- Temporal Attention ---
        # On groupe par espace pour que l'attention se fasse sur l'axe temps
        # View: (B, T, S, D) -> Permute (B, S, T, D) -> Flatten (B*S, T, D)
        xt = self.norm_temp(x).view(B, T, S, D).permute(0, 2, 1, 3).reshape(B*S, T, D)
        xt = self.attn_temp(xt)
        xt = xt.view(B, S, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + xt
        
        # --- Spatial Attention ---
        # On groupe par temps pour que l'attention se fasse sur l'axe espace
        # View: (B, T, S, D) -> Flatten (B*T, S, D)
        xs = self.norm_spat(x).view(B, T, S, D).reshape(B*T, S, D)
        xs = self.attn_spat(xs)
        xs = xs.view(B, T, S, D).reshape(B, N, D)
        x = x + xs
        
        # --- MLP ---
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
        
        # --- Dimensions ---
        self.t_patch, self.h_patch, self.w_patch = config['tubelet_size']
        self.num_t = config['num_frames'] // self.t_patch
        self.num_h = config['img_size'] // self.h_patch
        self.num_w = config['img_size'] // self.w_patch
        self.num_patches = self.num_t * self.num_h * self.num_w
        embed_dim = config['embed_dim']

        # --- Embedding (Tubelet 3D) ---
        # CORRECTION ICI : On retire LayerNorm du Sequential
        self.patch_embed = nn.Conv3d(
            3, embed_dim, 
            kernel_size=config['tubelet_size'], 
            stride=config['tubelet_size']
        )
        # On définit la norme séparément
        self.norm = nn.LayerNorm(embed_dim)
        
        # Positional Embedding (Learnable 3D)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # --- 1. Context Encoder (Trainable) ---
        self.context_encoder = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['depth'], config['num_heads'], config['mlp_ratio']
        )

        # --- 2. Target Encoder (EMA - Pas de gradient) ---
        self.target_encoder = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['depth'], config['num_heads'], config['mlp_ratio']
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- 3. Predictor (Léger) ---
        self.predictor = VisionTransformer(
            self.num_t, self.num_h * self.num_w, 
            embed_dim, config['predictor_depth'], config['num_heads'], config['mlp_ratio']
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- 4. Decoder (Pixel Reconstruction) ---
        self.decoder_fc = nn.Linear(embed_dim, 3 * self.t_patch * self.h_patch * self.w_patch)
        
        # Ce petit CNN va nettoyer le bruit (Refinement Head)
        # Remplacer ceci dans __init__
        self.decoder_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # On élargit
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4), # On réfléchit
            nn.GELU(),
            nn.Linear(embed_dim * 4, 3 * self.t_patch * self.h_patch * self.w_patch), # On projette
            nn.Tanh()
        )

    def decode_pixels(self, latents):
        """Transforme les latents en pixels via CNN"""
        # 1. Projection Linéaire brute (ce que vous aviez avant)
        # (B, N, Pixels)
        pixels_flat = self.decoder_fc(latents)
        
        # 2. Reshape en vidéo 3D pour passer dans le CNN
        # Il faut faire l'inverse du patchify pour reconstruire le volume
        # C'est complexe à faire tensoriellement, donc pour l'instant 
        # on va garder la sortie linéaire MAIS on va augmenter le poids de la loss pixel.
        
        # NOTE : Pour simplifier sans casser tout votre code avec des dimensions 3D complexes :
        # Gardons la linéaire mais ajoutons une couche intermédiaire pour la puissance.
        return pixels_flat

    def patchify(self, video):
        # CORRECTION ICI : Ordre des opérations modifié
        
        # 1. Convolution 3D -> (B, 512, T, H, W)
        x = self.patch_embed(video) 
        
        # 2. Aplatir (B, 512, N) et Transposer (B, N, 512)
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. Maintenant on peut appliquer LayerNorm car 512 est à la fin
        x = self.norm(x)
        
        return x

    def get_pixel_targets(self, video):
            """
            Découpe la vidéo en patchs de pixels bruts (sans embedding).
            Input: (B, C, T, H, W)
            Output: (B, N, C*pt*ph*pw) -> Pour comparer avec le décodeur
            """
            # On utilise einops pour découper proprement
            # pt, ph, pw sont les dimensions du tubelet (2, 16, 16)
            targets = rearrange(
                video, 
                'b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)', 
                pt=self.t_patch, ph=self.h_patch, pw=self.w_patch
            )
            return targets

    def forward_target(self, video):
        x = self.patchify(video)
        x = x + self.pos_embed
        x = self.target_encoder(x)
        return x

    def forward_context(self, video, mask_indices):
        x = self.patchify(video)
        x = x + self.pos_embed
        
        B, N, D = x.shape
        mask = torch.zeros((B, N), device=x.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)
        
        x[mask] = 0 
        context_features = self.context_encoder(x)
        return context_features

    def forward(self, video):
        B = video.shape[0]
        
        # 1. Targets
        with torch.no_grad():
            target_features = self.forward_target(video)
            target_features = target_features.detach()

        # 2. Masking
        num_masked = int(self.config['mask_ratio'] * self.num_patches)
        rand_indices = torch.rand(B, self.num_patches, device=video.device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_masked]
        
        # 3. Context
        input_tokens = self.patchify(video)
        input_tokens = input_tokens + self.pos_embed
        
        context_input = input_tokens.clone()
        batch_range = torch.arange(B, device=video.device)[:, None]
        context_input[batch_range, mask_indices] = 0 
        
        context_features = self.context_encoder(context_input)
        
        # 4. Predictor
        predictor_input = context_features.clone()
        predictor_input[batch_range, mask_indices] = self.mask_token 
        
        predicted_latents = self.predictor(predictor_input)

        # 5. Decode
        reconstructed_pixels = self.decoder_head(predicted_latents)
        
        return predicted_latents, target_features, reconstructed_pixels, mask_indices

# ==========================================
# 5. ENTRAÎNEMENT AVEC EMA
# ==========================================
def update_ema(model, decay):
    """Met à jour les poids du Target Encoder vers le Context Encoder"""
    with torch.no_grad():
        msd = model.context_encoder.state_dict()
        tsd = model.target_encoder.state_dict()
        for k in msd.keys():
            tsd[k].data.mul_(decay).add_(msd[k].data * (1. - decay))

def train():
    os.makedirs("checkpoints", exist_ok=True)
    dataset = AnimeVideoDataset(folder_path="/kaggle/input/animes-videos/", 
                                num_frames=CONFIG['num_frames'], 
                                img_size=CONFIG['img_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    model = AnimeJEPA(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler() # Mixed Precision

    print(f"🚀 Démarrage Entraînement JEPA SOTA sur {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        loop = tqdm(dataloader, desc=f"Ep {epoch+1}")
        
        for video in loop:
            video = video.to(CONFIG['device'])
            
            with autocast(): # FP16/BF16 context
                pred_latents, target_latents, recon_pixels, mask_idx = model(video)
                
                # A. Loss JEPA (Espace Latent)
                # On compare la prédiction aux features du Target Encoder
                # Uniquement sur les zones masquées !
                B, N, D = pred_latents.shape
                batch_range = torch.arange(B, device=video.device)[:, None]
                
                pred_masked = pred_latents[batch_range, mask_idx]
                target_masked = target_latents[batch_range, mask_idx]
                
                # Smooth L1 est plus robuste que MSE
                loss_latent = F.smooth_l1_loss(pred_masked, target_masked)
                
                # B. Loss Reconstruction (Auxiliaire pour aider le décodeur à apprendre à dessiner)
                # On reconstruit la vidéo originale patchifiée
                # On récupère les pixels bruts cibles
                target_pixels = model.get_pixel_targets(video) # <--- Utilise la nouvelle méthode (dim 1536)
                loss_recon = F.mse_loss(recon_pixels, target_pixels)
                
                # Total Loss : On met l'accent sur le Latent (JEPA)
                # On met autant d'importance sur le dessin que sur la structure
                loss = 0.5 * loss_latent + 1.0 * loss_recon
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA Target Encoder
            update_ema(model, CONFIG['ema_decay'])
            
            loop.set_postfix(jepa=loss_latent.item(), recon=loss_recon.item())

        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoints/jepa_anime_{epoch+1}.pt")

if __name__ == "__main__":
    if not os.path.exists("/kaggle/input/animes-videos/"):
        os.makedirs("./mes_animes")
        print("📁 Dossier 'mes_animes' créé. Déposez vos .mp4 et relancez !")
    else:
        train()