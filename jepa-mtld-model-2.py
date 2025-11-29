"""
JEPA + MTLD Hybrid
- Encodeur / prédicteur : JEPA vidéo (transformer spatio-temporel)
- Décodeur : CompositionalAnimeDecoder (MTLD v3.2)
- Dataset : AnimeFrameDataset (séquences d'images)
"""

import os, glob, re, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

try:
    import lpips
except ImportError:
    print("="*80)
    print("WARN: lpips non installé. Perte LPIPS désactivée.")
    print("Installe-le avec : pip install lpips")
    print("="*80)
    lpips = None


# ============================================================
# 0. CONFIG
# ============================================================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    # Dataset
    DATASET_PATH = os.environ.get(
        "JEPA_DATASET_PATH",
        "/root/.cache/kagglehub/datasets/jeremgaming099/anima-s-dataset/versions/9/animes_dataset/animes_dataset",
    )
    PRELOAD_DATASET_IN_RAM = True
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    SEQ_LEN = 16

    # Latents / décodeur MTLD
    LATENT_DIM = 256
    NOISE_EMB_DIM = 16
    PALETTE_K = 10
    HGU_WIDTH = 256

    # JEPA
    TUBELET_SIZE = (1, 16, 16)   # (T, H, W) -> T=1 pour garder SEQ_LEN frames
    EMBED_DIM = LATENT_DIM       # pour matcher le décodeur
    DEPTH = 4
    PREDICTOR_DEPTH = 2
    NUM_HEADS = 8
    MLP_RATIO = 4.0
    MASK_RATIO = 0.85
    EMA_DECAY = 0.996

    # Optim / training
    EPOCHS = 50
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    NUM_WORKERS = 4
    DEVICE = get_device()

    # Loss weights
    LAMBDA_LATENT = 1.0
    LAMBDA_REC_L1 = 15.0
    LAMBDA_REC_LPIPS = 5.0
    LAMBDA_RPC = 6.0
    LAMBDA_EDGE = 4.0
    LAMBDA_TV_ASSIGN = 0.15
    LAMBDA_PALETTE = 1.0
    LAMBDA_TEMPORAL_PALETTE = 1.0
    
    # --- AJOUT ---
    LAMBDA_KL = 1e-6  # Poids très faible pour ne pas casser la reconstruction au début

    RPC_MARGIN = 0.5
    RPC_PATCH_SIZE = 24
    RPC_PROJ_DIM = 128

    MODEL_SAVE_PATH = "./models_jepa_mtld/"
    OUTPUT_SAVE_PATH = "./outputs_jepa_mtld/"
    SAVE_EPOCH_INTERVAL = 5
    MAX_CHECKPOINTS = 3


# ============================================================
# 1. DATASET (reprend ton AnimeFrameDataset)
# ============================================================
class AnimeFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None, config: Config=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.config = config
        self.sequences = []
        self.cumulative_lengths = []
        self.preloaded_data = None

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset not found: {root_dir}\n"
                f"Set the JEPA_DATASET_PATH environment variable to point to your dataset root."
            )

        arc_dirs = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            image_paths = sorted(
                glob.glob(os.path.join(arc_path, "*.png")),
                key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
            )
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_seq = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_seq
                self.cumulative_lengths.append(total_valid_sequences)

        if not self.sequences:
            raise ValueError("No valid arcs found in dataset.")

        if self.config and self.config.PRELOAD_DATASET_IN_RAM:
            self._preload_images()

    def _preload_images(self):
        all_paths = sorted(list(set(path for arc in self.sequences for path in arc)))
        self.preloaded_data = {}
        for path in all_paths:
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    self.preloaded_data[path] = self.transform(img)
            except Exception:
                pass

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
        else:
            images_pil = [Image.open(p).convert("RGB") for p in sequence_paths]
            images = [self.transform(img) for img in images_pil] if self.transform else images_pil
        # (S, C, H, W)
        return torch.stack(images)


def get_dataloader(config: Config):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = AnimeFrameDataset(
        root_dir=config.DATASET_PATH,
        sequence_length=config.SEQ_LEN,
        transform=transform,
        config=config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    return dataloader, dataset.total_sequences()


# ============================================================
# 2. DÉCODEUR MTLD (CompositionalAnimeDecoder + helpers)
# ============================================================
class HarmonicGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mix = nn.Conv2d(c, c, 1, bias=True)
        self.high = nn.Conv2d(c, c, 3, padding=1, bias=False, groups=c)
        nn.init.dirac_(self.high.weight)
    def forward(self, x):
        h = self.high(x) - x
        g = torch.sigmoid(self.mix(x))
        return x + g * h


class CompositionalAnimeDecoder(nn.Module):
    """
    Copie adaptée de ton CompositionalAnimeDecoder (sans FiLM / condition).
    Entrée:
        z: (B,S,D) latents temporels
        noise_code: (B,S,noise_dim)
    Sortie:
        imgs: (B,S,3,H,W) dans [-1,1]
        assign_logits: (B,S,K,H,W)
        palette_flat: (B,S,K*3)
        shading: (B,S,3,H,W)
        edge_mask: (B,S,1,H,W)
    """
    def __init__(self, latent_dim: int, out_channels: int, noise_dim: int,
                 img_size: int, K: int, width: int):
        super().__init__()
        self.img_size = img_size
        self.K = K
        fm = img_size // 64

        self.fc = nn.Linear(latent_dim, 1024 * fm * fm)
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                HarmonicGate(out_c),
                nn.ReLU(inplace=True),
            )

        ch0 = 1024
        ch1 = width
        ch2 = width // 2
        ch3 = width // 4
        ch4 = width // 8

        self.up1 = up(1024, ch0)
        self.up2 = up(ch0, ch1)
        self.up3 = up(ch1, ch2)
        self.up4 = up(ch2, ch3)
        self.up5 = up(ch3, ch4)

        self.assign_head = nn.ConvTranspose2d(ch4, K, 4, 2, 1)
        self.shade_head = nn.ConvTranspose2d(ch4, out_channels, 4, 2, 1)
        self.edge_head = nn.ConvTranspose2d(ch4, 1, 4, 2, 1)

        self.palette_mlp = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, K * 3)
        )

    def forward(self, z, noise_code):
        seq = (z.dim() == 3)  # (B,S,D)
        if seq:
            b, s, d = z.shape
            z_flat = z.view(b * s, d)
            n_flat = noise_code.view(b * s, -1)
        else:
            b, d = z.shape
            s = 1
            z_flat = z
            n_flat = noise_code

        fm = self.img_size // 64
        base = self.fc(z_flat).view(b * s, 1024, fm, fm)
        nemb = self.noise_mlp(n_flat)

        x = self.up1(base)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        assign_logits = self.assign_head(x)           # (B*S, K, H, W)
        shading = torch.tanh(self.shade_head(x))      # (B*S, 3, H, W)
        edge_mask = torch.sigmoid(self.edge_head(x))  # (B*S, 1, H, W)

        pal = torch.tanh(
            self.palette_mlp(torch.cat([z_flat, nemb], dim=1))
        ).view(b * s, self.K, 3)                      # (B*S, K, 3)

        H = W = self.img_size
        assign = torch.softmax(assign_logits, dim=1)  # (B*S, K, H, W)
        assign_ = assign.view(b * s, self.K, H * W)
        pal_ = pal.view(b * s, 3, self.K)
        base_colors = torch.bmm(pal_, assign_).view(b * s, 3, H, W)
        img = torch.clamp(base_colors + shading * edge_mask, -1.0, 1.0)

        def unflat(t):
            return t.view(b, s, *t.shape[1:])

        return (
            unflat(img),                # (B,S,3,H,W)
            unflat(assign_logits),      # (B,S,K,H,W)
            unflat(pal.view(b * s, self.K * 3)),  # (B,S,K*3)
            unflat(shading),            # (B,S,3,H,W)
            unflat(edge_mask)           # (B,S,1,H,W)
        )


class PerceptualProjector(nn.Module):
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
        f = self.net(x).flatten(1)
        e = self.head(f)
        return F.normalize(e, dim=1)


def sample_noise_code(shape, device):
    return torch.randn(*shape, device=device)


# ============================================================
# 3. PERTES VISUELLES MTLD (reprises)
# ============================================================
def degrade(imgs):
    is_seq = (imgs.dim() == 5)
    x = imgs if not is_seq else imgs.reshape(-1, *imgs.shape[2:])
    with torch.no_grad():
        x = (x + 1) / 2
        k = 3
        x_blur = nn.AvgPool2d(k, stride=1, padding=k // 2)(x)
        noise = 0.02 * torch.randn_like(x)
        x = torch.clamp(x_blur + noise, 0, 1)
        x = x * 2 - 1
    return x if not is_seq else x.view_as(imgs)


def ranked_patch_contrast_loss(projector: PerceptualProjector, real_imgs, fake_imgs,
                               margin: float, patch_size: int):
    b, s, c, h, w = fake_imgs.shape
    real = real_imgs.reshape(b * s, c, h, w)
    fake = fake_imgs.reshape(b * s, c, h, w)
    deg = degrade(real)

    ps = int(min(patch_size, h, w))
    ps = max(ps, 2)

    def pick_unfold_patches(x):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=ps, stride=1)
        L = patches.size(-1)
        idx = torch.randint(0, L, (B,), device=x.device)
        idx_expand = idx.view(B, 1, 1).expand(B, C * ps * ps, 1)
        sel = torch.gather(patches, 2, idx_expand).view(B, C, ps, ps)
        return sel

    r = pick_unfold_patches(real)
    f = pick_unfold_patches(fake)
    d = pick_unfold_patches(deg)

    er = projector((r + 1) / 2)
    ef = projector((f + 1) / 2)
    ed = projector((d + 1) / 2)

    def dist(a, b):
        return 1 - (a * b).sum(-1)

    loss1 = torch.relu(-(dist(er, er) - dist(er, ef)) + margin).mean()
    loss2 = torch.relu(-(dist(ef, er) - dist(ef, ed)) + margin).mean()
    return 0.5 * (loss1 + loss2)


def sobel_edges(x):
    x01 = (x + 1) / 2
    gray = 0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]
    gray = gray.unsqueeze(1)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.amax(dim=[2, 3], keepdim=True) + 1e-6)
    return mag


def edge_loss(pred_edge_mask, real_imgs):
    b, s, c, h, w = real_imgs.shape
    real = real_imgs.reshape(b * s, c, h, w)
    target = sobel_edges(real).detach()
    pred = pred_edge_mask.reshape(b * s, 1, h, w)
    return F.mse_loss(pred, target)


def tv_loss_assign(assign_logits):
    b, s, k, h, w = assign_logits.shape
    assign = torch.softmax(assign_logits, dim=2)
    dx = assign[:, :, :, :, 1:] - assign[:, :, :, :, :-1]
    dy = assign[:, :, :, 1:, :] - assign[:, :, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())


def palette_chamfer_loss(palette_flat, real_imgs, K, num_samples=2048):
    b, s, _, h, w = real_imgs.shape
    pal = palette_flat.view(b * s, K, 3)
    img = real_imgs.view(b * s, 3, h, w)
    img = (img + 1) / 2
    pal01 = (pal + 1) / 2
    N = min(num_samples, h * w)
    idx = torch.randint(0, h * w, (b * s, N), device=img.device)
    img_flat = img.view(b * s, 3, -1).transpose(1, 2)
    pix = torch.gather(img_flat, 1, idx.unsqueeze(-1).expand(b * s, N, 3))
    pix2 = pix.unsqueeze(2)
    pal2 = pal01.unsqueeze(1)
    d = (pix2 - pal2).pow(2).sum(-1)
    min_pix = d.min(dim=2).values.mean()
    min_pal = d.min(dim=1).values.mean()
    return 0.5 * (min_pix + min_pal)


def temporal_palette_loss(palette_flat):
    pal = palette_flat.view(palette_flat.size(0), palette_flat.size(1), -1)
    if pal.size(1) < 2:
        return torch.tensor(0.0, device=pal.device)
    diff = pal[:, 1:, :] - pal[:, :-1, :]
    return diff.pow(2).mean()


# ============================================================
# 4. JEPA BLOCKS (Factorized ViT)
# ============================================================
class FactorizedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop=0.0):
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
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, frames_per_group=8):
        super().__init__()
        self.frames = frames_per_group
        self.norm_temp = nn.LayerNorm(dim)
        self.attn_temp = FactorizedAttention(dim, num_heads, drop)

        self.norm_spat = nn.LayerNorm(dim)
        self.attn_spat = FactorizedAttention(dim, num_heads, drop)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x, T, S):
        B, N, D = x.shape

        # Temporal attn
        xt = self.norm_temp(x).view(B, T, S, D).permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt = self.attn_temp(xt)
        xt = xt.view(B, S, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + xt

        # Spatial attn
        xs = self.norm_spat(x).view(B, T, S, D).reshape(B * T, S, D)
        xs = self.attn_spat(xs)
        xs = xs.view(B, T, S, D).reshape(B, N, D)
        x = x + xs

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


# ============================================================
# 5. MODÈLE HYBRIDE JEPA + MTLD
# ============================================================
# ============================================================
# 5. MODÈLE HYBRIDE JEPA + REGULARISATION + MTLD
# ============================================================

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            # CORRECTION ICI :
            # On calcule la divergence KL.
            # Au lieu de forcer dim=[1,2,3], on somme sur toutes les dimensions sauf le batch (dim 0).
            # Cela marche pour (B, T, D) comme pour (B, C, H, W).
            dims = list(range(1, len(self.mean.shape)))
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)

class RegularizedJEPAMTLD(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.EMBED_DIM
        
        # Patch embedding 3D
        self.patch_embed = nn.Conv3d(
            config.IMG_CHANNELS, self.dim,
            kernel_size=config.TUBELET_SIZE, stride=config.TUBELET_SIZE
        )
        self.norm = nn.LayerNorm(self.dim)
        
        self.t_patch, self.h_patch, self.w_patch = config.TUBELET_SIZE
        self.num_t = config.SEQ_LEN // self.t_patch
        self.num_h = config.IMG_SIZE // self.h_patch
        self.num_w = config.IMG_SIZE // self.w_patch
        self.num_patches = self.num_t * self.num_h * self.num_w

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.dim) * 0.02)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # JEPA Encoders
        self.context_encoder = VisionTransformer(self.num_t, self.num_h * self.num_w, self.dim, config.DEPTH, config.NUM_HEADS, config.MLP_RATIO)
        # On peut garder un target encoder pour la loss JEPA, mais pour la régularisation VAE, 
        # on utilise la sortie directe du prédicteur ou de l'encodeur.
        self.target_encoder = VisionTransformer(self.num_t, self.num_h * self.num_w, self.dim, config.DEPTH, config.NUM_HEADS, config.MLP_RATIO)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters(): p.requires_grad = False
        
        self.predictor = VisionTransformer(self.num_t, self.num_h * self.num_w, self.dim, config.PREDICTOR_DEPTH, config.NUM_HEADS, config.MLP_RATIO)

        # --- PONT VARIATIONNEL (REGULARISATION) ---
        # Projection de l'espace latent brut vers (Mean, LogVar)
        self.vae_proj = nn.Linear(self.dim, self.dim * 2) 
        # Facteur d'échelle pour stabiliser la variance (similaire à SD)
        self.register_buffer('scale_factor', torch.tensor(0.18215)) 

        # Décodeur MTLD
        self.decoder = CompositionalAnimeDecoder(
            latent_dim=config.LATENT_DIM, out_channels=config.IMG_CHANNELS,
            noise_dim=config.NOISE_EMB_DIM, img_size=config.IMG_SIZE,
            K=config.PALETTE_K, width=config.HGU_WIDTH
        )
        self.proj = PerceptualProjector(config.RPC_PROJ_DIM)

    def patchify(self, video):
        # video: (B, C, T, H, W)
        x = self.patch_embed(video).flatten(2).transpose(1, 2)
        return self.norm(x)

    def encode_to_z(self, video_seq):
        """
        UTILITAIRE POUR PHASE 2 (DIFFUSION)
        Encode une vidéo brute directement en latent régularisé Z.
        """
        # 1. Encode sans masque (Target Encoder ou Context complet)
        with torch.no_grad():
            video = video_seq.permute(0, 2, 1, 3, 4)
            x = self.patchify(video)
            x_pos = x + self.pos_embed
            tokens = self.target_encoder(x_pos) # (B, N, D)
            
            # 2. Reshape pour avoir (B, T, D)
            B, N, D = tokens.shape
            x_reshaped = tokens.view(B, self.num_t, self.num_h * self.num_w, D)
            z_spatial_avg = x_reshaped.mean(dim=2) # (B, T, D)
            
            # 3. Projection VAE
            moments = self.vae_proj(z_spatial_avg)
            posterior = DiagonalGaussianDistribution(moments)
            
            # 4. Sample & Scale
            z = posterior.sample() * self.scale_factor
            return z

    @torch.no_grad()
    def update_ema(self, decay):
        msd = self.context_encoder.state_dict()
        tsd = self.target_encoder.state_dict()
        for k in msd.keys():
            tsd[k].data.mul_(decay).add_(msd[k].data * (1.0 - decay))

    def forward(self, seq_imgs_bschw):
        B, S, C, H, W = seq_imgs_bschw.shape
        video = seq_imgs_bschw.permute(0, 2, 1, 3, 4)

        # 1. Patchify
        x = self.patchify(video)
        
        # 2. Target Encoder (EMA) -> Vérité Terrain
        x_target = x + self.pos_embed
        with torch.no_grad():
            target_tokens = self.target_encoder(x_target.detach())

        # 3. Masking
        num_masked = int(self.config.MASK_RATIO * self.num_patches)
        rand_indices = torch.rand(B, self.num_patches, device=video.device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_masked]
        batch_range = torch.arange(B, device=video.device)[:, None]

        # 4. Context Encoder
        context_input = x.clone()
        context_input[batch_range, mask_indices] = self.enc_mask_token
        context_input = context_input + self.pos_embed
        context_features = self.context_encoder(context_input)

        # 5. Predictor
        predictor_input = context_features.clone()
        mask_tokens_pos = (self.mask_token + self.pos_embed).expand(B, -1, -1)
        predictor_input[batch_range, mask_indices] = mask_tokens_pos[batch_range, mask_indices]
        pred_tokens = self.predictor(predictor_input)

        # 6. PONT VARIATIONNEL (La nouveauté)
        # On convertit les tokens prédits en latents temporels
        # (B, N, D) -> (B, T, H*W, D) -> Mean -> (B, T, D)
        pred_reshaped = pred_tokens.view(B, self.num_t, self.num_h * self.num_w, self.dim)
        z_raw = pred_reshaped.mean(dim=2) 
        
        # Projection vers (Mean, LogVar)
        moments = self.vae_proj(z_raw)
        posterior = DiagonalGaussianDistribution(moments)
        z_sample = posterior.sample() # C'est notre Z régularisé
        
        # 7. Decodeur
        noise_code = sample_noise_code((B, self.config.SEQ_LEN, self.config.NOISE_EMB_DIM), device=video.device)
        fake_seq, assign_logits, palette_flat, shading, edge_mask = self.decoder(z_sample, noise_code)

        return (pred_tokens, target_tokens, 
                fake_seq, assign_logits, palette_flat, shading, edge_mask, 
                mask_indices, posterior) # On retourne posterior pour la loss

# ============================================================
# 6. TRAINING LOOP
# ============================================================
def save_checkpoint(epoch, model, optimizer, scaler, config: Config):
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    filename = os.path.join(config.MODEL_SAVE_PATH, f"jepa_mtld_epoch_{epoch}.pth")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(state, filename)
    print(f"\nCheckpoint saved: {filename}")

    # prune old
    files = [f for f in os.listdir(config.MODEL_SAVE_PATH)
             if f.startswith("jepa_mtld_epoch_") and f.endswith(".pth")]
    def _epoch_num(fname):
        nums = re.findall(r"(\d+)", fname)
        return int(nums[-1]) if nums else -1
    files.sort(key=_epoch_num)
    if len(files) > config.MAX_CHECKPOINTS:
        for f in files[:-config.MAX_CHECKPOINTS]:
            try:
                os.remove(os.path.join(config.MODEL_SAVE_PATH, f))
            except Exception:
                pass


def train_jepa_mtld():
    config = Config()
    print(f"Device: {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)

    dataloader, total_sequences = get_dataloader(config)
    print(f"Total sequences: {total_sequences}")

    # --- CORRECTION ICI : On utilise la nouvelle classe Régularisée ---
    model = RegularizedJEPAMTLD(config).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    if lpips is not None:
        lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    else:
        lpips_vgg = None

    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda"))

    print("\n=== Start JEPA+MTLD (Regularized) training ===")
    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for i, real_seq in enumerate(pbar):
            real_seq = real_seq.to(config.DEVICE)   # (B,S,C,H,W)
            B, S, C, H, W = real_seq.shape

            optimizer.zero_grad(set_to_none=True)
            # Nouvelle syntaxe PyTorch standard
            with torch.amp.autocast('cuda', enabled=(config.DEVICE == "cuda")):
                # Appel du modèle qui retourne maintenant 'posterior'
                (pred_tokens, target_tokens,
                 fake_seq, assign_logits, palette_flat,
                 shading, edge_mask, mask_idx, posterior) = model(real_seq)

                # JEPA latent loss sur tokens masqués
                B_, N, D = pred_tokens.shape
                batch_range = torch.arange(B_, device=real_seq.device)[:, None]
                pred_masked = pred_tokens[batch_range, mask_idx]
                target_masked = target_tokens[batch_range, mask_idx]
                loss_latent = F.smooth_l1_loss(pred_masked, target_masked)

                # L1 reconstruction
                loss_rec_l1 = F.l1_loss(fake_seq, real_seq)

                # LPIPS
                if lpips_vgg is not None:
                    fake_flat = fake_seq.view(B * S, C, H, W)
                    real_flat = real_seq.view(B * S, C, H, W)
                    loss_rec_lpips = lpips_vgg(fake_flat, real_flat).mean()
                else:
                    loss_rec_lpips = torch.tensor(0.0, device=real_seq.device)

                # RPC
                loss_rpc = ranked_patch_contrast_loss(
                    model.proj, real_seq, fake_seq,
                    config.RPC_MARGIN, config.RPC_PATCH_SIZE
                )

                # Edges
                loss_edges = edge_loss(edge_mask, real_seq)

                # TV assign
                loss_tv = tv_loss_assign(assign_logits)

                # Palette
                loss_pal = palette_chamfer_loss(
                    palette_flat, real_seq,
                    config.PALETTE_K, num_samples=1536
                )
                loss_pal_temp = temporal_palette_loss(palette_flat)

                # --- KL DIVERGENCE (NOUVEAU) ---
                # Empêche l'explosion des latents pour la diffusion future
                loss_kl = posterior.kl().mean()

                total_loss = (
                    config.LAMBDA_LATENT * loss_latent +
                    config.LAMBDA_REC_L1 * loss_rec_l1 +
                    config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                    config.LAMBDA_RPC * loss_rpc +
                    config.LAMBDA_EDGE * loss_edges +
                    config.LAMBDA_TV_ASSIGN * loss_tv +
                    config.LAMBDA_PALETTE *
                    (loss_pal + config.LAMBDA_TEMPORAL_PALETTE * loss_pal_temp) +
                    config.LAMBDA_KL * loss_kl  # <--- AJOUT
                )

            scaler.scale(total_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            model.update_ema(config.EMA_DECAY)

            pbar.set_postfix({
                "L_lat": f"{loss_latent.item():.3f}",
                "L_rec": f"{loss_rec_l1.item():.3f}",
                "L_kl": f"{loss_kl.item():.5f}", # On surveille KL
            })

        if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
            save_checkpoint(epoch + 1, model, optimizer, scaler, config)
            # Preview code (inchangé, sauf appel model)
            try:
                model.eval()
                with torch.no_grad():
                    real_preview = real_seq[:1]
                    # Note: on ignore les sorties extra ici
                    outputs = model(real_preview)
                    fake_seq = outputs[2] # fake_seq est le 3eme element maintenant
                    comp = torch.cat([real_preview, fake_seq], dim=0)
                    comp_flat = comp.permute(1, 0, 2, 3, 4).reshape(-1, C, H, W)
                    grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"preview_epoch_{epoch+1}.png")
                    utils.save_image(comp_flat, grid_path, nrow=S, normalize=True, padding=0)
            except Exception as e:
                print(f"[WARN] preview failed: {e}")

    print("Training finished.")

# ============================================================
# 8. LATENT DIFFUSION MODEL (Le Cerveau Créatif)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LatentDenoiser(nn.Module):
    """
    Un Transformer simple qui travaille sur la séquence temporelle de latents.
    Input: (B, T, D) -> Latents bruités
    Condition: (B, 1, D) -> Premier frame latent (l'image d'entrée)
    Output: (B, T, D) -> Noise prédiction
    """
    def __init__(self, latent_dim, num_layers=6, num_heads=8, d_ff=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=num_heads, 
            dim_feedforward=d_ff, 
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection
        self.final_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, t, condition_z0):
        """
        x: (B, T, D) sequence of latents (noisy)
        t: (B,) timesteps
        condition_z0: (B, 1, D) latent of the first frame (clean)
        """
        # 1. Add time embedding
        time_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, D)
        x = x + time_emb
        
        # 2. Concatener la condition au début de la séquence (comme un prompt token)
        # On veut prédire la suite, donc on donne le frame 0 clean au réseau
        # Note: Dans la diffusion, on bruite souvent tout, mais ici on fait de la génération conditionnelle.
        # Stratégie simple : On remplace le premier token bruité par le token clean conditionnel
        # Ou mieux : On utilise le conditionnement par Cross-Attention.
        # Pour faire simple ici : On concatène et on maskera la perte sur le 1er token.
        
        token_seq = torch.cat([condition_z0, x[:, 1:, :]], dim=1) # (B, T, D)
        
        # 3. Transformer
        output = self.transformer(token_seq)
        
        # 4. Project back
        return self.final_proj(output)

class LatentDiffusionManager(nn.Module):
    """
    Gère le bruitage (forward diffusion) et l'échantillonnage (reverse diffusion).
    """
    def __init__(self, denoiser, config: Config):
        super().__init__()
        self.denoiser = denoiser
        self.config = config
        self.num_timesteps = 1000
        self.device = config.DEVICE
        
        # Beta schedule (linear)
        beta_start = 0.0001
        beta_end = 0.02
        # On s'assure que tout est sur le bon device dès le début
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        """
        Extrait les coefficients à l'index t pour chaque élément du batch.
        """
        batch_size = t.shape[0]
        # CORRECTION ICI : On garde t sur le même device que a (GPU)
        out = a.gather(-1, t) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def p_loss(self, x_start, condition_z0):
        B, T, D = x_start.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x_start)
        
        # x_noisy
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # prediction
        predicted_noise = self.denoiser(x_noisy, t, condition_z0)
        
        # Loss sur les frames 1 à T (on ne prédit pas le frame 0 qui est donné)
        loss = F.mse_loss(predicted_noise[:, 1:, :], noise[:, 1:, :])
        return loss

    @torch.no_grad()
    def sample(self, condition_z0):
        """ Génère une séquence à partir d'un latent initial """
        B = condition_z0.shape[0]
        shape = (B, self.config.SEQ_LEN, self.config.LATENT_DIM)
        
        img = torch.randn(shape, device=self.device)
        
        # On fixe le frame 0
        img[:, 0, :] = condition_z0[:, 0, :]
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            
            pred_noise = self.denoiser(img, t, condition_z0)
            
            alpha = self.extract(self.alphas, t, shape)
            alpha_cumprod = self.extract(self.alphas_cumprod, t, shape)
            beta = self.extract(self.betas, t, shape)
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
                
            # x_{t-1} step
            term1 = 1 / torch.sqrt(alpha)
            term2 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
            img = term1 * (img - term2 * pred_noise) + torch.sqrt(beta) * noise
            
            # Re-fixer le frame 0 (guidage fort)
            img[:, 0, :] = condition_z0[:, 0, :]
            
        return img
# ============================================================
# 9. TRAINING LOOP - PHASE 2 (LATENT DIFFUSION)
# ============================================================
def train_latent_diffusion(jepa_checkpoint_path):
    config = Config()
    
    # 1. Charger le modèle JEPA (FROZEN) avec le bon nom de classe
    print("Chargement du modèle JEPA Regularized (Frozen)...")
    
    # --- CORRECTION ICI ---
    jepa_model = RegularizedJEPAMTLD(config).to(config.DEVICE)
    
    checkpoint = torch.load(jepa_checkpoint_path, map_location=config.DEVICE)
    jepa_model.load_state_dict(checkpoint['model_state_dict'])
    jepa_model.eval()
    for p in jepa_model.parameters():
        p.requires_grad = False 
        
    # 2. Créer le modèle de Diffusion (LatentDenoiser)
    denoiser = LatentDenoiser(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    
    # On instancie le manager (assure-toi d'avoir appliqué le fix .cpu() donné avant)
    diffusion = LatentDiffusionManager(denoiser, config).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4, weight_decay=0.01)
    dataloader, _ = get_dataloader(config)
    
    print("\n=== Start Phase 2: Latent Diffusion Training ===")
    
    for epoch in range(config.EPOCHS):
        denoiser.train()
        pbar = tqdm(dataloader, desc=f"Diff Epoch {epoch+1}")
        
        loss_history = []
        
        for i, real_seq in enumerate(pbar):
            real_seq = real_seq.to(config.DEVICE)
            
            # A. Extraire les Latents avec encode_to_z (Nouvelle méthode)
            with torch.no_grad():
                # Cette méthode fait Patchify -> VAE Proj -> Sample
                # Elle retourne (B, T, D) prêt pour la diffusion
                z_clean = jepa_model.encode_to_z(real_seq)
            
            # B. Entraîner la Diffusion
            condition_z0 = z_clean[:, 0:1, :] # Frame 0 comme condition
            
            optimizer.zero_grad()
            loss = diffusion.p_loss(z_clean, condition_z0)
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            pbar.set_postfix({"Loss": f"{np.mean(loss_history):.4f}"})
            
        # C. Test de génération
        if (epoch + 1) % 5 == 0:
            print("Sampling generation example...")
            denoiser.eval()
            with torch.no_grad():
                z_cond = z_clean[:1, 0:1, :]
                generated_latents = diffusion.sample(z_cond)
                
                # Décodage via MTLD
                noise_code = sample_noise_code((1, config.SEQ_LEN, config.NOISE_EMB_DIM), config.DEVICE)
                gen_video, _, _, _, _ = jepa_model.decoder(generated_latents, noise_code)
                
                grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"gen_diffusion_epoch_{epoch+1}.png")
                utils.save_image(gen_video[0], grid_path, nrow=4, normalize=True, value_range=(-1, 1))
                print(f"Generated video saved to {grid_path}")
            
            torch.save(denoiser.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f"diffusion_epoch_{epoch+1}.pth"))

# ============================================================
# 10. INFERENCE / GENERATION FINALE
# ============================================================
def generate_anime_scene(jepa_path, diffusion_path, input_media_path, output_filename="gen_result.png"):
    config = Config()
    device = config.DEVICE
    print(f"--- GÉNÉRATION START ---\nInput: {input_media_path}")

    # 1. Chargement JEPA (Regularized)
    print("Chargement JEPA...")
    # --- CORRECTION ICI ---
    jepa_model = RegularizedJEPAMTLD(config).to(device)
    
    jepa_ckpt = torch.load(jepa_path, map_location=device)
    jepa_model.load_state_dict(jepa_ckpt['model_state_dict'])
    jepa_model.eval()

    # 2. Chargement Diffusion
    print("Chargement Diffusion...")
    denoiser = LatentDenoiser(latent_dim=config.LATENT_DIM).to(device)
    diff_ckpt = torch.load(diffusion_path, map_location=device)
    denoiser.load_state_dict(diff_ckpt)
    denoiser.eval()
    
    diffusion = LatentDiffusionManager(denoiser, config).to(device)

    # 3. Input Image
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    try:
        img_pil = Image.open(input_media_path).convert("RGB")
    except Exception as e:
        print(f"Erreur ouverture image: {e}")
        return

    first_frame = transform(img_pil).to(device)
    fake_seq = torch.zeros(1, config.SEQ_LEN, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)
    fake_seq[:, 0] = first_frame
    
    # 4. Encodage vers Z0 (Regularized)
    with torch.no_grad():
        # On utilise encode_to_z pour avoir un Z dans le bon espace
        z_all = jepa_model.encode_to_z(fake_seq)
        z0 = z_all[:, 0:1, :]

    # 5. Diffusion
    print("Diffusion en cours...")
    generated_latents = diffusion.sample(z0)

    # 6. Décodage
    print("Décodage...")
    with torch.no_grad():
        noise_code = sample_noise_code((1, config.SEQ_LEN, config.NOISE_EMB_DIM), device)
        gen_video, _, _, _, _ = jepa_model.decoder(generated_latents, noise_code)

    # 7. Sauvegarde
    grid = utils.make_grid(gen_video[0], nrow=config.SEQ_LEN, normalize=True, value_range=(-1, 1), padding=2)
    to_pil = transforms.ToPILImage()
    res_img = to_pil(grid)
    res_img.save(output_filename)
    print(f"Terminé : {output_filename}")

# ============================================================
# EXEMPLE D'UTILISATION DANS LE MAIN
# ============================================================
if __name__ == "__main__":
    # --- Pour entraîner la diffusion ---
    # train_latent_diffusion("models_jepa_mtld/jepa_mtld_epoch_35.pth")
    train_jepa_mtld()

    # --- Pour générer une vidéo (Test Final) ---
    # generate_anime_scene(
    #     jepa_path="models_jepa_mtld/jepa_mtld_epoch_50.pth",
    #     diffusion_path="models_jepa_mtld/diffusion_epoch_20.pth", 
    #     input_media_path="mon_image_anime.png",
    #     output_filename="resultat_anime.png"
    # )
    #pass