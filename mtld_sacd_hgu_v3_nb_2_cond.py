"""
MTLD v3.2 - SACD-HGU (Notebook-friendly, Anti-Drift Trajectory) + Conditioning
"""
import os, glob, re, time, warnings, tempfile, shutil
from collections import deque
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
from tqdm import tqdm

try:
    import lpips
except ImportError:
    print("="*80); print("ERROR: LPIPS not found."); print("!pip install lpips"); print("="*80); raise

try:
    import pynvml
except ImportError:
    print("="*80); print("WARN: pynvml not found, GPU monitoring disabled."); print("!pip install nvidia-ml-py"); print("="*80)
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
            print(f"GPUMonitor ready on {pynvml.nvmlDeviceGetName(self.handle)}.")
        except Exception as e:
            print(f"ERROR: pynvml init failed: {e}")
            self.enabled = False

    def check(self):
        if not self.enabled:
            return "continue"
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        if temp > self.config.GPU_TEMP_THRESHOLD_C:
            print(f"\\n! GPU ALERT ! Temp: {temp}C (Thresh: {self.config.GPU_TEMP_THRESHOLD_C}C)")
            return self._handle_alert()
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
            power_percent = (power_usage / power_limit) * 100
            if power_percent > self.config.GPU_POWER_THRESHOLD_PERCENT:
                print(f"\\n! GPU ALERT ! Power: {power_percent:.1f}% (Thresh: {self.config.GPU_POWER_THRESHOLD_PERCENT}%)")
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
            print(f"\\n!! CRITICAL !! {len(self.recent_alerts)} alerts in {self.config.GPU_SHUTDOWN_WINDOW_S}s. Emergency stop.")
            return "shutdown"
        else:
            print(f"Pausing training for {self.config.GPU_PAUSE_DURATION_S}s...")
            time.sleep(self.config.GPU_PAUSE_DURATION_S)
            return "paused"

    def shutdown(self):
        if self.enabled:
            pynvml.nvmlShutdown()


class Config:
    # --- Original hyperparams ---
    DATASET_PATH = "/root/.cache/kagglehub/datasets/jeremgaming099/anima-s-dataset/versions/7/animes-dataset"
    PRELOAD_DATASET_IN_RAM = True
    IMG_SIZE = 256
    IMG_CHANNELS = 3
    TRAINING_SEQUENCE_LENGTH = 16
    LATENT_DIM = 256
    GRU_HIDDEN_DIM = 512
    NOISE_EMB_DIM = 16
    PALETTE_K = 10                   # + richer colors
    HGU_WIDTH = 256
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE_G = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    DEVICE = get_device()
    NUM_WORKERS = 4
    # Loss weights
    LAMBDA_REC_L1 = 15.0
    LAMBDA_REC_LPIPS = 5.0
    LAMBDA_CONSIST = 6.0             # lower to avoid over-smoothing
    LAMBDA_RPC = 6.0
    LAMBDA_EDGE_BASE = 4.0
    LAMBDA_EDGE_BOOST = 6.0          # stronger edges for epochs 10-20
    EDGE_BOOST_START = 10
    EDGE_BOOST_END = 20
    LAMBDA_PALETTE = 1.0
    LAMBDA_TV_ASSIGN = 0.15
    LAMBDA_TEMPORAL_PALETTE = 1.0
    RPC_MARGIN = 0.5
    RPC_PATCH_SIZE = 24              # finer details
    RPC_PROJ_DIM = 128
    # Latent trajectory losses
    LAMBDA_LATENT_TF = 150.0         # teacher-forced rollout
    LAMBDA_LATENT_SS = 150.0         # scheduled sampling rollout
    LAMBDA_LATENT_DELTA = 30.0       # match temporal differences
    LAMBDA_RECOVERY = 20.0           # robustness to large perturbations
    LAMBDA_MMD = 2.0                 # keep predicted latents on-manifold
    # Scheduled sampling
    SCHED_SAMPLING_START = 0.0
    SCHED_SAMPLING_END = 0.6
    SCHED_SAMPLING_EPOCHS = 20
    NOISE_INJ_STD = 0.05
    RECOVERY_NOISE_STD = 0.10
    # IO
    MODEL_SAVE_PATH = "./models_mtld_v3.2/"
    OUTPUT_SAVE_PATH = "./outputs_mtld_v3.2/"
    SAVE_EPOCH_INTERVAL = 2
    MAX_CHECKPOINTS = 3
    RESUME_TRAINING = False
    CHECKPOINT_TO_RESUME = "./models_mtld_v3.2/mtld_v3.2_checkpoint_epoch_10.pth"
    # GPU monitor
    GPU_MONITORING_ENABLED = True
    GPU_TEMP_THRESHOLD_C = 85
    GPU_POWER_THRESHOLD_PERCENT = 95
    GPU_PAUSE_DURATION_S = 60
    GPU_SHUTDOWN_THRESHOLD_COUNT = 5
    GPU_SHUTDOWN_WINDOW_S = 300

    # --- New: Conditioning ---
    # Raw condition components we derive automatically:
    #   [darkness (nightness), motion_intensity, edge_density, saturation_proxy]
    C_RAW_DIM = 4
    C_EMB_DIM = 64                # embedding size used by FiLM
    CFG_DROP_PROB = 0.2           # p(c -> 0) during training (classifier-free guidance)
    CFG_STRENGTH = 2.0            # guidance strength at inference
    USE_PROXY_COND = True         # compute proxies from batch for training
    USE_DECODER_FILM = True       # apply FiLM on decoder features
    USE_GRU_FILM = True           # apply FiLM on GRU hidden
    NO_TELEPORT_HINGE = False     # optional; set True if you want explicit anti-jump penalty
    NO_TELEPORT_MARGIN = 1.5

    # Generation presets (optional; you can override in __main__)
    GEN_COND_MODE = 'auto'        # 'auto' (from priming frame) or 'manual'
    GEN_MANUAL_RAW = [1.0, 0.8, 0.6, 0.5]  # [darkness, motion, edges, saturation]
    GEN_CFG_STRENGTH = 2.0        # override for generation

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
            raise FileNotFoundError(f"Dataset not found: {root_dir}")

        arc_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        total_valid_sequences = 0
        for arc_dir in arc_dirs:
            arc_path = os.path.join(root_dir, arc_dir)
            image_paths = sorted(
                glob.glob(os.path.join(arc_path, "*.png")),
                key=lambda x: int(re.search(r'(\\d+)', os.path.basename(x)).group())
            )
            if len(image_paths) >= self.sequence_length:
                self.sequences.append(image_paths)
                num_seq = len(image_paths) - self.sequence_length + 1
                total_valid_sequences += num_seq
                self.cumulative_lengths.append(total_valid_sequences)

        if not self.sequences:
            raise ValueError("No valid arcs found.")

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


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, img_size: int):
        super().__init__();
        fm = img_size // 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, 4, 2, 1, bias=False), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(1024 * fm * fm, latent_dim)
        )
    def forward(self, x): return self.model(x)


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


class FiLM1dTo2d(nn.Module):
    """Maps a 1D condition embedding to 2D feature-wise affine parameters."""
    def __init__(self, c_dim, feat_channels):
        super().__init__()
        self.to_params = nn.Sequential(
            nn.Linear(c_dim, 2*feat_channels),
        )
    def forward(self, feat, c_emb):
        # feat: (B*S,C,H,W) or (B,C,H,W); c_emb: (B*S,c) or (B,c)
        if c_emb is None: return feat
        if feat.dim()==4 and c_emb.dim()==2:
            B,C,H,W = feat.shape
            gamma, beta = self.to_params(c_emb).chunk(2, dim=-1)  # (B,C)
            gamma = gamma.view(B,C,1,1); beta = beta.view(B,C,1,1)
            return (1+gamma)*feat + beta
        elif feat.dim()==4 and c_emb.dim()==3:
            B,S,C,H,W = feat.shape[0], 1, feat.shape[1], feat.shape[2], feat.shape[3]
            # shouldn't happen here; keep simple
            raise RuntimeError("Unexpected shapes in FiLM1dTo2d.")
        else:
            return feat


class CompositionalAnimeDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, noise_dim: int, img_size: int, K: int, width: int, c_emb_dim: int = 0, use_film: bool = True):
        super().__init__()
        self.img_size = img_size
        self.K = K
        self.c_emb_dim = c_emb_dim
        self.use_film = use_film and (c_emb_dim > 0)
        fm = img_size // 64
        self.fc = nn.Linear(latent_dim, 1024 * fm * fm)
        self.noise_mlp = nn.Sequential(nn.Linear(noise_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                HarmonicGate(out_c),
                nn.ReLU(inplace=True),
            )
        ch = [1024, width, width//2, width//4, width//8, width//16]
        self.up1 = up(1024, ch[0])
        self.up2 = up(ch[0], ch[1])
        self.up3 = up(ch[1], ch[2])
        self.up4 = up(ch[2], ch[3])
        self.up5 = up(ch[3], ch[4])
        self.assign_head = nn.ConvTranspose2d(ch[4], K, 4, 2, 1)
        self.shade_head  = nn.ConvTranspose2d(ch[4], out_channels, 4, 2, 1)
        self.edge_head   = nn.ConvTranspose2d(ch[4], 1, 4, 2, 1)
        self.palette_mlp = nn.Sequential(
            nn.Linear(latent_dim + 64, 256), nn.ReLU(),
            nn.Linear(256, K*3)
        )
        if self.use_film:
            self.film_dec = FiLM1dTo2d(c_emb_dim, ch[4])

    def forward(self, z, noise_code, cond_emb=None):
        # cond_emb: (B,c) or (B,S,c) or None
        seq = (z.dim()==3)
        if seq:
            b,s,d = z.shape
            z_flat = z.view(b*s, d)
            if noise_code.dim()==3:
                n_flat = noise_code.view(b*s, -1)
            else:
                n_flat = noise_code
        else:
            b,d = z.shape
            s = 1
            z_flat = z
            n_flat = noise_code
        base = self.fc(z_flat).view(b*s, 1024, self.img_size//64, self.img_size//64)
        nemb = self.noise_mlp(n_flat).view(b*s, -1)
        x = self.up1(base); x = self.up2(x); x = self.up3(x); x = self.up4(x); x = self.up5(x)

        # Apply FiLM from condition (if provided)
        if self.use_film and cond_emb is not None:
            if cond_emb.dim()==2:
                c_flat = cond_emb
            elif cond_emb.dim()==3:
                # (B,S,C) -> (B*S,C)
                c_flat = cond_emb.reshape(b*s, -1)
            else:
                c_flat = None
            if c_flat is not None:
                x = self.film_dec(x, c_flat)

        assign_logits = self.assign_head(x)
        shading = torch.tanh(self.shade_head(x))
        edge_mask = torch.sigmoid(self.edge_head(x))
        pal = torch.tanh(self.palette_mlp(torch.cat([z_flat, nemb], dim=1))).view(b*s, self.K, 3)
        H=W=self.img_size
        assign = torch.softmax(assign_logits, dim=1)
        assign_ = assign.view(b*s, self.K, H*W)
        pal_ = pal.view(b*s, 3, self.K)
        base_colors = torch.bmm(pal_, assign_).view(b*s, 3, H, W)
        img = torch.clamp(base_colors + shading * edge_mask, -1.0, 1.0)
        def unflat(t): return t.view(b, s, *t.shape[1:])
        return (unflat(img),
                unflat(assign_logits),
                unflat(pal.view(b*s, self.K*3)),
                unflat(shading),
                unflat(edge_mask))


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


class ConditionEncoder(nn.Module):
    """Maps raw condition vector (proxies or tags) to embedding for FiLM/concat."""
    def __init__(self, raw_dim, emb_dim):
        super().__init__()
        if raw_dim <= 0 or emb_dim <= 0:
            self.net = None
            self.emb_dim = 0
        else:
            self.emb_dim = emb_dim
            self.net = nn.Sequential(
                nn.Linear(raw_dim, 128), nn.ReLU(),
                nn.Linear(128, emb_dim)
            )
    def forward(self, c_raw):
        if self.net is None or c_raw is None:
            return None
        return self.net(c_raw)


class ConditionalLatentTrajectoryGenerator(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, c_emb_dim: int = 0, use_film: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_emb_dim = c_emb_dim
        self.use_film = use_film and (c_emb_dim > 0)
        in_dim = latent_dim + (c_emb_dim if c_emb_dim>0 else 0)
        self.z_to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)
        if self.use_film:
            self.film = nn.Linear(c_emb_dim, 2*hidden_dim)

    def _apply_film(self, y, c_emb):
        if not self.use_film or c_emb is None: return y
        gamma, beta = self.film(c_emb).chunk(2, dim=-1)  # (B,H)
        return (1 + gamma)[:,None,:] * y + beta[:,None,:]

    def forward(self, z_start, max_len, cond_emb=None, guidance_strength=0.0):
        """Pure self-feeding rollout. If guidance_strength>0: do CFG (uncond vs cond)."""
        if cond_emb is None or self.c_emb_dim == 0 or guidance_strength <= 0.0:
            # Unconditional or plain conditional without CFG mix.
            h0 = self.z_to_hidden(z_start).unsqueeze(0).repeat(2, 1, 1)
            x_latent = z_start
            outs = []
            B = z_start.size(0)
            for _ in range(max_len):
                if cond_emb is not None and self.c_emb_dim>0:
                    x_in = torch.cat([x_latent, cond_emb], dim=-1).unsqueeze(1)
                else:
                    x_in = x_latent.unsqueeze(1)
                gru_out, h0 = self.gru(x_in, h0)
                if cond_emb is not None and self.c_emb_dim>0:
                    gru_out = self._apply_film(gru_out, cond_emb)
                next_z = self.head(gru_out)
                outs.append(next_z)
                x_latent = next_z.squeeze(1)
            return torch.cat(outs, dim=1)

        # CFG mix
        B = z_start.size(0)
        h0_c = self.z_to_hidden(z_start).unsqueeze(0).repeat(2, 1, 1)
        h0_u = h0_c.clone()
        x_latent_c = z_start
        x_latent_u = z_start
        outs = []
        zero_c = torch.zeros_like(cond_emb)
        s = guidance_strength
        for _ in range(max_len):
            x_in_c = torch.cat([x_latent_c, cond_emb], dim=-1).unsqueeze(1)
            x_in_u = torch.cat([x_latent_u, zero_c], dim=-1).unsqueeze(1)
            gru_out_c, h0_c = self.gru(x_in_c, h0_c)
            gru_out_u, h0_u = self.gru(x_in_u, h0_u)
            gru_out_c = self._apply_film(gru_out_c, cond_emb)
            # uncond path not FiLM'ed
            z_c = self.head(gru_out_c)  # (B,1,D)
            z_u = self.head(gru_out_u)  # (B,1,D)
            z_next = z_u + s * (z_c - z_u)
            outs.append(z_next)
            x_latent_c = z_next.squeeze(1)
            x_latent_u = x_latent_c  # follow same fused latent to maintain stability
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def rollout_teacher(self, z_start_true, z_future_true, cond_emb=None):
        b, T, d = z_future_true.shape
        h0 = self.z_to_hidden(z_start_true).unsqueeze(0).repeat(2, 1, 1)
        outs = []
        prev_true = torch.cat([z_start_true.unsqueeze(1), z_future_true[:, :-1, :]], dim=1)  # (B, T, d)
        for t in range(T):
            x_t = prev_true[:, t, :]
            if cond_emb is not None and self.c_emb_dim>0:
                x_t = torch.cat([x_t, cond_emb], dim=-1)
            x_in = x_t.unsqueeze(1)
            gru_out, h0 = self.gru(x_in, h0)
            if cond_emb is not None and self.c_emb_dim>0:
                gru_out = self._apply_film(gru_out, cond_emb)
            next_z = self.head(gru_out)
            outs.append(next_z)
        return torch.cat(outs, dim=1)

    def rollout_scheduled(self, z_start_true, z_future_true, sched_prob: float, noise_std: float = 0.0, cond_emb=None):
        b, T, d = z_future_true.shape
        h0 = self.z_to_hidden(z_start_true).unsqueeze(0).repeat(2, 1, 1)
        outs = []
        prev_pred = z_start_true
        prev_true = torch.cat([z_start_true.unsqueeze(1), z_future_true[:, :-1, :]], dim=1)
        for t in range(T):
            if sched_prob <= 0.0:
                mix = prev_true[:, t, :]
            elif sched_prob >= 1.0:
                mix = prev_pred
            else:
                mask = (torch.rand(b, 1, device=z_start_true.device) < sched_prob).float()
                mix = mask * prev_pred + (1.0 - mask) * prev_true[:, t, :]
            if noise_std > 0:
                mix = mix + noise_std * torch.randn_like(mix)
            if cond_emb is not None and self.c_emb_dim>0:
                mix = torch.cat([mix, cond_emb], dim=-1)
            x_in = mix.unsqueeze(1)
            gru_out, h0 = self.gru(x_in, h0)
            if cond_emb is not None and self.c_emb_dim>0:
                gru_out = self._apply_film(gru_out, cond_emb)
            next_z = self.head(gru_out)
            outs.append(next_z)
            prev_pred = next_z.squeeze(1)
        return torch.cat(outs, dim=1)


class MTLD(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.IMG_CHANNELS, config.LATENT_DIM, config.IMG_SIZE)
        self.decoder = CompositionalAnimeDecoder(config.LATENT_DIM, config.IMG_CHANNELS, config.NOISE_EMB_DIM,
                                                 config.IMG_SIZE, config.PALETTE_K, config.HGU_WIDTH,
                                                 c_emb_dim=config.C_EMB_DIM, use_film=config.USE_DECODER_FILM)
        self.trajectory_generator = ConditionalLatentTrajectoryGenerator(config.GRU_HIDDEN_DIM, config.LATENT_DIM,
                                                                         c_emb_dim=config.C_EMB_DIM, use_film=config.USE_GRU_FILM)
        self.proj = PerceptualProjector(config.RPC_PROJ_DIM)
        self.cond_encoder = ConditionEncoder(config.C_RAW_DIM, config.C_EMB_DIM)


class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)


def sample_noise_code(shape, device):
    return torch.randn(*shape, device=device)


def degrade(imgs):
    is_seq = (imgs.dim()==5)
    x = imgs if not is_seq else imgs.reshape(-1, *imgs.shape[2:])
    with torch.no_grad():
        x = (x+1)/2
        k = 3
        x_blur = nn.AvgPool2d(k, stride=1, padding=k//2)(x)
        noise = 0.02*torch.randn_like(x)
        x = torch.clamp(x_blur + noise, 0, 1)
        x = x*2 - 1
    return x if not is_seq else x.view_as(imgs)


def consistency_loss(decoder: CompositionalAnimeDecoder, z_seq, noise_dim, device, cond_emb=None):
    if z_seq.dim()==2:
        b = z_seq.size(0)
        n1 = sample_noise_code((b, noise_dim), device)
        n2 = sample_noise_code((b, noise_dim), device)
    else:
        b,s,_ = z_seq.shape
        n1 = sample_noise_code((b,s, noise_dim), device)
        n2 = sample_noise_code((b,s, noise_dim), device)
    y1, *_ = decoder(z_seq, n1, cond_emb=cond_emb)
    y2, *_ = decoder(z_seq, n2, cond_emb=cond_emb)
    return F.l1_loss(y1, y2)


def ranked_patch_contrast_loss(projector: PerceptualProjector, real_imgs, fake_imgs, margin: float, patch_size: int):
    b,s,c,h,w = fake_imgs.shape
    real = real_imgs.reshape(b*s, c, h, w)
    fake = fake_imgs.reshape(b*s, c, h, w)
    deg  = degrade(real)
    ps = int(min(patch_size, h, w)); ps = max(ps, 2)
    def pick_unfold_patches(x):
        B,C,H,W = x.shape
        patches = F.unfold(x, kernel_size=ps, stride=1)
        L = patches.size(-1)
        idx = torch.randint(0, L, (B,), device=x.device)
        idx_expand = idx.view(B, 1, 1).expand(B, C*ps*ps, 1)
        sel = torch.gather(patches, 2, idx_expand).view(B, C, ps, ps)
        return sel
    r = pick_unfold_patches(real); f = pick_unfold_patches(fake); d = pick_unfold_patches(deg)
    er = projector((r+1)/2); ef = projector((f+1)/2); ed = projector((d+1)/2)
    def dist(a,b): return 1 - (a*b).sum(-1)
    loss1 = torch.relu(-(dist(er,er) - dist(er,ef)) + margin).mean()
    loss2 = torch.relu(-(dist(ef,er) - dist(ef,ed)) + margin).mean()
    return 0.5*(loss1+loss2)


def sobel_edges(x):
    x01 = (x+1)/2
    gray = 0.299*x01[:,0]+0.587*x01[:,1]+0.114*x01[:,2]
    gray = gray.unsqueeze(1)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)
    mag = mag / (mag.amax(dim=[2,3], keepdim=True) + 1e-6)
    return mag


def edge_loss(pred_edge_mask, real_imgs):
    b,s,c,h,w = real_imgs.shape
    real = real_imgs.reshape(b*s, c, h, w)
    target = sobel_edges(real).detach()
    pred = pred_edge_mask.reshape(b*s, 1, h, w)
    return F.mse_loss(pred, target)


def tv_loss_assign(assign_logits):
    b,s,k,h,w = assign_logits.shape
    assign = torch.softmax(assign_logits, dim=2)
    dx = assign[:,:,:,:,1:] - assign[:,:,:,:,:-1]
    dy = assign[:,:,:,1:,:] - assign[:,:,:,:-1,:]
    return (dx.abs().mean() + dy.abs().mean())


def palette_chamfer_loss(palette_flat, real_imgs, K, num_samples=2048):
    b,s,_,h,w = real_imgs.shape
    pal = palette_flat.view(b*s, K, 3)
    img = real_imgs.view(b*s, 3, h, w)
    img = (img+1)/2
    pal01 = (pal+1)/2
    N = min(num_samples, h*w)
    idx = torch.randint(0, h*w, (b*s, N), device=img.device)
    img_flat = img.view(b*s, 3, -1).transpose(1,2)
    pix = torch.gather(img_flat, 1, idx.unsqueeze(-1).expand(b*s, N, 3))
    pix2 = pix.unsqueeze(2)
    pal2 = pal01.unsqueeze(1)
    d = (pix2 - pal2).pow(2).sum(-1)
    min_pix = d.min(dim=2).values.mean()
    min_pal = d.min(dim=1).values.mean()
    return 0.5*(min_pix + min_pal)


def temporal_palette_loss(palette_flat):
    pal = palette_flat.view(palette_flat.size(0), palette_flat.size(1), -1)
    if pal.size(1) < 2: return torch.tensor(0.0, device=pal.device)
    diff = pal[:,1:,:] - pal[:,:-1,:]
    return diff.pow(2).mean()


def latent_delta_loss(z_pred, z_true):
    dp = z_pred[:,1:,:] - z_pred[:,:-1,:]
    dt = z_true[:,1:,:] - z_true[:,:-1,:]
    return F.mse_loss(dp, dt)


def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    def pdist(a, b):
        a2 = (a*a).sum(dim=1, keepdim=True)
        b2 = (b*b).sum(dim=1, keepdim=True).transpose(0,1)
        ab = a @ b.t()
        d2 = a2 + b2 - 2*ab
        return torch.clamp(d2, min=0.0)
    K = 0.0
    dxx = pdist(x, x)
    dyy = pdist(y, y)
    dxy = pdist(x, y)
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s**2))
        K += torch.exp(-gamma * dxx).mean() + torch.exp(-gamma * dyy).mean() - 2.0*torch.exp(-gamma * dxy).mean()
    return torch.relu(K)


def save_checkpoint(epoch, model, opt_g, scaler, config: Config):
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    filename = os.path.join(config.MODEL_SAVE_PATH, f"mtld_v3.2_checkpoint_epoch_{epoch}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    try:
        total, used, free = shutil.disk_usage(config.MODEL_SAVE_PATH)
        if free < 300 * 1024 * 1024:
            print("[WARN] Low disk space. Skipping checkpoint to avoid writer errors.")
            return
    except Exception:
        pass
    def _atomic_save(obj, target, legacy=False):
        dirpath = os.path.dirname(target)
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(dir=dirpath, suffix=".tmp", delete=False, mode="wb")
            tmppath = tmp.name
            torch.save(obj, tmppath, _use_new_zipfile_serialization=not legacy)
            tmp.flush(); os.fsync(tmp.fileno()); tmp.close()
            os.replace(tmppath, target)
        except Exception as e:
            try:
                if tmp and not tmp.closed: tmp.close()
            except Exception: pass
            try:
                if tmp is not None: os.remove(tmppath)
            except Exception: pass
            raise e
    try:
        _atomic_save(state, filename, legacy=False)
    except Exception as e1:
        print(f"[WARN] New zip writer failed: {e1}. Retrying with legacy serialization...")
        try:
            _atomic_save(state, filename, legacy=True)
        except Exception as e2:
            print(f"[ERROR] Legacy writer also failed: {e2}.")
            return
    try:
        keep = getattr(config, "MAX_CHECKPOINTS", 3)
        files = [f for f in os.listdir(config.MODEL_SAVE_PATH) if f.startswith("mtld_v3.2_checkpoint_epoch_") and f.endswith(".pth")]
        def _epoch_num(fname):
            import re
            nums = re.findall(r"(\\d+)", fname)
            return int(nums[-1]) if nums else -1
        files.sort(key=_epoch_num)
        if len(files) > keep:
            to_rm = files[:-keep]
            for f in to_rm:
                fp = os.path.join(config.MODEL_SAVE_PATH, f)
                try: os.remove(fp)
                except Exception: pass
    except Exception as e:
        print(f"[WARN] Pruning checkpoints failed: {e}")
    print(f"\\nCheckpoint saved: {filename}")


def schedule_prob(epoch, cfg: Config):
    t = min(max(epoch, 0), cfg.SCHED_SAMPLING_EPOCHS)
    p = cfg.SCHED_SAMPLING_START + (cfg.SCHED_SAMPLING_END - cfg.SCHED_SAMPLING_START) * (t / max(1, cfg.SCHED_SAMPLING_EPOCHS))
    return float(max(0.0, min(1.0, p)))


# === New: simple automatic proxy condition from a sequence batch ===
def compute_proxy_condition(real_seq_imgs, device):
    """
    real_seq_imgs: (B,S,C,H,W) in [-1,1]
    Returns c_raw: (B,4) = [darkness, motion_intensity, edge_density, saturation_proxy] in [0,1]
    """
    with torch.no_grad():
        B,S,C,H,W = real_seq_imgs.shape
        x01 = (real_seq_imgs + 1.0) / 2.0  # [0,1]
        # darkness (1 - luminance mean)
        lum = 0.299*x01[:,:,0] + 0.587*x01[:,:,1] + 0.114*x01[:,:,2]  # (B,S,H,W)
        darkness = 1.0 - lum.mean(dim=(1,2,3))  # (B,)
        darkness = darkness.clamp(0,1)

        # motion intensity: mean absolute diff between consecutive frames (grayscale)
        if S > 1:
            diff = (lum[:,1:,:,:] - lum[:,:-1,:,:]).abs().mean(dim=(1,2,3))  # (B,)
        else:
            diff = torch.zeros(B, device=device)
        motion = diff.clamp(0,1)

        # edge density via sobel magnitude
        real_flat = real_seq_imgs.reshape(B*S, C, H, W)
        edges = sobel_edges(real_flat)  # (B*S,1,H,W) in [0,1]
        edge_density = edges.mean(dim=(1,2,3)).view(B, S).mean(dim=1).clamp(0,1)

        # saturation proxy: per-pixel channel std / (mean+eps)
        mean_rgb = x01.mean(dim=2)  # (B,S,H,W)
        std_rgb = x01.std(dim=2)    # (B,S,H,W)
        sat = (std_rgb / (mean_rgb + 1e-4)).mean(dim=(1,2,3))  # (B,)
        sat = sat.clamp(0,1)

        c_raw = torch.stack([darkness, motion, edge_density, sat], dim=-1)  # (B,4)
        return c_raw


def train_mtld():
    config = Config()
    print(f"Device: {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_SAVE_PATH, exist_ok=True)
    dataloader, total_sequences = get_dataloader(config)
    model = MTLD(config).to(config.DEVICE)
    ema = EMA(model, decay=0.999)

    params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
             list(model.trajectory_generator.parameters()) + list(model.proj.parameters())
    if model.cond_encoder is not None and model.cond_encoder.emb_dim>0:
        params += list(model.cond_encoder.parameters())

    opt_g = optim.Adam(params, lr=config.LEARNING_RATE_G, betas=(config.BETA1, config.BETA2))
    loss_l1 = nn.L1Loss(); loss_mse = nn.MSELoss(); loss_lpips_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    scaler = torch.amp.GradScaler(enabled=(config.DEVICE == "cuda"))
    start_epoch = 0
    if config.RESUME_TRAINING:
        path = config.CHECKPOINT_TO_RESUME
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=config.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_g_state_dict' in checkpoint: opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                ema = EMA(model, decay=0.999)

    gpu_monitor = GPUMonitor(config)
    print("\\nStart training (SACD-HGU, anti-drift + conditioning)...")
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            sched_p = schedule_prob(epoch, config)
            lambda_edge = config.LAMBDA_EDGE_BOOST if (config.EDGE_BOOST_START <= epoch < config.EDGE_BOOST_END) else config.LAMBDA_EDGE_BASE

            for i, real_seq_imgs in enumerate(pbar):
                real_seq_imgs = real_seq_imgs.to(config.DEVICE)
                b, s, c, h, w = real_seq_imgs.shape

                priming_img = real_seq_imgs[:, 0, :, :, :]
                future_imgs = real_seq_imgs[:, 1:, :, :, :]

                # === New: condition embedding (proxy-based) ===
                if config.USE_PROXY_COND and model.cond_encoder is not None and model.cond_encoder.emb_dim>0:
                    c_raw = compute_proxy_condition(real_seq_imgs, config.DEVICE)  # (B,4)
                    c_emb = model.cond_encoder(c_raw)  # (B,C_EMB)
                    # Classifier-free dropout
                    if config.CFG_DROP_PROB > 0:
                        drop_mask = (torch.rand(b, 1, device=config.DEVICE) < config.CFG_DROP_PROB).float()
                        c_emb = (1.0 - drop_mask) * c_emb  # zero some conditions
                else:
                    c_emb = None

                opt_g.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(config.DEVICE=="cuda")):
                    with torch.no_grad():
                        z_start_true = model.encoder(priming_img)                              # (B, D)
                        z_future_true = model.encoder(future_imgs.reshape(-1, c, h, w)).view(b, s - 1, -1)  # (B,T,D)

                    # 1) Teacher-forced rollout (conditioned)
                    z_future_pred_tf = model.trajectory_generator.rollout_teacher(z_start_true, z_future_true, cond_emb=c_emb)

                    # 2) Scheduled-sampling rollout (self-feeding mix + small noise) (conditioned)
                    z_future_pred_ss = model.trajectory_generator.rollout_scheduled(
                        z_start_true, z_future_true, sched_prob=sched_p, noise_std=config.NOISE_INJ_STD, cond_emb=c_emb
                    )

                    # 3) Recovery rollout (heavier perturbations) (conditioned)
                    z_future_pred_rec = model.trajectory_generator.rollout_scheduled(
                        z_start_true, z_future_true, sched_prob=max(0.5, sched_p), noise_std=config.RECOVERY_NOISE_STD, cond_emb=c_emb
                    )

                    # Decode with the robust trajectory (scheduled sampling)
                    z_pred_full_seq = torch.cat([z_start_true.unsqueeze(1), z_future_pred_ss], dim=1)  # (B,S,D)
                    noise_code = sample_noise_code((b, s, config.NOISE_EMB_DIM), config.DEVICE)

                    # Repeat condition over time for decoder FiLM
                    cond_seq = None
                    if c_emb is not None and model.decoder.use_film:
                        cond_seq = c_emb.unsqueeze(1).expand(b, s, -1)

                    fake_seq_imgs, assign_logits, palette_flat, shading, edge_mask = model.decoder(z_pred_full_seq, noise_code, cond_emb=cond_seq)

                    # Losses (unchanged except consistency uses cond)
                    fake_imgs_flat = fake_seq_imgs.view(b*s, c, h, w)
                    real_imgs_flat = real_seq_imgs.view(b*s, c, h, w)

                    loss_rec_l1 = loss_l1(fake_seq_imgs, real_seq_imgs)
                    loss_rec_lpips = loss_lpips_vgg(fake_imgs_flat, real_imgs_flat).mean()
                    loss_consist = consistency_loss(model.decoder, z_pred_full_seq, config.NOISE_EMB_DIM, config.DEVICE, cond_emb=cond_seq)
                    loss_rpc = ranked_patch_contrast_loss(model.proj, real_seq_imgs, fake_seq_imgs, config.RPC_MARGIN, config.RPC_PATCH_SIZE)
                    loss_edges = edge_loss(edge_mask, real_seq_imgs)
                    loss_tv = tv_loss_assign(assign_logits)
                    loss_pal = palette_chamfer_loss(palette_flat, real_seq_imgs, config.PALETTE_K, num_samples=1536)
                    loss_pal_temp = temporal_palette_loss(palette_flat)

                    # Latent trajectory losses
                    loss_latent_tf = F.mse_loss(z_future_pred_tf, z_future_true)
                    loss_latent_ss = F.mse_loss(z_future_pred_ss, z_future_true)
                    loss_latent_d = latent_delta_loss(z_future_pred_ss, z_future_true)
                    loss_recovery = F.mse_loss(z_future_pred_rec, z_future_true)

                    z_ss_flat = z_future_pred_ss.reshape(b*(s-1), -1)
                    z_true_flat = z_future_true.reshape(b*(s-1), -1)
                    loss_mmd = mmd_rbf(z_ss_flat, z_true_flat)

                    loss_g = (
                        config.LAMBDA_REC_L1 * loss_rec_l1 +
                        config.LAMBDA_REC_LPIPS * loss_rec_lpips +
                        config.LAMBDA_CONSIST * loss_consist +
                        config.LAMBDA_RPC * loss_rpc +
                        lambda_edge * loss_edges +
                        config.LAMBDA_TV_ASSIGN * loss_tv +
                        config.LAMBDA_PALETTE * (loss_pal + config.LAMBDA_TEMPORAL_PALETTE * loss_pal_temp) +
                        config.LAMBDA_LATENT_TF * loss_latent_tf +
                        config.LAMBDA_LATENT_SS * loss_latent_ss +
                        config.LAMBDA_LATENT_DELTA * loss_latent_d +
                        config.LAMBDA_RECOVERY * loss_recovery +
                        config.LAMBDA_MMD * loss_mmd
                    )

                    # Optional no-teleport hinge (latent step norm guard)
                    if config.NO_TELEPORT_HINGE:
                        dp = z_pred_full_seq[:,1:,:] - z_pred_full_seq[:,:-1,:]   # (B,S-1,D)
                        with torch.no_grad():
                            dt = z_true_flat.view(b, s-1, -1)  # (B,S-1,D)
                            norm_t = dt.norm(dim=-1).mean()
                        excess = F.relu(dp.norm(dim=-1) - config.NO_TELEPORT_MARGIN * norm_t)
                        loss_nojump = excess.mean()
                        loss_g = loss_g + 0.1 * loss_nojump  # small weight

                scaler.scale(loss_g).backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt_g); scaler.update()
                ema.update(model)

                pbar.set_postfix({
                    "p_ss": f"{sched_p:.2f}",
                    "L_TF": f"{loss_latent_tf.item():.3f}",
                    "L_SS": f"{loss_latent_ss.item():.3f}",
                    "L_Δ": f"{loss_latent_d.item():.3f}",
                    "L_Rec": f"{loss_rec_l1.item():.3f}",
                    "L_RPC": f"{loss_rpc.item():.3f}",
                    "L_Edg": f"{loss_edges.item():.3f}",
                })

                if i > 0 and i % 20 == 0:
                    status = gpu_monitor.check()
                    if status == "shutdown":
                        save_checkpoint(epoch + 1, model, opt_g, scaler, config)
                        gpu_monitor.shutdown()
                        raise SystemExit

            if (epoch + 1) % config.SAVE_EPOCH_INTERVAL == 0:
                save_checkpoint(epoch + 1, model, opt_g, scaler, config)
                # EMA eval preview
                with torch.no_grad():
                    model.eval()
                    backup = {k: v.clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema.shadow, strict=False)

                    try:
                        # quick preview on last batch tensors
                        b_eval, s_eval = b, s
                        noise_eval = sample_noise_code((b_eval, s_eval, config.NOISE_EMB_DIM), config.DEVICE)

                        # use same cond for preview if available
                        cond_seq_eval = None
                        if 'c_emb' in locals() and c_emb is not None and model.decoder.use_film:
                            cond_seq_eval = c_emb.unsqueeze(1).expand(b_eval, s_eval, -1)

                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(config.DEVICE=="cuda")):
                            eval_imgs, _, _, _, _ = model.decoder(z_pred_full_seq, noise_eval, cond_emb=cond_seq_eval)

                        comparison = torch.cat([real_seq_imgs[0].unsqueeze(0), eval_imgs[0].unsqueeze(0)], dim=0)
                        comparison_flat = comparison.permute(1, 0, 2, 3, 4).reshape(-1, c, h, w)
                        grid_path = os.path.join(config.OUTPUT_SAVE_PATH, f"comparison_epoch_{epoch+1}.png")
                        utils.save_image(comparison_flat, grid_path, nrow=s_eval, normalize=True, padding=0)
                    except Exception as e:
                        print(f"[WARN] EMA preview failed: {e}")
                    finally:
                        model.load_state_dict(backup, strict=False)
                        model.train()
    finally:
        gpu_monitor.shutdown()
    print("Training done.")


def build_manual_condition(raw_list, device):
    if raw_list is None or len(raw_list)==0:
        return None
    raw = torch.tensor(raw_list, dtype=torch.float32, device=device).unsqueeze(0)
    return raw.clamp(0,1)


def compute_proxy_condition_from_priming(priming_tensor, device):
    # priming_tensor: (1,3,H,W) in [-1,1]
    x = priming_tensor.unsqueeze(1)  # (1,1,3,H,W)
    return compute_proxy_condition(x, device)  # (1,4)


def generate_sequence(model_path, priming_image_path, num_frames_to_generate, config: Config,
                      manual_cond_raw=None, guidance_strength=None):
    print("--- Generating sequence (SACD-HGU v3.2 + Conditioning) ---")
    if not os.path.exists(model_path): print(f"ERROR: model file not found {model_path}"); return
    if not os.path.exists(priming_image_path): print(f"ERROR: priming image not found {priming_image_path}"); return
    output_dir_frames = os.path.join(config.OUTPUT_SAVE_PATH, "generated_frames_conditional")
    os.makedirs(output_dir_frames, exist_ok=True)
    model = MTLD(config).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    priming_img_pil = Image.open(priming_image_path).convert("RGB")
    priming_tensor = transform(priming_img_pil).unsqueeze(0).to(config.DEVICE)

    # Build condition
    if manual_cond_raw is not None:
        c_raw = build_manual_condition(manual_cond_raw, config.DEVICE)  # (1,4)
    else:
        # auto from priming
        c_raw = compute_proxy_condition_from_priming(priming_tensor, config.DEVICE)  # (1,4)

    c_emb = None
    if model.cond_encoder is not None and model.cond_encoder.emb_dim>0 and c_raw is not None:
        c_emb = model.cond_encoder(c_raw)  # (1,C_EMB)

    if guidance_strength is None:
        guidance_strength = config.GEN_CFG_STRENGTH

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(config.DEVICE=="cuda")):
            z_start = model.encoder(priming_tensor)  # (1,D)
            # CFG-enabled conditional rollout
            z_future = model.trajectory_generator.forward(z_start, max_len=num_frames_to_generate - 1,
                                                          cond_emb=c_emb, guidance_strength=guidance_strength)
            z_full_seq = torch.cat([z_start.unsqueeze(1), z_future], dim=1)
            noise_code = sample_noise_code((1, num_frames_to_generate, config.NOISE_EMB_DIM), config.DEVICE)
            cond_seq = None
            if c_emb is not None and model.decoder.use_film:
                cond_seq = c_emb.unsqueeze(1).expand(1, num_frames_to_generate, -1)
            generated_imgs_seq, *_ = model.decoder(z_full_seq, noise_code, cond_emb=cond_seq)
            generated_imgs_seq = generated_imgs_seq.squeeze(0)

    video_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 24, (config.IMG_SIZE, config.IMG_SIZE))
    pil_images_for_gif = []
    for i, img_tensor in enumerate(generated_imgs_seq):
        img_np = np.clip((img_tensor.permute(1, 2, 0).float() * 0.5 + 0.5).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        img_pil.save(os.path.join(output_dir_frames, f"frame_{i:04d}.png"))
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        pil_images_for_gif.append(img_pil)
    video_writer.release()
    gif_path = os.path.join(config.OUTPUT_SAVE_PATH, f"generated_sequence_conditional.gif")
    pil_images_for_gif[0].save(gif_path, save_all=True, append_images=pil_images_for_gif[1:], duration=int(1000/24), loop=0)
    print(f"Saved: {video_path}")
    print(f"Saved: {gif_path}")


# --- Notebook-friendly runner ---
if __name__ == '__main__':
    cfg = Config()

    # === EDIT HERE IF RUNNING AS A SCRIPT ===
    MODE = 'train'  # 'train' or 'gen'
    MODEL_PATH = os.path.join("best-v3.2/mtld_v3.2_checkpoint_epoch_36.pth")
    PRIMING_PATH = "animes_dataset/my_hero_academia_opening_final/frame_0010.png"
    FRAMES = 200

    # Conditioning at generation:
    #   GEN_COND_MODE: 'auto' -> infer proxies from priming, 'manual' -> use GEN_MANUAL_RAW
    #   GEN_MANUAL_RAW: [darkness, motion, edges, saturation] each in [0,1]
    #   GEN_CFG_STRENGTH: guidance strength; 0.0 disables CFG mix.
    # =======================================

    if MODE == 'train':
        train_mtld()
    else:
        if PRIMING_PATH is None:
            print("ERROR: set PRIMING_PATH to generate.")
        elif not os.path.exists(MODEL_PATH):
            print(f"ERROR: checkpoint not found: {MODEL_PATH}")
        else:
            manual = cfg.GEN_MANUAL_RAW if (cfg.GEN_COND_MODE == 'manual') else None
            generate_sequence(MODEL_PATH, PRIMING_PATH, FRAMES, cfg,
                              manual_cond_raw=manual, guidance_strength=cfg.GEN_CFG_STRENGTH)