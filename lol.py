# ==============================================================================
#  Latent Field Morphogenesis (LFM) - A Novel Generative AI Model
# ==============================================================================
#
#  Description:
#  This script implements the LFM architecture, a generative model inspired
#  by biological morphogenesis and continuous field physics. It avoids standard
#  GAN or Diffusion paradigms, focusing on stable training and structured generation.
#
#  Key Components:
#  1. Spatial Variational Encoder: Encodes an image into a spatial latent field.
#  2. Continuous Normalizing Flow (CNF): Models a complex prior distribution
#     over the latent field for a rich and structured latent space.
#  3. Parameterized Neural Cellular Automata (p-NCA): "Grows" an image from a
#     seed, guided by the latent field, ensuring local and global coherence.
#
#  To run this code:
#  1. Set up a Python virtual environment.
#  2. Run: pip install torch torchvision numpy matplotlib lpips torchdiffeq
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as F
from tqdm import tqdm
import warnings

# --- Installation Check & Imports ---
try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    print("torchdiffeq not found. Please run: pip install torchdiffeq")
    exit()

try:
    import lpips
except ImportError:
    print("lpips not found. Please run: pip install lpips")
    exit()

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
IMG_CHANNELS = 3
LATENT_FIELD_SIZE = 16  # h, w of the latent field
LATENT_DIM = 32         # d of the latent field
NCA_STATE_CHANNELS = 16 # RGB channels + hidden channels for NCA state
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 100            # Increase for production training
BETA_KL = 0.1           # Initial KL weight, can be annealed
NCA_STEPS_MIN = 70      # Min steps for NCA growth
NCA_STEPS_MAX = 100     # Max steps for NCA growth

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
#  COMPONENT 1: SPATIAL VARIATIONAL ENCODER
#  Encodes an image x into a distribution over latent fields Z.
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return F.relu(x + self.conv_block(x))

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.Conv2d(256, 2 * latent_dim, kernel_size=1) # Output mu and logvar
        )

    def forward(self, x):
        params = self.net(x)
        mu, logvar = torch.chunk(params, 2, dim=1)
        return mu, logvar

# ==============================================================================
#  COMPONENT 2: CONTINUOUS NORMALIZING FLOW (CNF)
#  Models the prior p(Z) by transforming a simple distribution via an ODE.
# ==============================================================================
class ODEFunc(nn.Module):
    """Defines the dynamics d(z)/dt = f(z, t) for the CNF."""
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t, z):
        # Reshape for processing: [B, D, h, w] -> [B*h*w, D]
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        dz_dt_flat = self.net(z_flat)
        # Reshape back to field format
        return dz_dt_flat.reshape(B, h, w, D).permute(0, 3, 1, 2)

class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, latent_dim, solver='dopri5'):
        super().__init__()
        self.ode_func = ODEFunc(latent_dim)
        self.solver = solver
        self.integration_time = torch.tensor([0, 1]).float()
        # The base distribution p_u(z(1))
        self.base_dist = torch.distributions.Normal(
            loc=torch.zeros(latent_dim, device=DEVICE),
            scale=torch.ones(latent_dim, device=DEVICE)
        )

    def forward(self, z0, reverse=False):
        t = self.integration_time.to(z0.device)
        if reverse:
            t = t.flip(0)
        
        # odeint_adjoint calculates both the transformed variable and the log determinant
        z_t, log_p_diff = odeint(
            self.ode_func, z0, t, method=self.solver, atol=1e-5, rtol=1e-5
        )
        z1, delta_log_p = z_t[-1], log_p_diff[-1]
        return z1, delta_log_p

    def log_prob(self, z0):
        z1, delta_log_p = self.forward(z0, reverse=False)
        # log p(z0) = log p_u(z1) - ∫ Tr(df/dz) dt
        # The base log_prob is summed over the latent dimension D
        log_p_u = self.base_dist.log_prob(z1.permute(0, 2, 3, 1)).sum(-1)
        return (log_p_u - delta_log_p).mean() # Mean over batch and spatial dims

    def sample(self, num_samples, field_size):
        z1 = self.base_dist.sample((num_samples, field_size, field_size)).permute(0, 3, 1, 2)
        z0, _ = self.forward(z1, reverse=True)
        return z0

# ==============================================================================
#  COMPONENT 3: PARAMETERIZED NEURAL CELLULAR AUTOMATA (p-NCA) GENERATOR
#  Grows an image from a seed, guided by the upsampled latent field.
# ==============================================================================
class Perception(nn.Module):
    """Applies fixed Sobel and Laplacian filters to perceive local state."""
    def __init__(self, state_channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        laplacian = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 8.0
        
        filters = torch.stack([sobel_x, sobel_y, laplacian]).unsqueeze(1)
        self.kernel = filters.repeat(state_channels, 1, 1, 1)
        self.conv = nn.Conv2d(state_channels, state_channels * 3, kernel_size=3, padding=1, bias=False, groups=state_channels)
        self.conv.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        return self.conv(x)

class UpdateNet(nn.Module):
    """The core neural network of the NCA, computes the state update."""
    def __init__(self, state_channels, latent_dim, hidden_channels=128):
        super().__init__()
        # Input: self state (C) + perceived state (3*C) + latent guidance (D)
        input_dim = state_channels * 4 + latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=True)
        )
        # Initialize last layer weights and bias to zero for stable start
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, x, z_field):
        return self.net(torch.cat([x, z_field], dim=1))

class GeneratorNCA(nn.Module):
    """Orchestrates the NCA growth process."""
    def __init__(self, state_channels, latent_dim):
        super().__init__()
        self.state_channels = state_channels
        self.perception = Perception(state_channels)
        self.update_net = UpdateNet(state_channels, latent_dim)

    def forward(self, z_field, target_size, steps):
        B, _, h, w = z_field.shape
        z_upsampled = F.resize(z_field, size=target_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        
        # Initialize grid with a central "living" seed
        state = torch.zeros(B, self.state_channels, *target_size, device=z_field.device)
        state[:, 3:, target_size[0]//2, target_size[1]//2] = 1.0 # Hidden channels are "alive"

        for _ in range(steps):
            perceived = self.perception(state)
            update_input = torch.cat([state, perceived], dim=1)
            ds = self.update_net(update_input, z_upsampled)
            
            # Stochastic update: only update a random half of cells
            mask = (torch.rand_like(state[:, :1, :, :]) > 0.5).float()
            state = state + ds * mask
            
            # Alive mask: cell dies if its alpha channel (e.g., channel 3) is <= 0.1
            alive_mask = (F.max_pool2d(state[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1).float()
            state = state * alive_mask

        return torch.sigmoid(state[:, :IMG_CHANNELS, :, :]) # Return RGB channels, squashed to [0,1]

# ==============================================================================
#  TOP-LEVEL LFM MODEL
#  Integrates all components for end-to-end training.
# ==============================================================================
class LFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpatialEncoder(IMG_CHANNELS, LATENT_DIM)
        self.cnf = ContinuousNormalizingFlow(LATENT_DIM)
        self.generator = GeneratorNCA(NCA_STATE_CHANNELS, LATENT_DIM)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z_q = self.reparameterize(mu, logvar)
        
        # Calculate KL divergence
        # log q(z|x) - log p(z)
        # log q(z|x) is the log prob of a Gaussian
        log_q_z_given_x = -0.5 * (logvar + (z_q - mu).pow(2) / logvar.exp()).sum(dim=(1, 2, 3))
        # log p(z) is calculated by the CNF
        log_p_z = self.cnf.log_prob(z_q) * z_q.numel() / z_q.shape[0] # Scale to match log_q
        
        kl_div = (log_q_z_given_x - log_p_z).mean()
        
        # Generate image
        nca_steps = torch.randint(NCA_STEPS_MIN, NCA_STEPS_MAX, (1,)).item()
        x_hat = self.generator(z_q, (IMG_SIZE, IMG_SIZE), nca_steps)
        
        return x_hat, kl_div

# ==============================================================================
#  TRAINING LOOP
# ==============================================================================
def train_lfm():
    print(f"Starting training on device: {DEVICE}")

    # Create a synthetic dataset (replace with your own DataLoader)
    print("Creating synthetic dataset...")
    # Simple dataset with colored shapes for structured learning
    data = torch.zeros(512, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    for i in range(data.shape[0]):
        c = torch.rand(3)
        x, y, r = torch.randint(10, 54, (3,))
        xx, yy = torch.meshgrid(torch.arange(IMG_SIZE), torch.arange(IMG_SIZE), indexing='ij')
        mask = ((xx - x)**2 + (yy - y)**2) < r**2
        data[i, 0, mask] = c[0]
        data[i, 1, mask] = c[1]
        data[i, 2, mask] = c[2]
    
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = LFM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Perceptual loss for higher quality reconstruction
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    l1_loss_fn = nn.L1Loss()
    
    print("Training started...")
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for (x,) in progress_bar:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            
            x_hat, kl_div = model(x)
            
            # Reconstruction Loss (L1 + Perceptual)
            l1_loss = l1_loss_fn(x_hat, x)
            perceptual_loss = lpips_loss_fn(x_hat, x).mean()
            recons_loss = l1_loss + 0.5 * perceptual_loss
            
            loss = recons_loss + BETA_KL * kl_div
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "Recons": f"{recons_loss.item():.4f}",
                "KL": f"{kl_div.item():.4f}"
            })

    print("--- Training Finished ---")
    
    # --- Generation and Visualization ---
    print("Generating sample images from the trained model...")
    model.eval()
    with torch.no_grad():
        # Sample from the learned prior using the CNF
        z_samples = model.cnf.sample(num_samples=4, field_size=LATENT_FIELD_SIZE)
        generated_images = model.generator(z_samples, (IMG_SIZE, IMG_SIZE), steps=NCA_STEPS_MAX)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, img in enumerate(generated_images):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img_np)
            axes[i].set_title(f"Sample {i+1}")
            axes[i].axis('off')
        
        fig.suptitle("Images Generated from LFM Model's Learned Prior", fontsize=16)
        plt.tight_layout()
        plt.savefig("lfm_generated_samples.png")
        print("Generated images saved to 'lfm_generated_samples.png'")
        plt.show()

if __name__ == '__main__':
    train_lfm()