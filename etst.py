# @title ITKF v2.6 : Correction de Robustesse (Lecture Vidéo)

# ==============================================================================
# 0. INSTALLATION DES DÉPENDANCES
# ==============================================================================
!pip install opencv-python-headless torch torchvision tqdm numpy Pillow --quiet
print("Dépendances installées.")

# ==============================================================================
# SECTION DES IMPORTS
# ==============================================================================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import math
import base64
from IPython.display import HTML, display
import sys
from PIL import Image

# Détection du périphérique et gestion de l'environnement Notebook
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    from google.colab import files

print(f"Environnement détecté : {'Google Colab' if IS_COLAB else 'Local'}")
print(f"Périphérique utilisé : {DEVICE}")


# ==============================================================================
# 1. CONFIGURATION DE L'EXPÉRIENCE
# ==============================================================================
# --- MODIFIEZ CES PARAMÈTRES SELON VOS BESOINS ---

# --- Mode d'Exécution ---
MODE = 'TRAIN_AND_GENERATE' 

# --- Chemins des Fichiers ---
MODEL_PATH = "itkf_model_v2.pth"
OUTPUT_PATH = "generated_video_v2.mp4"
VISUALIZATION_PATH = "training_progress"

# --- Paramètres du Modèle ---
LATENT_DIM = 64
HIDDEN_FEATURES = 256
HIDDEN_LAYERS = 8

# --- Paramètres de l'Entraînement ---
EPOCHS = 200
BATCH_SIZE = 65536 
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH = 1000

# --- Paramètres de la Génération ---
RESOLUTION = 384
NUM_FRAMES_TO_GENERATE = 300
GENERATION_SEED = 2025

# --- Paramètres de la Visualisation ---
ENABLE_VISUALIZATION = True
VISUALIZATION_INTERVAL_EPOCHS = 20


# ==============================================================================
# 2. MODULE DE VISUALISATION
# ==============================================================================
class TrainingVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Les images de progression seront sauvegardées dans : '{self.output_dir}'")

    @staticmethod
    def flow_to_hsv_image(flow_tensor):
        if flow_tensor.is_cuda:
            flow_tensor = flow_tensor.cpu()
        dx, dy = flow_tensor[..., 0], flow_tensor[..., 1]
        magnitude, angle = cv2.cartToPolar(dx.numpy(), dy.numpy())
        hsv = np.zeros((flow_tensor.shape[0], flow_tensor.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = magnitude.astype(np.uint8)
        bgr_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr_image

    @torch.no_grad()
    def save_progress(self, epoch, model, data_manager, device, z_vector):
        model.eval()
        frame_idx = data_manager.num_frames // 2
        gt_flow = data_manager.flows_tensor[frame_idx]
        gt_flow_img = self.flow_to_hsv_image(gt_flow)
        H, W = data_manager.H, data_manager.W
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        model_coords = torch.stack([y, x], dim=-1).view(-1, 2).to(device)
        t = torch.full((H * W, 1), fill_value=frame_idx / (data_manager.num_frames - 1), device=device)
        coords_t = torch.cat([t, model_coords], dim=1).unsqueeze(0)
        predicted_flow = model(coords_t, z_vector).squeeze(0).view(H, W, 2)
        predicted_flow_img = self.flow_to_hsv_image(predicted_flow)
        comparison_img = np.hstack((gt_flow_img, predicted_flow_img))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_img, 'Flux Reel', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison_img, 'Flux Predit', (gt_flow_img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison_img, f'Epoch: {epoch}', (10, comparison_img.shape[0] - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        filename = os.path.join(self.output_dir, f"progress_epoch_{epoch:04d}.png")
        cv2.imwrite(filename, comparison_img)
        model.train()


# ==============================================================================
# 3. DÉFINITION DE L'ARCHITECTURE DU MODÈLE (ITKF)
# ==============================================================================
class SineActivation(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class KineticFieldNet(nn.Module):
    def __init__(self, in_features=3, out_features=2, hidden_features=256, 
                 hidden_layers=8, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_projection = nn.Linear(latent_dim, hidden_features)
        self.in_mapping = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_features, hidden_features), SineActivation())
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_features, out_features)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.in_mapping.weight.uniform_(-1 / self.in_mapping.in_features, 1 / self.in_mapping.in_features)
            for module in self.layers:
                layer = module[0]
                bound = math.sqrt(6.0 / layer.in_features) / 30.0
                layer.weight.uniform_(-bound, bound)
            self.output_layer.weight.uniform_(-math.sqrt(6.0 / self.output_layer.in_features) / 30.0,
                                             math.sqrt(6.0 / self.output_layer.in_features) / 30.0)

    def forward(self, coords, z):
        projected_z = self.latent_projection(z).unsqueeze(1).expand(-1, coords.shape[1], -1)
        mapped_coords = torch.sin(30.0 * self.in_mapping(coords))
        x = mapped_coords + projected_z
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


# ==============================================================================
# 4. GESTIONNAIRE DE DONNÉES OPTIMISÉ
# ==============================================================================
class VideoDataManager:
    def __init__(self, video_paths, resolution_hw, device):
        self.resolution_hw = resolution_hw
        self.device = device
        self.keyframes = []
        self.flows_tensor = self._process_videos(video_paths)
        if self.flows_tensor is not None:
            self.num_frames, self.H, self.W, _ = self.flows_tensor.shape
            print(f"Tenseur de flux créé sur le GPU : {self.flows_tensor.shape} (T, H, W, 2)")
        else:
            self.num_frames = 0

    def _process_videos(self, video_paths):
        all_flows = []
        print("Pré-calcul du flux optique pour toutes les vidéos...")
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): 
                print(f"AVERTISSEMENT : Impossible d'ouvrir la vidéo : {video_path}")
                continue
            
            frames = []
            while True:
                # =======================================================================
                # DÉBUT DE LA CORRECTION DÉFINITIVE
                # =======================================================================
                ret, frame = cap.read()
                # Cette double vérification est cruciale. Elle gère les cas où la lecture
                # du flux vidéo s'arrête en retournant un 'frame' vide ou None même si
                # 'ret' est True, ce qui prévient le crash dans calcOpticalFlowFarneback.
                if not ret or frame is None:
                    break
                # =======================================================================
                # FIN DE LA CORRECTION DÉFINITIVE
                # =======================================================================
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dsize = (self.resolution_hw[1], self.resolution_hw[0])
                resized_frame = cv2.resize(frame_rgb, dsize, interpolation=cv2.INTER_AREA)
                pil_image = Image.fromarray(resized_frame)
                frames.append(TF.to_tensor(pil_image))

            cap.release()
            if not frames: continue
            self.keyframes.append(frames[0].to(self.device))
            
            frames_gray = [cv2.cvtColor((f.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for f in frames]
            
            video_flows = []
            for i in tqdm(range(len(frames_gray) - 1), desc=f"Calcul du flux pour {os.path.basename(video_path)}"):
                flow = cv2.calcOpticalFlowFarneback(frames_gray[i], frames_gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                video_flows.append(torch.from_numpy(flow))
            
            if not video_flows: continue
            all_flows.append(torch.stack(video_flows))
        
        if not all_flows:
            print("ERREUR : Aucun flux optique n'a pu être calculé. Vérifiez les vidéos sources.")
            return None

        flows_tensor = torch.cat(all_flows, dim=0).float()
        flows_tensor[..., 0] /= (self.resolution_hw[1] / 2.0)
        flows_tensor[..., 1] /= (self.resolution_hw[0] / 2.0)
        return flows_tensor.to(self.device)

    def get_random_batch(self, batch_size):
        if self.num_frames == 0: return None, None
        t_indices = torch.randint(0, self.num_frames, (batch_size,), device=self.device)
        y_indices = torch.randint(0, self.H, (batch_size,), device=self.device)
        x_indices = torch.randint(0, self.W, (batch_size,), device=self.device)
        target_flows = self.flows_tensor[t_indices, y_indices, x_indices]
        t_coords = t_indices.float() / self.num_frames
        y_coords = (y_indices.float() / (self.H - 1)) * 2 - 1
        x_coords = (x_indices.float() / (self.W - 1)) * 2 - 1
        coords = torch.stack([t_coords, y_coords, x_coords], dim=-1)
        return coords, target_flows


# ==============================================================================
# 5. CLASSE D'ORCHESTRATION MISE À JOUR
# ==============================================================================
class ITKFController:
    def __init__(self, latent_dim, resolution, hidden_features, hidden_layers, device, visualization_dir):
        self.resolution_hw = (resolution, resolution)
        self.device = device
        self.model = KineticFieldNet(latent_dim=latent_dim, hidden_features=hidden_features, hidden_layers=hidden_layers).to(device)
        self.data_manager = None
        self.keyframes = []
        if ENABLE_VISUALIZATION:
            self.visualizer = TrainingVisualizer(visualization_dir)
        print(f"Modèle ITKF initialisé sur le périphérique : {self.device}")

    def train(self, video_paths, epochs, steps_per_epoch, batch_size, lr, visualize_every_n_epochs):
        print("\n--- Démarrage de la phase d'entraînement (Optimisée) ---")
        self.data_manager = VideoDataManager(video_paths, self.resolution_hw, self.device)
        if self.data_manager.num_frames == 0:
            print("🛑 Entraînement annulé car aucune donnée n'a pu être chargée.")
            return
        self.keyframes = self.data_manager.keyframes
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        training_z = torch.randn(1, self.latent_dim).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for step in pbar:
                coords, flows = self.data_manager.get_random_batch(batch_size)
                if coords is None: continue
                optimizer.zero_grad()
                predicted_flows = self.model(coords.unsqueeze(0), training_z).squeeze(0)
                loss = criterion(predicted_flows, flows)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            print(f"Epoch {epoch+1}/{epochs}, Perte Moyenne: {total_loss / steps_per_epoch:.6f}")
            
            if ENABLE_VISUALIZATION and (epoch + 1) % visualize_every_n_epochs == 0:
                print(f"Génération de l'image de progression pour l'epoch {epoch+1}...")
                self.visualizer.save_progress(epoch + 1, self.model, self.data_manager, self.device, training_z)

        print("--- Entraînement terminé ---")

    @torch.no_grad()
    def generate_video(self, output_path, duration_frames, seed, start_frame_idx=0):
        if not self.keyframes: raise RuntimeError("Aucune keyframe disponible.")
        print(f"\n--- Génération de la vidéo ({duration_frames} trames) ---")
        self.model.eval()
        if seed is not None: torch.manual_seed(seed)
        generation_z = torch.randn(1, self.latent_dim).to(self.device)
        H, W = self.resolution_hw
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack([x, y], dim=-1).to(self.device)
        model_coords = grid.view(-1, 2)[:, [1, 0]]
        current_frame = self.keyframes[start_frame_idx].clone().unsqueeze(0)
        for i in tqdm(range(duration_frames), desc="Génération des trames"):
            frame_to_write = (current_frame.squeeze(0).permute(1, 2, 0) * 255.).cpu().numpy().astype(np.uint8)
            video_writer.write(cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR))
            t = torch.full((H * W, 1), fill_value=i / (duration_frames - 1), device=self.device)
            coords_t = torch.cat([t, model_coords], dim=1).unsqueeze(0)
            flow = self.model(coords_t, generation_z).squeeze(0).view(H, W, 2)
            warp_grid = grid + flow[:, :, [1, 0]]
            current_frame = nn.functional.grid_sample(current_frame, warp_grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=False)
        video_writer.release()
        print(f"--- Vidéo sauvegardée à : {output_path} ---")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path); print(f"Modèle sauvegardé à {path}")
    def load_model(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Modèle {path} non trouvé.")
        self.model.load_state_dict(torch.load(path, map_location=self.device)); print(f"Modèle chargé depuis {path}")

# ==============================================================================
# 6. LOGIQUE D'EXÉCUTION PRINCIPALE
# ==============================================================================
def run_experiment():
    video_paths = []
    if IS_COLAB:
        print("Veuillez téléverser vos fichiers vidéo (.mp4).")
        try:
            uploaded = files.upload()
            if not uploaded: 
                print("\n🛑 Aucune vidéo téléversée. Exécution annulée."); return
            video_paths = list(uploaded.keys())
            print(f"\nFichiers reçus : {video_paths}")
        except Exception as e:
            print(f"\n🛑 Une erreur est survenue lors du téléversement : {e}."); return
    else:
        local_video_path = "/kaggle/input/animes-videos/blue_spring_ride_opening.mp4"
        if not os.path.exists(local_video_path):
            print(f"\n🛑 Fichier local non trouvé: '{local_video_path}'."); return
        video_paths = [local_video_path]

    itkf = ITKFController(LATENT_DIM, RESOLUTION, HIDDEN_FEATURES, HIDDEN_LAYERS, DEVICE, VISUALIZATION_PATH)

    if MODE == 'TRAIN_AND_GENERATE':
        itkf.train(video_paths, EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE, LEARNING_RATE, VISUALIZATION_INTERVAL_EPOCHS)
        if not itkf.keyframes:
            print("🛑 Génération annulée."); return
        itkf.save_model(MODEL_PATH)
        itkf.generate_video(OUTPUT_PATH, NUM_FRAMES_TO_GENERATE, GENERATION_SEED)
    
    elif MODE == 'LOAD_AND_GENERATE':
        try:
            itkf.load_model(MODEL_PATH)
            print("Extraction d'une keyframe depuis la vidéo source...")
            temp_data_manager = VideoDataManager(video_paths=[video_paths[0]], resolution_hw=itkf.resolution_hw, device=DEVICE)
            if not temp_data_manager.keyframes:
                 print(f"\n🛑 Impossible d'extraire une keyframe de {video_paths[0]}."); return
            itkf.keyframes = temp_data_manager.keyframes
            itkf.generate_video(OUTPUT_PATH, NUM_FRAMES_TO_GENERATE, GENERATION_SEED)
        except FileNotFoundError as e:
            print(f"\n🛑 Erreur : {e}."); return
        except Exception as e:
            print(f"\n🛑 Une erreur inattendue est survenue : {e}"); return

    if os.path.exists(OUTPUT_PATH):
        print("\n--- Visualisation de la Vidéo Générée ---")
        try:
            with open(OUTPUT_PATH, "rb") as f: 
                video_encoded = base64.b64encode(f.read()).decode('ascii')
            display(HTML(f'<video width="{RESOLUTION}" controls autoplay loop><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>'))
            if IS_COLAB:
                print("\n--- Téléchargement des Résultats ---")
                if os.path.exists(MODEL_PATH): files.download(MODEL_PATH)
                files.download(OUTPUT_PATH)
        except Exception as e:
            print(f"Erreur lors de l'affichage ou du téléchargement de la vidéo : {e}")

# Démarrage de l'expérience
run_experiment()