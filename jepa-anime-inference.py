import torch
import torch.nn.functional as F
import numpy as np
import imageio
from einops import rearrange
import os

# ==========================================
# 1. OUTILS DE VISUALISATION (RECONSTRUCTION)
# ==========================================
def unpatchify(patch_pixels, t_patch, h_patch, w_patch, t_img, h_img, w_img):
    """
    Transforme la liste de patchs (B, N, 1536) en vidéo (B, C, T, H, W)
    C'est l'inverse exact de 'get_pixel_targets'.
    """
    # 1. On reshape les vecteurs plats en blocs 3D (C, pt, ph, pw)
    # Shapes: b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)
    reconstruction = rearrange(
        patch_pixels, 
        'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
        pt=t_patch, ph=h_patch, pw=w_patch,
        t=t_img//t_patch, h=h_img//h_patch, w=w_img//w_patch
    )
    return reconstruction

def save_comparison_gif(original, masked, reconstructed, filename="resultat_epoch100.gif"):
    """
    Crée un GIF side-by-side : Original | Masqué | Reconstruit
    Entrées: Tenseurs (C, T, H, W) normalisés [-1, 1]
    """
    def process_frames(tensor):
        # Denormalize [-1, 1] -> [0, 255]
        tensor = ((tensor + 1) / 2).clamp(0, 1) * 255
        # (C, T, H, W) -> (T, H, W, C)
        arr = tensor.permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)
        return arr

    orig_frames = process_frames(original)
    mask_frames = process_frames(masked)
    recon_frames = process_frames(reconstructed)
    
    combined_frames = []
    for i in range(len(orig_frames)):
        # Concaténation horizontale
        # On ajoute une ligne noire de séparation (width=2)
        h = orig_frames[i].shape[0]
        sep = np.zeros((h, 2, 3), dtype=np.uint8)
        
        combined = np.concatenate(
            (orig_frames[i], sep, mask_frames[i], sep, recon_frames[i]), 
            axis=1
        )
        combined_frames.append(combined)

    imageio.mimsave(filename, combined_frames, fps=8, loop=0)
    print(f"✨ GIF sauvegardé : {filename}")

# ==========================================
# 2. FONCTION D'EXECUTION PRINCIPALE
# ==========================================
def run_inference():
    # --- Config identique à l'entrainement ---
    CONFIG = {
        "img_size": 128, "num_frames": 16,
        "tubelet_size": (2, 16, 16),
        "embed_dim": 512, "depth": 8, "predictor_depth": 4,
        "num_heads": 8, "mlp_ratio": 4.0, "mask_ratio": 0.60, # 75% masqué !
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 1. Charger le Dataset (Juste une vidéo au hasard)
    # Assurez-vous que le chemin est correct
    dataset = AnimeVideoDataset(folder_path="/kaggle/input/animes-videos/", 
                                num_frames=CONFIG['num_frames'], 
                                img_size=CONFIG['img_size'])
    
    if len(dataset) == 0:
        print("Erreur : Pas de vidéos trouvées.")
        return

    # Prendre une vidéo random
    idx = np.random.randint(0, len(dataset))
    video = dataset[idx].unsqueeze(0).to(CONFIG['device']) # (1, C, T, H, W)

    # 2. Charger le Modèle
    print("Chargement du modèle epoch 100...")
    model = AnimeJEPA(CONFIG).to(CONFIG['device'])
    
    # CHARGEMENT DES POIDS (Gestion safe cpu/gpu)
    checkpoint_path = "checkpoints/jepa_anime_100.pt" # Vérifiez le nom exact
    if not os.path.exists(checkpoint_path):
        # Fallback si l'epoch 50 n'existe pas encore, essayer le dernier
        checkpoints = sorted(glob.glob("checkpoints/*.pt"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"Epoch 50 non trouvé, chargement de : {checkpoint_path}")
        else:
            print("Aucun checkpoint trouvé ! Avez-vous entraîné ?")
            return

    state_dict = torch.load(checkpoint_path, map_location=CONFIG['device'])
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Inférence (Forward Pass)
    print("Génération en cours...")
    with torch.no_grad():
        # Le modèle retourne : latents, targets, pixels_reconstruits, indices_masque
        _, _, recon_patch_pixels, mask_indices = model(video)

    # 4. Post-Processing pour visualisation
    
    # A. Reconstruire la vidéo complète depuis les patchs sortis du décodeur
    recon_video = unpatchify(
        recon_patch_pixels, 
        CONFIG['tubelet_size'][0], CONFIG['tubelet_size'][1], CONFIG['tubelet_size'][2],
        CONFIG['num_frames'], CONFIG['img_size'], CONFIG['img_size']
    )

    # B. Créer la vidéo "Masquée" pour montrer ce que l'IA n'a PAS vu
    # On crée un masque binaire pixel-wise
    B, N, _ = recon_patch_pixels.shape
    # Masque 1 = Caché, 0 = Visible
    mask_flat = torch.zeros((B, N, 1536), device=video.device)
    batch_range = torch.arange(B, device=video.device)[:, None]
    mask_flat[batch_range, mask_indices] = 1 # On marque les patchs masqués
    
    # On dé-patchify le masque pour l'avoir en format vidéo
    mask_video_map = unpatchify(
        mask_flat, 
        CONFIG['tubelet_size'][0], CONFIG['tubelet_size'][1], CONFIG['tubelet_size'][2],
        CONFIG['num_frames'], CONFIG['img_size'], CONFIG['img_size']
    )
    
    # On applique du gris (ou noir) là où c'était masqué
    masked_input_video = video.clone()
    # Si le masque vaut 1, on met l'image à -1 (noir/gris sombre en normalisation tanh)
    masked_input_video[mask_video_map > 0.5] = -0.8 

    # 5. Sauvegarde
    save_comparison_gif(video[0], masked_input_video[0], recon_video[0])
    
    # 6. (Bonus) Mesure de la MSE simple pour info
    mse = F.mse_loss(recon_video, video)
    print(f"MSE de reconstruction : {mse.item():.6f}")

if __name__ == "__main__":
    run_inference()