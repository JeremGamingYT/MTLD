"""
AQDN - Adaptive Quantum Diffusion Network
Architecture innovante pour la génération d'images prédictives avec anti-overfitting

Principe:
1. Quantum Feature Extraction (QFE) - Extraction multi-échelle
2. Stochastic Latent Diffusion (SLD) - Génération progressive
3. Diversity-Aware Regularization (DAR) - Anti-mémorisation
4. Adaptive Prediction Head (APH) - Prédiction contextualisée
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import json

# ============================================================================
# 1. QUANTUM FEATURE EXTRACTOR (QFE)
# ============================================================================

class QuantumConvBlock(layers.Layer):
    """Bloc de convolution quantique avec branchements multi-échelles"""
    
    def __init__(self, filters, name_prefix):
        super().__init__()
        # Trois branches à différentes échelles
        self.branch1 = layers.Conv2D(filters//3, 1, padding='same', name=f'{name_prefix}_b1')
        self.branch2 = layers.Conv2D(filters//3, 3, padding='same', name=f'{name_prefix}_b2')
        self.branch3 = layers.Conv2D(filters//3, 5, padding='same', name=f'{name_prefix}_b3')
        
        self.norm = layers.LayerNormalization()
        self.activation = layers.LeakyReLU(0.2)
        
    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Fusion quantique (pondération dynamique)
        concat = tf.concat([b1, b2, b3], axis=-1)
        out = self.norm(concat)
        return self.activation(out)


class QuantumFeatureExtractor(keras.Model):
    """Extracteur de features avec architecture quantique"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Architecture pyramidale
        self.qconv1 = QuantumConvBlock(64, 'qconv1')
        self.pool1 = layers.MaxPooling2D(2)
        
        self.qconv2 = QuantumConvBlock(128, 'qconv2')
        self.pool2 = layers.MaxPooling2D(2)
        
        self.qconv3 = QuantumConvBlock(256, 'qconv3')
        self.pool3 = layers.MaxPooling2D(2)
        
        self.qconv4 = QuantumConvBlock(512, 'qconv4')
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Projection vers l'espace latent
        self.mu = layers.Dense(latent_dim, name='mu')
        self.logvar = layers.Dense(latent_dim, name='logvar')
        
    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        # Forward pass pyramidal
        h1 = self.pool1(self.qconv1(x, training=training))
        h2 = self.pool2(self.qconv2(h1, training=training))
        h3 = self.pool3(self.qconv3(h2, training=training))
        h4 = self.global_pool(self.qconv4(h3, training=training))
        
        # VAE-style encoding pour diversité
        mu = self.mu(h4)
        logvar = self.logvar(h4)
        
        # Reparameterization trick
        if training:
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
        else:
            z = mu
            
        return z, mu, logvar, [h1, h2, h3]  # Features multi-échelles


# ============================================================================
# 2. STOCHASTIC LATENT DIFFUSION (SLD)
# ============================================================================

class DiffusionTransformerBlock(layers.Layer):
    """Transformer block pour le processus de diffusion"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads, dim // num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(dim * 4, activation='gelu'),
            layers.Dense(dim)
        ])
        
    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        # Self-attention avec résiduel
        attn_out = self.attention(x, x, training=training)
        x = self.norm1(x + attn_out)
        
        # Feed-forward avec résiduel
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class StochasticLatentDiffusion(keras.Model):
    """Modèle de diffusion dans l'espace latent avec prédiction de bruit"""
    
    def __init__(self, latent_dim=256, num_steps=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        # Embedding temporel pour les étapes de diffusion
        self.time_embed = keras.Sequential([
            layers.Dense(latent_dim),
            layers.LeakyReLU(0.2),
            layers.Dense(latent_dim)
        ])
        
        # Transformer blocks pour prédire le bruit
        self.transformer_blocks = [
            DiffusionTransformerBlock(latent_dim, num_heads=8)
            for _ in range(4)
        ]
        
        # Prédicteur de bruit
        self.noise_pred = keras.Sequential([
            layers.Dense(latent_dim * 2, activation='gelu'),
            layers.Dense(latent_dim)
        ])
        
    def get_time_embedding(self, t, batch_size):
        """Crée l'embedding temporel pour l'étape t"""
        t_embed = tf.cast(t, tf.float32) / self.num_steps
        t_embed = tf.reshape(t_embed, [-1, 1])
        t_embed = tf.tile(t_embed, [1, self.latent_dim])
        
        # Sinusoidal encoding
        freqs = tf.range(0, self.latent_dim, 2, dtype=tf.float32)
        freqs = 2 * np.pi * freqs / self.latent_dim
        
        t_sin = tf.sin(t_embed[:, ::2] * freqs)
        t_cos = tf.cos(t_embed[:, 1::2] * freqs)
        
        t_embed = tf.stack([t_sin, t_cos], axis=2)
        t_embed = tf.reshape(t_embed, [-1, self.latent_dim])
        
        return self.time_embed(t_embed)
    
    def add_noise(self, z, t):
        """Ajoute du bruit gaussien selon le schedule de diffusion"""
        noise = tf.random.normal(tf.shape(z))
        alpha = 1.0 - t / self.num_steps
        return tf.sqrt(alpha) * z + tf.sqrt(1 - alpha) * noise, noise
    
    def call(self, z, t=None, training=None):
        """
        Forward pass du modèle de diffusion
        z: latent code
        t: timestep (si None, on génère)
        """
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        batch_size = tf.shape(z)[0]
        
        if training:
            # Mode entraînement: prédire le bruit à partir de z bruité
            if t is None:
                t = tf.random.uniform([batch_size], 0, self.num_steps, dtype=tf.int32)
            
            z_noisy, noise_true = self.add_noise(z, tf.cast(t, tf.float32))
            
            # Conditionner sur le timestep
            t_embed = self.get_time_embedding(t, batch_size)
            x = z_noisy + t_embed
            
            # Passer à travers les transformers
            x = tf.expand_dims(x, 1)  # [B, 1, D]
            for block in self.transformer_blocks:
                x = block(x, training=training)
            x = tf.squeeze(x, 1)
            
            # Prédire le bruit
            noise_pred = self.noise_pred(x)
            
            return noise_pred, noise_true
        else:
            # Mode génération: débruitage progressif
            z_t = tf.random.normal(tf.shape(z))
            
            for t in reversed(range(self.num_steps)):
                t_batch = tf.ones([batch_size], dtype=tf.int32) * t
                t_embed = self.get_time_embedding(t_batch, batch_size)
                
                x = z_t + t_embed
                x = tf.expand_dims(x, 1)
                for block in self.transformer_blocks:
                    x = block(x, training=False)
                x = tf.squeeze(x, 1)
                
                noise_pred = self.noise_pred(x)
                
                # Débruitage
                alpha = 1.0 - t / self.num_steps
                alpha_next = 1.0 - (t - 1) / self.num_steps if t > 0 else 1.0
                
                z_t = (z_t - tf.sqrt(1 - alpha) * noise_pred) / tf.sqrt(alpha)
                if t > 0:
                    z_t = tf.sqrt(alpha_next) * z_t + tf.sqrt(1 - alpha_next) * tf.random.normal(tf.shape(z_t))
            
            return z_t


# ============================================================================
# 3. DIVERSITY-AWARE REGULARIZATION (DAR)
# ============================================================================

class DiversityRegularizer:
    """Régularisation pour éviter la mémorisation du dataset"""
    
    def __init__(self, memory_size=1000):
        self.memory_size = memory_size
        self.memory_bank = []
        
    def compute_diversity_loss(self, z_batch, generated_batch):
        """
        Calcule une perte qui encourage la diversité
        - Pénalise les features trop similaires au memory bank
        - Encourage la variance intra-batch
        """
        # Loss 1: Distance au memory bank (éviter reproduction)
        memory_loss = 0.0
        if len(self.memory_bank) > 0:
            memory_tensor = tf.concat(self.memory_bank[-50:], axis=0)
            distances = tf.reduce_mean(
                tf.square(tf.expand_dims(z_batch, 1) - tf.expand_dims(memory_tensor, 0)),
                axis=-1
            )
            min_distances = tf.reduce_min(distances, axis=1)
            memory_loss = -tf.reduce_mean(min_distances)  # Négatif = pénalise similarité
        
        # Loss 2: Variance intra-batch (encourage diversité)
        mean = tf.reduce_mean(generated_batch, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.square(generated_batch - mean))
        diversity_loss = -tf.math.log(variance + 1e-6)  # Encourage variance élevée
        
        # Loss 3: Regularisation spectrale (évite patterns répétitifs)
        fft = tf.signal.fft2d(tf.cast(generated_batch, tf.complex64))
        spectrum = tf.abs(fft)
        spectral_entropy = -tf.reduce_sum(spectrum * tf.math.log(spectrum + 1e-6))
        spectral_loss = -spectral_entropy  # Encourage haute entropie
        
        total_loss = 0.1 * memory_loss + 0.05 * diversity_loss + 0.01 * spectral_loss
        
        return total_loss
    
    def update_memory(self, z_batch):
        """Met à jour la banque de mémoire"""
        self.memory_bank.append(z_batch)
        if len(self.memory_bank) > self.memory_size:
            self.memory_bank.pop(0)


# ============================================================================
# 4. DECODER & IMAGE GENERATOR
# ============================================================================

class AdaptiveDecoder(keras.Model):
    """Décodeur adaptatif avec skip connections multi-échelles"""
    
    def __init__(self, img_size=64, img_channels=3):
        super().__init__()
        self.img_size = img_size
        
        # Projection initiale
        self.project = layers.Dense(8 * 8 * 512)
        self.reshape = layers.Reshape((8, 8, 512))
        
        # Upsampling progressif avec skip connections
        self.up1 = layers.Conv2DTranspose(256, 4, 2, 'same')
        self.norm1 = layers.LayerNormalization()
        self.act1 = layers.LeakyReLU(0.2)
        
        self.up2 = layers.Conv2DTranspose(128, 4, 2, 'same')
        self.norm2 = layers.LayerNormalization()
        self.act2 = layers.LeakyReLU(0.2)
        
        self.up3 = layers.Conv2DTranspose(64, 4, 2, 'same')
        self.norm3 = layers.LayerNormalization()
        self.act3 = layers.LeakyReLU(0.2)
        
        # Sortie finale
        self.output_conv = layers.Conv2D(img_channels, 3, padding='same', activation='tanh')
        
    def call(self, z, skip_features=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        x = self.project(z)
        x = self.reshape(x)
        
        # Upsampling avec normalisation
        x = self.act1(self.norm1(self.up1(x)))
        if skip_features and len(skip_features) > 2:
            x = x + tf.image.resize(skip_features[2], tf.shape(x)[1:3])
            
        x = self.act2(self.norm2(self.up2(x)))
        if skip_features and len(skip_features) > 1:
            x = x + tf.image.resize(skip_features[1], tf.shape(x)[1:3])
            
        x = self.act3(self.norm3(self.up3(x)))
        if skip_features and len(skip_features) > 0:
            x = x + tf.image.resize(skip_features[0], tf.shape(x)[1:3])
        
        # Sortie
        output = self.output_conv(x)
        
        return output


# ============================================================================
# 5. MODÈLE COMPLET AQDN
# ============================================================================

class AQDN(keras.Model):
    """
    Adaptive Quantum Diffusion Network
    Architecture complète pour génération d'images prédictives
    """
    
    def __init__(self, img_size=64, img_channels=3, latent_dim=256, diffusion_steps=10):
        super().__init__()
        
        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        
        # Composants du modèle
        self.encoder = QuantumFeatureExtractor(latent_dim)
        self.diffusion = StochasticLatentDiffusion(latent_dim, diffusion_steps)
        self.decoder = AdaptiveDecoder(img_size, img_channels)
        self.diversity_reg = DiversityRegularizer()
        
        # Métriques
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.diffusion_loss_tracker = keras.metrics.Mean(name="diffusion_loss")
        self.diversity_loss_tracker = keras.metrics.Mean(name="diversity_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.diffusion_loss_tracker,
            self.diversity_loss_tracker
        ]
    
    def call(self, x, training=None):
        """Forward pass complet"""
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        # Encoding
        z, mu, logvar, skip_features = self.encoder(x, training=training)
        
        if training:
            # Diffusion loss
            noise_pred, noise_true = self.diffusion(z, training=True)
            
            # Decoding
            recon = self.decoder(z, skip_features, training=training)
            
            return recon, mu, logvar, noise_pred, noise_true, z
        else:
            # Génération: diffusion puis décodage
            z_refined = self.diffusion(z, training=False)
            generated = self.decoder(z_refined, skip_features, training=False)
            
            return generated
    
    def train_step(self, data):
        """Step d'entraînement personnalisé"""
        x = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            recon, mu, logvar, noise_pred, noise_true, z = self(x, training=True)
            
            # Loss 1: Reconstruction
            recon_loss = tf.reduce_mean(tf.square(x - recon))
            
            # Loss 2: KL divergence (VAE regularization)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + logvar - tf.square(mu) - tf.exp(logvar)
            )
            
            # Loss 3: Diffusion (prédiction de bruit)
            diffusion_loss = tf.reduce_mean(tf.square(noise_pred - noise_true))
            
            # Loss 4: Diversity regularization
            diversity_loss = self.diversity_reg.compute_diversity_loss(z, recon)
            
            # Loss totale combinée
            total_loss = (
                recon_loss + 
                0.001 * kl_loss + 
                0.5 * diffusion_loss + 
                0.1 * diversity_loss
            )
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update memory bank
        self.diversity_reg.update_memory(z)
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.diffusion_loss_tracker.update_state(diffusion_loss)
        self.diversity_loss_tracker.update_state(diversity_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result(),
            "diversity_loss": self.diversity_loss_tracker.result()
        }
    
    def test_step(self, data):
        """Step de test"""
        x = data
        
        # Forward pass en mode training pour obtenir toutes les sorties
        recon, mu, logvar, noise_pred, noise_true, z = self(x, training=True)
        
        # Calcul des losses
        recon_loss = tf.reduce_mean(tf.square(x - recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        diffusion_loss = tf.reduce_mean(tf.square(noise_pred - noise_true))
        
        total_loss = recon_loss + 0.001 * kl_loss + 0.5 * diffusion_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.diffusion_loss_tracker.update_state(diffusion_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result()
        }


# ============================================================================
# 6. SYSTÈME D'ENTRAÎNEMENT ET UTILITAIRES
# ============================================================================

class AQDNTrainer:
    """Gestionnaire d'entraînement pour AQDN"""
    
    def __init__(self, model: AQDN, learning_rate=1e-4):
        self.model = model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate))
        self.history = {
            'loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'diffusion_loss': [],
            'diversity_loss': []
        }
        
    def train(self, train_data, epochs=50, batch_size=32, validation_data=None):
        """
        Entraîne le modèle AQDN
        
        Args:
            train_data: Dataset d'entraînement (numpy array ou tf.data.Dataset)
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            validation_data: Dataset de validation (optionnel)
        """
        # Préparation du dataset
        if isinstance(train_data, np.ndarray):
            train_ds = tf.data.Dataset.from_tensor_slices(train_data)
            train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            train_ds = train_data
            
        if validation_data is not None and isinstance(validation_data, np.ndarray):
            val_ds = tf.data.Dataset.from_tensor_slices(validation_data)
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_ds = validation_data
        
        print("🚀 Début de l'entraînement AQDN")
        print(f"Architecture: {self.model.latent_dim}D latent, {self.model.diffusion.num_steps} diffusion steps")
        print("=" * 80)
        
        for epoch in range(epochs):
            print(f"\nÉpoque {epoch + 1}/{epochs}")
            
            # Entraînement
            epoch_metrics = {'loss': [], 'recon_loss': [], 'kl_loss': [], 
                           'diffusion_loss': [], 'diversity_loss': []}
            
            for batch_idx, batch in enumerate(train_ds):
                metrics = self.model.train_step(batch)
                
                for key in epoch_metrics.keys():
                    if key in metrics:
                        epoch_metrics[key].append(float(metrics[key]))
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: loss={metrics['loss']:.4f}", end='\r')
            
            # Moyennes de l'époque
            for key in epoch_metrics.keys():
                if epoch_metrics[key]:
                    avg = np.mean(epoch_metrics[key])
                    self.history[key].append(avg)
            
            print(f"\n  ✓ Train Loss: {self.history['loss'][-1]:.4f} | "
                  f"Recon: {self.history['recon_loss'][-1]:.4f} | "
                  f"Diffusion: {self.history['diffusion_loss'][-1]:.4f} | "
                  f"Diversity: {self.history['diversity_loss'][-1]:.4f}")
            
            # Validation
            if val_ds is not None and (epoch + 1) % 5 == 0:
                val_metrics = []
                for batch in val_ds:
                    metrics = self.model.test_step(batch)
                    val_metrics.append(float(metrics['loss']))
                print(f"  ✓ Val Loss: {np.mean(val_metrics):.4f}")
        
        print("\n" + "=" * 80)
        print("✅ Entraînement terminé!")
        
        return self.history
    
    def plot_history(self):
        """Affiche les courbes d'apprentissage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AQDN Training History', fontsize=16, fontweight='bold')
        
        # Loss totale
        axes[0, 0].plot(self.history['loss'], linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(self.history['recon_loss'], color='green', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Diffusion loss
        axes[1, 0].plot(self.history['diffusion_loss'], color='orange', linewidth=2)
        axes[1, 0].set_title('Diffusion Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diversity loss
        axes[1, 1].plot(self.history['diversity_loss'], color='red', linewidth=2)
        axes[1, 1].set_title('Diversity Loss (Anti-Overfitting)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def generate_predictions(model: AQDN, input_images, num_variations=5):
    """
    Génère des prédictions variées à partir d'images d'entrée
    
    Args:
        model: Modèle AQDN entraîné
        input_images: Images d'entrée (batch)
        num_variations: Nombre de variations à générer par image
    
    Returns:
        Prédictions générées
    """
    predictions = []
    
    for _ in range(num_variations):
        pred = model(input_images, training=False)
        predictions.append(pred.numpy())
    
    return np.array(predictions)


def visualize_predictions(input_img, predictions):
    """Visualise les prédictions générées"""
    num_variations = len(predictions)
    
    fig, axes = plt.subplots(1, num_variations + 1, figsize=(3 * (num_variations + 1), 3))
    fig.suptitle('AQDN Predictions', fontsize=14, fontweight='bold')
    
    # Image d'entrée
    axes[0].imshow((input_img + 1) / 2)  # Dénormalisation [-1,1] -> [0,1]
    axes[0].set_title('Input', fontweight='bold')
    axes[0].axis('off')
    
    # Prédictions
    for i, pred in enumerate(predictions):
        axes[i + 1].imshow((pred[0] + 1) / 2)
        axes[i + 1].set_title(f'Prediction {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 7. EXEMPLE D'UTILISATION COMPLET
# ============================================================================

def demo_aqdn():
    """Démo complète de l'architecture AQDN"""
    
    print("=" * 80)
    print("ADAPTIVE QUANTUM DIFFUSION NETWORK (AQDN)")
    print("Architecture innovante pour la génération d'images prédictives")
    print("=" * 80)
    
    # 1. Créer des données synthétiques pour la démo
    print("\n📊 Génération de données synthétiques...")
    def create_synthetic_data(num_samples=1000, img_size=64):
        """Crée un dataset synthétique de formes géométriques"""
        images = []
        
        for _ in range(num_samples):
            img = np.zeros((img_size, img_size, 3))
            
            # Ajouter des formes aléatoires
            shape_type = np.random.choice(['circle', 'square', 'triangle'])
            color = np.random.rand(3)
            
            center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
            center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
            size = np.random.randint(img_size // 8, img_size // 4)
            
            if shape_type == 'circle':
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= size**2
                img[mask] = color
            elif shape_type == 'square':
                x1, x2 = max(0, center_x - size), min(img_size, center_x + size)
                y1, y2 = max(0, center_y - size), min(img_size, center_y + size)
                img[y1:y2, x1:x2] = color
            else:  # triangle
                for i in range(img_size):
                    for j in range(img_size):
                        if (abs(i - center_y) + abs(j - center_x) < size):
                            img[i, j] = color
            
            images.append(img)
        
        images = np.array(images).astype(np.float32)
        # Normalisation [-1, 1]
        images = images * 2 - 1
        
        return images
    
    train_data = create_synthetic_data(800, img_size=64)
    val_data = create_synthetic_data(200, img_size=64)
    
    print(f"✓ Dataset créé: {train_data.shape[0]} images d'entraînement, {val_data.shape[0]} de validation")
    
    # 2. Créer et compiler le modèle
    print("\n🏗️  Construction du modèle AQDN...")
    model = AQDN(
        img_size=64,
        img_channels=3,
        latent_dim=256,
        diffusion_steps=10
    )
    
    # Build du modèle
    dummy_input = tf.random.normal((1, 64, 64, 3))
    _ = model(dummy_input, training=False)
    
    print(f"✓ Modèle construit: {model.count_params():,} paramètres")
    
    # 3. Entraîner le modèle
    print("\n🎓 Entraînement du modèle...")
    trainer = AQDNTrainer(model, learning_rate=1e-4)
    
    history = trainer.train(
        train_data,
        epochs=20,  # Réduit pour la démo
        batch_size=16,
        validation_data=val_data
    )
    
    # 4. Visualiser l'apprentissage
    print("\n📈 Visualisation de l'apprentissage...")
    trainer.plot_history()
    
    # 5. Générer des prédictions
    print("\n🎨 Génération de prédictions...")
    test_images = val_data[:3]
    predictions = generate_predictions(model, test_images, num_variations=4)
    
    # 6. Visualiser les résultats
    print("\n🖼️  Visualisation des prédictions...")
    for i in range(3):
        visualize_predictions(test_images[i], predictions[:, i])
    
    print("\n✅ Démo complète!")
    print("\n💡 Points clés de l'architecture AQDN:")
    print("  • Quantum Feature Extraction: Extraction multi-échelle parallèle")
    print("  • Stochastic Latent Diffusion: Génération progressive avec transformers")
    print("  • Diversity-Aware Regularization: Anti-mémorisation du dataset")
    print("  • Adaptive Decoder: Reconstruction avec skip connections")
    
    return model, trainer, history


# ============================================================================
# 8. SAUVEGARDE ET CHARGEMENT
# ============================================================================

def save_model(model: AQDN, path: str):
    """Sauvegarde le modèle"""
    model.save_weights(path)
    print(f"✓ Modèle sauvegardé: {path}")


def load_model(path: str, img_size=64, img_channels=3, latent_dim=256, diffusion_steps=10):
    """Charge un modèle sauvegardé"""
    model = AQDN(img_size, img_channels, latent_dim, diffusion_steps)
    
    # Build
    dummy = tf.random.normal((1, img_size, img_size, img_channels))
    _ = model(dummy, training=False)
    
    # Load weights
    model.load_weights(path)
    print(f"✓ Modèle chargé: {path}")
    
    return model


# ============================================================================
# LANCER LA DÉMO
# ============================================================================

if __name__ == "__main__":
    print("\n🎯 Lancement de la démonstration AQDN...\n")
    model, trainer, history = demo_aqdn()
    
    print("\n" + "=" * 80)
    print("📚 Pour utiliser AQDN sur vos propres données:")
    print("=" * 80)
    print("""
# 1. Créer le modèle
model = AQDN(img_size=64, img_channels=3, latent_dim=256, diffusion_steps=10)

# 2. Entraîner
trainer = AQDNTrainer(model, learning_rate=1e-4)
history = trainer.train(your_data, epochs=100, batch_size=32)

# 3. Générer des prédictions
predictions = generate_predictions(model, test_images, num_variations=5)

# 4. Visualiser
visualize_predictions(test_images[0], predictions[:, 0])

# 5. Sauvegarder
save_model(model, 'aqdn_model.h5')
    """)
    print("=" * 80)