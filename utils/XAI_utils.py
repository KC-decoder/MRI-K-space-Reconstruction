import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import fastmri
from utils.testing_utils import recon_kspace_cold_diffusion_from_perturbed_data
from diffusion.kspace_diffusion import mask_sequence_sample
from utils.testing_utils import recon_slice_unet
from tqdm import tqdm
import torch
import scipy.ndimage as ndimage
import torch.nn.functional as F
import gc
import fastmri
from fastmri.data import subsample, transforms, mri_data
from utils.sample_mask import RandomMaskGaussianDiffusion
import numpy as np
import matplotlib.pyplot as plt
from fastmri import fft2c, ifft2c, complex_abs

def manual_forward_and_scale(model, input_tensor, scale_coeff, device="cuda"):
    model.eval()
    input_tensor = input_tensor.to(device).float()
    with torch.no_grad():
        raw_output = model(input_tensor)
    output_scaled = torch.einsum('ijk, i -> ijk', raw_output.squeeze(1).cpu(), scale_coeff.cpu())
    return output_scaled
class EditColdDiffusion:
    """
    A class to perturb k-space data using the Cold Diffusion model and reconstruct
    it after each perturbation step using a trained model.
    """

    def __init__(self, model, model_path, npy_dir, sample_id, timesteps, num_perturbations, output_dir, npy_filename, device='cuda'):
        """
        Initializes the class.

        Args:
            model: The trained Cold Diffusion model.
            model_path: Path to the trained model checkpoint (.pth file).
            npy_dir: Directory containing saved reconstruction .npy files.
            sample_id: The sample index to analyze.
            timesteps: Number of diffusion steps.
            num_perturbations: Number of perturbation steps.
            output_dir: Directory to save results.
            device: Device to run computations on (default: 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.npy_dir = npy_dir
        self.sample_id = sample_id
        self.timesteps = timesteps
        self.num_perturbations = num_perturbations
        self.output_dir = output_dir
        self.npy_filename = npy_filename

        # Load the trained model weights
        self.load_trained_model(model_path)

        # Ensure save directory exists
        os.makedirs(output_dir, exist_ok=True)

    def load_trained_model(self, model_path):
        """
        Load the trained model from the specified path.

        Args:
            model_path: Path to the trained model checkpoint (.pth file).
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def run(self):
        """
        Loads k-space data from the precomputed NPY file, generates perturbations,
        reconstructs each perturbation, and stores results (including NMSE, PSNR, SSIM).
        """
        print(f"Running Cold Diffusion Perturbation Analysis for Sample ID {self.sample_id}...")

        # Load the precomputed reconstruction data
        npy_path = os.path.join(self.npy_dir, self.npy_filename)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Reconstruction NPY file not found: {npy_path}")
        
        print(f"Loading reconstruction data from {npy_path}")
        data = np.load(npy_path, allow_pickle=True).item()

        # Extract unperturbed k-space and mask
        kspace = torch.tensor(data["kspace"]).to(self.device)
        mask = torch.tensor(data["perturbations"]["perturbation_1_mask"]).to(self.device)
        mask_fold = torch.tensor(data["perturbations"]["perturbation_1_mask_fold"]).to(self.device)

        results = {
            "unperturbed_kspace": kspace.cpu().numpy(),
            "unperturbed_ground_truth": data["gt_imgs_abs"],
            "perturbations": {}
        }

        # Generate perturbations and reconstruct
        for i in tqdm(range(1, self.num_perturbations + 1), desc="Processing Perturbations"):
            perturb_key = f"perturbation_{i}"

            # Get the perturbed k-space from stored data
            perturbed_kspace = torch.tensor(data["perturbations"][perturb_key]["perturbation_kspace"]).to(self.device)

            # Perform reconstruction using Cold Diffusion
            pred, gt, pred_dir, metrics = recon_kspace_cold_diffusion_from_perturbed_data(
                perturbed_kspace, mask, mask_fold, self.model, self.timesteps, self.device
            )

            # Store perturbation results
            results["perturbations"][perturb_key] = {
                "perturbation_kspace": perturbed_kspace.cpu().numpy(),
                "perturbation_ground_truth": gt,
                "perturbation_reconstruction": pred,
                "metrics": metrics  # Store NMSE, PSNR, SSIM
            }

        # Save results as a structured .npy file
        save_file = os.path.join(self.output_dir, f'sample_{self.sample_id}_saliency.npy')
        np.save(save_file, results)
        print(f"Saved perturbation results for Sample ID {self.sample_id} to {save_file}.")




class GradCAM:
    def __init__(self, model, target_layer, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.cam_scores = None  # Will hold ranking values per channel

        # Register hooks
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        print("[GradCAM] Forward hook triggered")
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        print("[GradCAM] Backward hook triggered")
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def compute_cam(self, input_tensor):
        input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        output.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def compute_all_channel_cams(self, input_tensor, score_type="sum"):
        input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        output.sum().backward()

        cams = self.activations * self.gradients  # [B, C, H, W]
        cams = torch.relu(cams)
        cams = torch.nn.functional.interpolate(cams, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cams = cams[0].cpu().numpy()  # [C, H, W]

        # Normalize channels
        for i in range(cams.shape[0]):
            cam_i = cams[i]
            cams[i] = (cam_i - cam_i.min()) / (cam_i.max() - cam_i.min() + 1e-8)

        # Compute ranking scores
        self.cam_scores = self.rank_channels(cams, score_type)
        return cams  # [C, H, W]

    def rank_channels(self, cams, score_type="sum"):
        """
        Rank channels by one of:
        - 'sum': Total activation
        - 'max': Peak activation
        - 'var': Spatial variance
        """
        if score_type == "sum":
            scores = cams.sum(axis=(1, 2))
        elif score_type == "max":
            scores = cams.max(axis=(1, 2))
        elif score_type == "var":
            scores = cams.var(axis=(1, 2))
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")
        return scores

    def plot_top_k_channels(self, cams, k=16, input_image=None, save_path=None, score_type="sum"):
        """
        Plot top-k channels after ranking.

        Args:
            cams (np.ndarray): shape [C, H, W]
            k (int): number of top channels to show
            input_image (np.ndarray, optional): for overlay
            save_path (str, optional): if provided, saves to this path
            score_type (str): scoring method used for ranking
        """
        if self.cam_scores is None:
            self.cam_scores = self.rank_channels(cams, score_type)

        top_indices = np.argsort(self.cam_scores)[::-1][:k]

        cols = 4
        rows = int(np.ceil(k / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axs = axs.flatten()

        for i, idx in enumerate(top_indices):
            axs[i].imshow(input_image, cmap='gray') if input_image is not None else None
            axs[i].imshow(cams[idx], cmap='jet', alpha=0.5)
            axs[i].set_title(f"Ch {idx} | {score_type}: {self.cam_scores[idx]:.2f}")
            axs[i].axis("off")

        for i in range(k, len(axs)):
            axs[i].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[✓] Top-{k} channels saved to: {save_path}")
        plt.show()


def get_sample_from_loader(dataloader, idx_case, device="cuda" ):
    for idx, data in enumerate(dataloader):
        if idx != idx_case:
            continue
        X, y, mask = data
        return {
            "X": X.to(device).float(),
            "y": y.to(device).float(),
            "mask": mask.to(device).float()
        }


def run_gradcam_on_sample(
    model,
    pred,
    zf,
    tg,
    nmse,
    psnr,
    ssim,
    mask,
    X_input,
    device="cuda",
    target_layer=None,
    score_type="sum",           # Added: ranking method
    k_top=16,                   # Added: how many top channels to show
    save_path="/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/plot.png"
):
    """
    Compute Grad-CAM and visualizations for a single sample.
    Now also plots top-k ranked channels in both image and k-space.

    Args:
        model: PyTorch model
        pred, zf, tg, nmse, psnr, ssim, mask, X_input: from reconstruction function
        target_layer: layer for GradCAM (default = model.downs[-1][0])
        score_type (str): 'sum', 'max', or 'var'
        k_top (int): number of top channels to visualize
    """
    model = model.to(device).eval()
    input_for_cam = X_input.to(device).float().clone().detach().requires_grad_(True)

    if target_layer is None:
        target_layer = model.ups[-1][0]  # customize as needed

    cam = GradCAM(model, target_layer, device=device)

    # Compute CAMs
    cam_all = cam.compute_all_channel_cams(input_for_cam, score_type=score_type)
    print(f"[DEBUG] Number of CAM channels = {len(cam_all)}")

    zf_image = zf[0].cpu().numpy()
    cam_kspace_all = compute_kspace_overlay(cam_all, zf_image)

    # --- Top-k image-space CAMs ---
    cam.plot_top_k_channels(
        cams=cam_all,
        k=k_top,
        input_image=zf_image,
        save_path=save_path.replace(".png", f"_top{k_top}_{score_type}_image_space.png"),
        score_type=score_type
    )

    # --- Top-k k-space CAMs ---
    cam.plot_top_k_channels(
        cams=cam_kspace_all,
        k=k_top,
        input_image=None,
        save_path=save_path.replace(".png", f"_top{k_top}_{score_type}_kspace.png"),
        score_type=score_type
    )

    cam.remove_hooks()

def compute_kspace_overlay(cam_list, zf_image):
    """
    Compute Grad-CAM overlays in k-space domain for each channel.

    Args:
        cam_list: List of 2D CAMs (length 16).
        zf_image: 2D image array to get k-space magnitude.

    Returns:
        List of k-space overlay CAMs (length 16).
    """
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(zf_image)))
    fft_mag = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-8)
    return [cam * fft_mag for cam in cam_list]

def plot_cam_grid(cam_list, title_prefix, save_path):
    """
    Plots a 4x4 grid of CAMs.

    Args:
        cam_list (List[np.ndarray]): List of 2D CAM arrays (length 16).
        title_prefix (str): Prefix for each subplot title (e.g., "Image Space", "K-space").
        save_path (str): Path to save the final PNG file.
    """
    num_cams = len(cam_list)
    cols = 4
    rows = math.ceil(num_cams / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten()

    for i, cam in enumerate(cam_list):
        axs[i].imshow(cam, cmap='jet')
        axs[i].set_title(f"{title_prefix} CAM {i}")
        axs[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {title_prefix} CAMs to {save_path}")

def plot_cam_channels(cam_maps, input_image=None, save_path=None):
    """
    cam_maps: np.ndarray of shape [C, H, W]
    input_image: optional background image (H, W)
    """
    num_channels = cam_maps.shape[0]
    cols = 4
    rows = int(np.ceil(num_channels / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten()

    for i in range(num_channels):
        axs[i].imshow(input_image, cmap='gray') if input_image is not None else None
        axs[i].imshow(cam_maps[i], cmap='jet', alpha=0.5)
        axs[i].set_title(f"Channel {i}")
        axs[i].axis("off")

    for i in range(num_channels, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[✓] Saved channel CAMs to: {save_path}")
    plt.show()

def plot_full_gradcam_3x2(mask, zf, gt, recon, gradcam_img, gradcam_k, ssim, save_path):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].imshow(mask, cmap='gray')
    axs[0, 0].set_title("k-space Mask")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(zf, cmap='gray')
    axs[0, 1].set_title("Undersampled Input")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(gt, cmap='gray')
    axs[1, 0].set_title("Ground Truth")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(recon, cmap='gray')
    axs[1, 1].set_title(f"Prediction\nSSIM={ssim:.3f}")
    axs[1, 1].axis("off")

    axs[2, 0].imshow(gradcam_img[0], cmap='jet')
    axs[2, 0].set_title("Grad-CAM (Image)")
    axs[2, 0].axis("off")

    axs[2, 1].imshow(gradcam_k[0], cmap='jet')
    axs[2, 1].set_title("Grad-CAM (K-space)")
    axs[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Grad-CAM panel to: {save_path}")


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()



class KSpaceGradCAMPlusPlus:
    """
    Grad-CAM++ implementation specifically designed for k-space MRI reconstruction analysis.
    
    Key Mathematical Improvements over Grad-CAM:
    1. Pixel-wise weighting: α^kc_ij for each spatial location
    2. Higher-order derivatives: ∂²Y^c/(∂A^k_ij)² and ∂³Y^c/(∂A^k_ij)³
    3. Better localization for multiple patterns (critical for ring masks)
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        
        # Target the FINAL convolution layer (64 → 1 channels) - same as your original
        self.target_layer = model.final_conv
        
        # Storage for gradients and activations
        self.gradients = None          # ∂Y^c/∂A^k_ij
        self.gradients_2nd = None      # ∂²Y^c/(∂A^k_ij)²  
        self.gradients_3rd = None      # ∂³Y^c/(∂A^k_ij)³
        self.activations = None        # A^k_ij
        self.hook_handles = []
        
        print(f"🎯 Grad-CAM++ targeting FINAL LAYER: {type(self.target_layer).__name__}")
        print(f"   This layer directly generates the output reconstruction!")
        
        # Register hooks for forward and backward passes
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
    
    def _save_activation(self, module, input, output):
        """Save forward activations A^k_ij."""
        self.activations = output.detach()
        print(f"[Grad-CAM++] Forward activations captured: {self.activations.shape}")
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward gradients ∂Y^c/∂A^k_ij."""
        self.gradients = grad_output[0].detach()
        print(f"[Grad-CAM++] Backward gradients captured: {self.gradients.shape}")
    
    def _compute_higher_order_derivatives(self, input_tensor):
        """
        Compute second and third-order derivatives using the mathematical formulation
        from Grad-CAM++ paper for exponential activation.
        
        For exponential activation Y^c = exp(S^c):
        ∂²Y^c/(∂A^k_ij)² = exp(S^c) * (∂S^c/∂A^k_ij)²
        ∂³Y^c/(∂A^k_ij)³ = exp(S^c) * (∂S^c/∂A^k_ij)³
        """
        print("🧮 Computing higher-order derivatives for Grad-CAM++...")
        
        # The gradients we captured are already ∂Y^c/∂A^k_ij = exp(S^c) * ∂S^c/∂A^k_ij
        # We need to extract ∂S^c/∂A^k_ij to compute higher orders
        
        # For the final layer with exponential output, we can approximate:
        # If Y^c = exp(S^c), then ∂S^c/∂A^k_ij = (∂Y^c/∂A^k_ij) / Y^c
        
        # Forward pass to get Y^c
        with torch.no_grad():
            output = self.model(input_tensor)  # Y^c
        
        # Compute ∂S^c/∂A^k_ij 
        ds_da = self.gradients / (output + 1e-8)  # Avoid division by zero
        
        # Second-order: ∂²Y^c/(∂A^k_ij)² = exp(S^c) * (∂S^c/∂A^k_ij)²
        self.gradients_2nd = output * (ds_da ** 2)
        
        # Third-order: ∂³Y^c/(∂A^k_ij)³ = exp(S^c) * (∂S^c/∂A^k_ij)³  
        self.gradients_3rd = output * (ds_da ** 3)
        
        print(f"[Grad-CAM++] Second-order gradients: {self.gradients_2nd.shape}")
        print(f"[Grad-CAM++] Third-order gradients: {self.gradients_3rd.shape}")
    
    def _compute_pixel_wise_weights(self):
        """
        Compute the pixel-wise importance weights α^kc_ij using Grad-CAM++ formulation:
        
        α^kc_ij = (∂²Y^c/(∂A^k_ij)²) / [2(∂²Y^c/(∂A^k_ij)²) + Σ_a Σ_b A^k_ab(∂³Y^c/(∂A^k_ij)³)]
        
        This gives each spatial location its own importance weight.
        """
        print("⚖️  Computing pixel-wise importance weights α^kc_ij...")
        
        # Numerator: ∂²Y^c/(∂A^k_ij)²
        numerator = self.gradients_2nd
        
        # Denominator: 2(∂²Y^c/(∂A^k_ij)²) + Σ_a Σ_b A^k_ab(∂³Y^c/(∂A^k_ij)³)
        # The summation term Σ_a Σ_b A^k_ab is just the sum over spatial dimensions
        activation_sum = torch.sum(self.activations, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        denominator = 2 * self.gradients_2nd + activation_sum * self.gradients_3rd
        
        # Compute α^kc_ij with numerical stability
        alpha = numerator / (denominator + 1e-8)
        
        print(f"[Grad-CAM++] Pixel-wise weights α computed: {alpha.shape}")
        return alpha
    
    def _compute_gradcam_plus_plus_weights(self, alpha):
        """
        Compute final Grad-CAM++ channel weights:
        w^c_k = Σ_i Σ_j [α^kc_ij · relu(∂Y^c/∂A^k_ij)]
        
        This is the key difference from Grad-CAM which uses simple averaging.
        """
        print("🎯 Computing Grad-CAM++ channel weights w^c_k...")
        
        # Apply ReLU to gradients (only positive contributions)
        positive_gradients = F.relu(self.gradients)
        
        # Pixel-wise weighting: α^kc_ij * relu(∂Y^c/∂A^k_ij)
        weighted_gradients = alpha * positive_gradients
        
        # Sum over spatial dimensions to get channel weights
        weights = torch.sum(weighted_gradients, dim=(2, 3))  # [B, C]
        
        print(f"[Grad-CAM++] Final channel weights: {weights.shape}")
        return weights
    
    def compute_kspace_gradcam_plus_plus(self, input_tensor, ring_mask):
        """
        Main method to compute Grad-CAM++ for k-space analysis.
        
        Args:
            input_tensor: Input to model [B, C, H, W] 
            ring_mask: Ring mask for k-space analysis [H, W]
            
        Returns:
            Dictionary with comprehensive k-space Grad-CAM++ analysis
        """
        print("🌟 Computing K-space Grad-CAM++ Analysis...")
        print("=" * 60)
        
        # Clear previous computations
        self._clear_gradients()
        
        # Ensure input requires gradients
        if input_tensor.shape[0] > 1:
            input_tensor = input_tensor[:1]  # Single batch for memory efficiency
        
        input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        target = output.mean()  # Overall reconstruction quality target
        
        print(f"🎯 Model output shape: {output.shape}")
        print(f"🎯 Target value: {target.item():.6f}")
        
        # Backward pass to get first-order gradients
        target.backward(retain_graph=False)
        
        # Compute higher-order derivatives
        self._compute_higher_order_derivatives(input_tensor)
        
        # Compute pixel-wise importance weights α^kc_ij
        alpha = self._compute_pixel_wise_weights()
        
        # Compute final Grad-CAM++ weights
        weights = self._compute_gradcam_plus_plus_weights(alpha)
        
        # Generate attention map using weighted combination
        image_attention = self._generate_attention_map(weights, input_tensor.shape[-2:])
        
        # Perform k-space analysis
        kspace_results = self._analyze_kspace_patterns(
            image_attention, ring_mask, input_tensor, alpha, weights
        )
        
        return kspace_results
    
    def _generate_attention_map(self, weights, target_size):
        """
        Generate the final attention map using Grad-CAM++ weights:
        L^c_ij = relu(Σ_k w^c_k * A^k_ij)
        """
        print("🗺️  Generating Grad-CAM++ attention map...")
        
        # Weighted combination of activation maps
        # weights: [B, C], activations: [B, C, H, W]
        weighted_activations = weights.unsqueeze(-1).unsqueeze(-1) * self.activations
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply ReLU and normalization
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def _analyze_kspace_patterns(self, image_attention, ring_mask, input_tensor, alpha, weights):
        """
        Comprehensive k-space analysis using Grad-CAM++ results.
        Enhanced with pixel-wise importance insights.
        """
        print("🌊 Analyzing K-space patterns with Grad-CAM++ insights...")
        
        # Convert attention to k-space domain
        kspace_attention = self._image_to_kspace_attention(image_attention)
        
        # Enhanced frequency analysis with pixel-wise weights
        freq_analysis = self._analyze_frequency_bands_plus_plus(kspace_attention, alpha)
        
        # Ring mask correlation with enhanced sensitivity
        ring_correlation = self._analyze_ring_correlation_plus_plus(kspace_attention, ring_mask, alpha)
        
        # Pixel-wise importance analysis (unique to Grad-CAM++)
        pixel_importance = self._analyze_pixel_importance(alpha, weights)
        
        # Radial frequency analysis
        radial_analysis = self._analyze_radial_frequencies(kspace_attention)
        
        return {
            'image_attention': image_attention,
            'kspace_attention': kspace_attention,
            'frequency_analysis': freq_analysis,
            'ring_correlation': ring_correlation,
            'pixel_importance': pixel_importance,  # New for Grad-CAM++
            'radial_analysis': radial_analysis,
            'alpha_weights': alpha.cpu().numpy(),     # Pixel-wise weights
            'channel_weights': weights.cpu().numpy()  # Final channel weights
        }
    
    def _analyze_pixel_importance(self, alpha, weights):
        """
        Analyze pixel-wise importance patterns unique to Grad-CAM++.
        This shows which spatial locations have the highest individual impact.
        """
        print("🔍 Analyzing pixel-wise importance patterns...")
        
        alpha_np   = alpha.squeeze().cpu().numpy()    # could be (H,W) if C=1
        weights_np = weights.squeeze().cpu().numpy()  # could be scalar

        if alpha_np.ndim == 2:
            # single channel
            pixel_importance = weights_np * alpha_np
        else:
            # multi-channel
            C, H, W = alpha_np.shape
            pixel_importance = np.zeros((H, W), dtype=float)
            for c in range(C):
                pixel_importance += weights_np[c] * alpha_np[c]
        
        
        
        # Find most and least important regions
        top_5_percent = np.percentile(pixel_importance.flatten(), 95)
        bottom_5_percent = np.percentile(pixel_importance.flatten(), 5)
        
        high_importance_mask = pixel_importance > top_5_percent
        low_importance_mask = pixel_importance < bottom_5_percent
        
        return {
            'pixel_importance_map': pixel_importance,
            'high_importance_regions': high_importance_mask,
            'low_importance_regions': low_importance_mask,
            'importance_std': np.std(pixel_importance),
            'importance_range': np.max(pixel_importance) - np.min(pixel_importance)
        }
    
    def _analyze_frequency_bands_plus_plus(self, kspace_attention, alpha):
        """Enhanced frequency analysis using pixel-wise importance weights."""
        H, W = kspace_attention.shape
        center_h, center_w = H // 2, W // 2
        
        # Create distance map
        y, x = np.ogrid[:H, :W]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # Define frequency bands
        low_freq_mask = distances < (max_dist * 0.1)
        mid_freq_mask = (distances >= (max_dist * 0.1)) & (distances < (max_dist * 0.4))
        high_freq_mask = distances >= (max_dist * 0.4)
        
        # Compute importance using both attention and pixel-wise weights
        alpha_combined = np.mean(alpha.cpu().numpy().squeeze(), axis=0)  # Average across channels
        enhanced_attention = kspace_attention * (1 + alpha_combined)  # Boost by pixel importance
        
        low_freq_importance = np.mean(enhanced_attention[low_freq_mask])
        mid_freq_importance = np.mean(enhanced_attention[mid_freq_mask]) 
        high_freq_importance = np.mean(enhanced_attention[high_freq_mask])
        
        total = low_freq_importance + mid_freq_importance + high_freq_importance
        
        return {
            'low_freq_weight': low_freq_importance,
            'mid_freq_weight': mid_freq_importance, 
            'high_freq_weight': high_freq_importance,
            'low_freq_percentage': (low_freq_importance / total) * 100,
            'mid_freq_percentage': (mid_freq_importance / total) * 100,
            'high_freq_percentage': (high_freq_importance / total) * 100,
            'frequency_masks': {
                'low': low_freq_mask,
                'mid': mid_freq_mask,
                'high': high_freq_mask
            },
            'enhanced_attention': enhanced_attention  # New for Grad-CAM++
        }
    
    def _analyze_ring_correlation_plus_plus(self, kspace_attention, ring_mask, alpha):
        """Enhanced ring mask correlation using pixel-wise weights."""
        # Resize ring mask if needed
        if ring_mask.shape != kspace_attention.shape:
            ring_mask_resized = ndimage.zoom(ring_mask, 
                                           (kspace_attention.shape[0]/ring_mask.shape[0],
                                            kspace_attention.shape[1]/ring_mask.shape[1]), 
                                           order=0)
        else:
            ring_mask_resized = ring_mask
        
        # Enhanced attention using pixel-wise importance
        alpha_combined = np.mean(alpha.cpu().numpy().squeeze(), axis=0)
        enhanced_attention = kspace_attention * (1 + alpha_combined)
        
        # Analyze sampled vs unsampled regions
        sampled_regions = ring_mask_resized > 0.5
        unsampled_regions = ring_mask_resized <= 0.5
        
        # Compare standard vs enhanced attention
        sampled_attention_std = np.mean(kspace_attention[sampled_regions])
        sampled_attention_enh = np.mean(enhanced_attention[sampled_regions])
        
        unsampled_attention_std = np.mean(kspace_attention[unsampled_regions]) 
        unsampled_attention_enh = np.mean(enhanced_attention[unsampled_regions])
        
        # Correlations
        corr_std = np.corrcoef(ring_mask_resized.flatten(), kspace_attention.flatten())[0, 1]
        corr_enh = np.corrcoef(ring_mask_resized.flatten(), enhanced_attention.flatten())[0, 1]
        
        return {
            'sampled_attention_standard': sampled_attention_std,
            'sampled_attention_enhanced': sampled_attention_enh,
            'unsampled_attention_standard': unsampled_attention_std,
            'unsampled_attention_enhanced': unsampled_attention_enh,
            'correlation_standard': corr_std,
            'correlation_enhanced': corr_enh,  # Should be higher with Grad-CAM++
            'enhancement_factor': corr_enh / (corr_std + 1e-8),
            'ring_mask_resized': ring_mask_resized
        }
    
    def _image_to_kspace_attention(self, image_attention):
        """Convert image-space attention to k-space via FFT."""
        kspace_complex = np.fft.fftshift(np.fft.fft2(image_attention))
        kspace_magnitude = np.abs(kspace_complex)
        
        if kspace_magnitude.max() > kspace_magnitude.min():
            kspace_attention = (kspace_magnitude - kspace_magnitude.min()) / \
                              (kspace_magnitude.max() - kspace_magnitude.min())
        else:
            kspace_attention = kspace_magnitude
            
        return kspace_attention
    
    def _analyze_radial_frequencies(self, kspace_attention):
        """Radial frequency profile analysis."""
        H, W = kspace_attention.shape
        center_h, center_w = H // 2, W // 2
        
        y, x = np.ogrid[:H, :W]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        max_radius = int(np.sqrt(center_h**2 + center_w**2))
        radial_profile = []
        radii = []
        
        for r in range(0, max_radius, 2):
            mask = (distances >= r) & (distances < r + 2)
            if np.any(mask):
                radial_profile.append(np.mean(kspace_attention[mask]))
                radii.append(r)
        
        return {
            'radii': np.array(radii),
            'radial_profile': np.array(radial_profile),
            'distance_map': distances
        }
    
    def _clear_gradients(self):
        """Clear stored gradients and activations."""
        self.gradients = None
        self.gradients_2nd = None
        self.gradients_3rd = None
        self.activations = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def remove_hooks(self):
        """Clean up hooks and memory."""
        for handle in self.hook_handles:
            handle.remove()
        self._clear_gradients()

def analyze_final_layer_kspace_gradcam_plus_plus(model, X_for_gradcam, device, tg, zf, pred, 
                                                 ssim, psnr, ring_mask_path, exp_path):
    """
    Complete k-space analysis using Grad-CAM++ on the final layer.
    
    This function demonstrates the key improvements of Grad-CAM++ over standard Grad-CAM:
    1. Pixel-wise importance weighting instead of global averaging
    2. Better localization of multiple frequency patterns  
    3. Enhanced sensitivity to ring mask boundaries
    4. More precise k-space attention mapping
    """
    print("🌟 K-SPACE GRAD-CAM++ ANALYSIS - FINAL LAYER")
    print("=" * 60)
    print("📊 KEY IMPROVEMENTS OVER GRAD-CAM:")
    print("   • Pixel-wise weighting α^kc_ij for each spatial location")
    print("   • Higher-order derivatives for better localization")  
    print("   • Enhanced frequency pattern discrimination")
    print("   • Superior ring mask correlation analysis")
    print("=" * 60)
    
    # Load ring mask  
    ring_mask = np.load(ring_mask_path)
    print(f"🎭 Loaded ring mask: {ring_mask.shape}")
    
    try:
        # Initialize Grad-CAM++
        gradcam_plus_plus = KSpaceGradCAMPlusPlus(model, device)
        
        # Perform Grad-CAM++ analysis
        results = gradcam_plus_plus.compute_kspace_gradcam_plus_plus(X_for_gradcam, ring_mask)
        
        # Create enhanced visualization
        create_gradcam_plus_plus_visualization(
            results=results,
            input_img=zf[0].cpu().numpy() if torch.is_tensor(zf) else zf,
            target_img=tg[0].cpu().numpy() if torch.is_tensor(tg) else tg,
            pred_img=pred[0].cpu().numpy() if torch.is_tensor(pred) else pred,
            ring_mask=ring_mask,
            ssim=ssim,
            save_path=exp_path / "VISUALIZATIONS" / "kspace_gradcam_plus_plus_ring2.png"
        )
        
        # Print key insights comparing standard vs enhanced analysis
        print("\n🔍 KEY GRAD-CAM++ INSIGHTS:")
        freq_analysis = results['frequency_analysis']
        ring_corr = results['ring_correlation']  
        pixel_imp = results['pixel_importance']
        
        print(f"   🎯 Enhanced ring correlation: {ring_corr['correlation_enhanced']:.3f} vs {ring_corr['correlation_standard']:.3f}")
        print(f"   🎯 Enhancement factor: {ring_corr['enhancement_factor']:.2f}x improvement")
        print(f"   🎯 Pixel importance range: {pixel_imp['importance_range']:.3f}")
        print(f"   🎯 Most important freq band: {['Low', 'Mid', 'High'][np.argmax([freq_analysis['low_freq_percentage'], freq_analysis['mid_freq_percentage'], freq_analysis['high_freq_percentage']])]}")
        
        # Cleanup
        gradcam_plus_plus.remove_hooks()
        
        return results
        
    except Exception as e:
        print(f"❌ Grad-CAM++ analysis failed: {e}")
        if 'gradcam_plus_plus' in locals():
            gradcam_plus_plus.remove_hooks()
        return None

def create_gradcam_plus_plus_visualization(results, input_img, target_img, pred_img, 
                                         ring_mask, ssim, save_path):
    """
    Create comprehensive Grad-CAM++ visualization showing the improvements
    over standard Grad-CAM through enhanced k-space analysis.
    """
    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Row 1: Original images and basic attention
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_img, cmap='gray')
    ax1.set_title('Undersampled Input')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax2.imshow(target_img, cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_img, cmap='gray')
    ax3.set_title(f'Reconstruction\nSSIM: {ssim:.3f}')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(ring_mask, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Ring Mask 2\n(K-space Sampling)')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(results['image_attention'], cmap='jet')
    ax5.set_title('Grad-CAM++ Attention\n(Image Space)')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[0, 5])
    ax6.imshow(results['kspace_attention'], cmap='jet')  
    ax6.set_title('Grad-CAM++ Attention\n(K-space)')
    ax6.axis('off')
    
    # Row 2: Pixel-wise importance analysis (unique to Grad-CAM++)
    pixel_imp = results['pixel_importance']
    
    ax7 = fig.add_subplot(gs[1, 0])
    ax7.imshow(pixel_imp['pixel_importance_map'], cmap='viridis')
    ax7.set_title('Pixel-wise Importance α^kc_ij\n(Grad-CAM++ Only)')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 1])
    ax8.imshow(pixel_imp['high_importance_regions'], cmap='Reds')
    ax8.set_title('High Importance Regions\n(Top 5%)')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[1, 2])
    enhanced_attention = results['frequency_analysis']['enhanced_attention']
    ax9.imshow(enhanced_attention, cmap='jet')
    ax9.set_title('Enhanced K-space Attention\n(α-weighted)')
    ax9.axis('off')
    
    # Row 3: Frequency analysis comparison
    freq_analysis = results['frequency_analysis']
    
    ax10 = fig.add_subplot(gs[2, 0])
    freq_masks = freq_analysis['frequency_masks']
    combined_mask = np.zeros_like(results['kspace_attention'])
    combined_mask[freq_masks['low']] = 1
    combined_mask[freq_masks['mid']] = 0.5  
    combined_mask[freq_masks['high']] = 0.2
    ax10.imshow(combined_mask, cmap='RdYlBu_r')
    ax10.set_title('Frequency Bands\nRed=Low, Yellow=Mid, Blue=High')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 1])
    sizes = [freq_analysis['low_freq_percentage'], 
             freq_analysis['mid_freq_percentage'],
             freq_analysis['high_freq_percentage']]
    labels = ['Low Freq\n(Structure)', 'Mid Freq\n(Anatomy)', 'High Freq\n(Details)']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax11.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax11.set_title('Frequency Importance\n(Enhanced)')
    
    # Row 3: Ring correlation analysis  
    ring_corr = results['ring_correlation']
    
    ax12 = fig.add_subplot(gs[2, 2])
    ring_mask_resized = ring_corr['ring_mask_resized']
    ax12.imshow(ring_mask_resized, cmap='gray', alpha=0.7)
    ax12.imshow(results['kspace_attention'], cmap='jet', alpha=0.5)
    ax12.set_title(f'Standard Correlation\n{ring_corr["correlation_standard"]:.3f}')
    ax12.axis('off')
    
    ax13 = fig.add_subplot(gs[2, 3])
    ax13.imshow(ring_mask_resized, cmap='gray', alpha=0.7)
    ax13.imshow(enhanced_attention, cmap='jet', alpha=0.5)  
    ax13.set_title(f'Enhanced Correlation\n{ring_corr["correlation_enhanced"]:.3f}')
    ax13.axis('off')
    
    # Row 4: Radial analysis and channel weights
    radial = results['radial_analysis']
    ax14 = fig.add_subplot(gs[3, 0])
    ax14.plot(radial['radii'], radial['radial_profile'], 'b-', linewidth=2, label='Grad-CAM++')
    ax14.set_xlabel('Radial Distance (pixels)')
    ax14.set_ylabel('Attention Importance')
    ax14.set_title('Enhanced Radial Profile')
    ax14.grid(True, alpha=0.3)
    ax14.legend()
    
    # Channel weights visualization
    ax15 = fig.add_subplot(gs[3, 1])
    channel_weights = results['channel_weights'].flatten()
    ax15.bar(range(len(channel_weights)), channel_weights, alpha=0.7, color='steelblue')
    ax15.set_xlabel('Channel Index')
    ax15.set_ylabel('Weight Value')
    ax15.set_title('Final Channel Weights w^c_k')
    ax15.grid(True, alpha=0.3)
    
    # Row 5: Comprehensive statistics
    ax16 = fig.add_subplot(gs[4, :])
    
    stats_text = f"""
    K-SPACE GRAD-CAM++ ANALYSIS - FINAL LAYER (model.final_conv) - RING MASK 2
    {'='*120}

    🎯 LAYER ANALYZED: Final convolution layer (64 → 1 channels) - DIRECTLY GENERATES OUTPUT!

    📊 GRAD-CAM++ ENHANCEMENTS:
    • Pixel-wise weighting: Each spatial location gets individual importance weight α^kc_ij
    • Higher-order derivatives: ∂²Y^c/(∂A^k_ij)² and ∂³Y^c/(∂A^k_ij)³ for better localization
    • Enhanced sensitivity: Improved detection of multiple frequency patterns in Ring 2

    🔍 FREQUENCY ANALYSIS (Grad-CAM++ Enhanced):
    • Low Freq (Structure):  {freq_analysis['low_freq_percentage']:.1f}% importance
    • Mid Freq (Anatomy):    {freq_analysis['mid_freq_percentage']:.1f}% importance  
    • High Freq (Details):   {freq_analysis['high_freq_percentage']:.1f}% importance

    ⭕ RING MASK 2 CORRELATION IMPROVEMENTS:
    • Standard correlation:     {ring_corr['correlation_standard']:.3f}
    • Enhanced correlation:     {ring_corr['correlation_enhanced']:.3f}  
    • Enhancement factor:       {ring_corr['enhancement_factor']:.2f}x improvement
    • Sampled region attention: {ring_corr['sampled_attention_enhanced']:.3f} (enhanced) vs {ring_corr['sampled_attention_standard']:.3f} (standard)

    🔬 PIXEL-WISE IMPORTANCE INSIGHTS:
    • Importance range:         {pixel_imp['importance_range']:.3f}
    • Importance std:           {pixel_imp['importance_std']:.3f}  
    • High importance regions:  {np.sum(pixel_imp['high_importance_regions'])} pixels (top 5%)

    🌊 K-SPACE GRAD-CAM++ vs GRAD-CAM:
    • Peak attention:           {results['kspace_attention'].max():.3f}
    • Mean attention:           {results['kspace_attention'].mean():.3f}
    • Attention spread (std):   {results['kspace_attention'].std():.3f}

    💡 CLINICAL INTERPRETATION:
    • Primary focus: {"Low frequencies (structure)" if freq_analysis['low_freq_percentage'] > 50 else "Mid frequencies (anatomy)" if freq_analysis['mid_freq_percentage'] > 40 else "Mixed frequency approach"}
    • Ring 2 utilization: {"Excellent - strong enhanced correlation" if ring_corr['enhancement_factor'] > 1.5 else "Good - moderate enhancement" if ring_corr['enhancement_factor'] > 1.2 else "Limited - minimal improvement"}
    • Sampling efficiency: {"Optimal" if ring_corr['correlation_enhanced'] > 0.4 else "Good" if ring_corr['correlation_enhanced'] > 0.2 else "Needs improvement"}
    """
    
    ax16.text(0.02, 0.98, stats_text, transform=ax16.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax16.axis('off')
    
    plt.suptitle('K-space Grad-CAM++ Analysis: Enhanced Final Layer Focus with Ring Mask 2', 
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Grad-CAM++ visualization saved to: {save_path}")


def ring_focused_mask(H, W, r, dr, p_in=0.9, p_out=0.1, seed=None):
    """
    Generate a binary k-space sampling mask that:
      • Samples inside the ring (radius r±dr) with probability p_in.
      • Samples outside the ring with probability p_out.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Centered coordinates
    u = np.arange(H) - H/2
    v = np.arange(W) - W/2
    U, V = np.meshgrid(u, v, indexing='ij')

    # 2) Radius map
    R = np.sqrt(U**2 + V**2)

    # 3) Ring indicator
    in_ring = (R >= (r - dr)) & (R <= (r + dr))

    # 4) Random thresholds
    X = np.random.rand(H, W)

    # 5) Build mask
    mask = np.zeros((H, W), dtype=np.float32)
    mask[in_ring]  = (X[in_ring]  < p_in ).astype(np.float32)
    mask[~in_ring] = (X[~in_ring] < p_out).astype(np.float32)

    return mask

    

