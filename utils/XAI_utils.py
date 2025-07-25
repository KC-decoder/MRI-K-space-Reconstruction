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

class KSpaceGradCAM:
    """
    Grad-CAM specifically designed for k-space analysis of MRI reconstruction.
    Focuses on understanding attention patterns in frequency domain.
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # Target the VERY FINAL LAYER (64 channels → 1 channel output)
        self.target_layer = model.final_conv
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        print(f"Analyzing FINAL LAYER: {type(self.target_layer).__name__}")
        print(f"This layer directly generates the output image!")
        
        # Register hooks
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self.save_activation)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self.save_gradient)
        )
    
    def save_activation(self, module, input, output):
        """Save activation from the final layer."""
        self.activations = output.detach()
        print(f"[DEBUG] Final layer activations captured: {self.activations.shape}")
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients from the final layer."""
        self.gradients = grad_output[0].detach()
        print(f"[DEBUG] Final layer gradients captured: {self.gradients.shape}")
    
    def compute_kspace_gradcam(self, input_tensor, ring_mask):
        """
        Compute Grad-CAM focused on k-space analysis.
        
        Args:
            input_tensor: Input to model [B, C, H, W]
            ring_mask: Ring mask used for undersampling [H, W]
            
        Returns:
            Dictionary with k-space analysis results
        """
        print("Computing K-space Grad-CAM for FINAL LAYER...")
        
        # Clear memory and prepare input
        clear_gpu_memory()
        if input_tensor.shape[0] > 1:
            input_tensor = input_tensor[:1]  # Single batch for memory
        
        input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        target = output.mean()  # Overall reconstruction quality
        
        print(f"[DEBUG] Model output shape: {output.shape}")
        print(f"[DEBUG] Target value: {target.item():.6f}")
        
        # Backward pass
        target.backward(retain_graph=False)
        
        # Compute Grad-CAM attention map
        image_attention = self._compute_image_attention(input_tensor.shape[-2:])
        
        # Convert to k-space analysis
        kspace_results = self._analyze_kspace_patterns(
            image_attention, ring_mask, input_tensor
        )
        
        return kspace_results
    
    def _compute_image_attention(self, target_size):
        """Compute standard Grad-CAM attention map."""
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")
        
        # For final conv layer: gradients and activations should be [B, 64, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [B, 64, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def _analyze_kspace_patterns(self, image_attention, ring_mask, input_tensor):
        """
        Analyze k-space patterns from image-space attention.
        This is the core k-space analysis.
        """
        print("🌊 Analyzing K-space patterns...")
        
        # 1. Convert attention to k-space
        kspace_attention = self._image_to_kspace_attention(image_attention)
        
        # 2. Analyze frequency bands
        freq_analysis = self._analyze_frequency_bands(kspace_attention)
        
        # 3. Ring mask correlation analysis
        ring_correlation = self._analyze_ring_mask_correlation(kspace_attention, ring_mask)
        
        # 4. Radial frequency analysis
        radial_analysis = self._analyze_radial_frequencies(kspace_attention)
        
        # 5. K-space sampling efficiency
        sampling_analysis = self._analyze_sampling_efficiency(kspace_attention, ring_mask)
        
        return {
            'image_attention': image_attention,
            'kspace_attention': kspace_attention,
            'frequency_analysis': freq_analysis,
            'ring_correlation': ring_correlation,
            'radial_analysis': radial_analysis,
            'sampling_analysis': sampling_analysis
        }
    
    def _image_to_kspace_attention(self, image_attention):
        """Convert image-space attention to k-space attention via FFT."""
        # Apply FFT to convert spatial attention to frequency attention
        kspace_complex = np.fft.fftshift(np.fft.fft2(image_attention))
        kspace_magnitude = np.abs(kspace_complex)
        
        # Normalize k-space attention
        if kspace_magnitude.max() > kspace_magnitude.min():
            kspace_attention = (kspace_magnitude - kspace_magnitude.min()) / \
                              (kspace_magnitude.max() - kspace_magnitude.min())
        else:
            kspace_attention = kspace_magnitude
        
        return kspace_attention
    
    def _analyze_frequency_bands(self, kspace_attention):
        """Analyze importance of different frequency bands."""
        H, W = kspace_attention.shape
        center_h, center_w = H // 2, W // 2
        
        # Create distance map from k-space center
        y, x = np.ogrid[:H, :W]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # Define frequency bands
        low_freq_mask = distances < (max_dist * 0.1)      # Central 10% - structure
        mid_freq_mask = (distances >= (max_dist * 0.1)) & (distances < (max_dist * 0.4))  # Anatomy
        high_freq_mask = distances >= (max_dist * 0.4)     # Details/edges
        
        # Compute importance weights for each band
        low_freq_importance = np.mean(kspace_attention[low_freq_mask])
        mid_freq_importance = np.mean(kspace_attention[mid_freq_mask])
        high_freq_importance = np.mean(kspace_attention[high_freq_mask])
        
        total_importance = low_freq_importance + mid_freq_importance + high_freq_importance
        
        return {
            'low_freq_weight': low_freq_importance,
            'mid_freq_weight': mid_freq_importance,
            'high_freq_weight': high_freq_importance,
            'low_freq_percentage': (low_freq_importance / total_importance) * 100,
            'mid_freq_percentage': (mid_freq_importance / total_importance) * 100,
            'high_freq_percentage': (high_freq_importance / total_importance) * 100,
            'frequency_masks': {
                'low': low_freq_mask,
                'mid': mid_freq_mask,
                'high': high_freq_mask
            }
        }
    
    def _analyze_ring_mask_correlation(self, kspace_attention, ring_mask):
        """Analyze how attention correlates with ring mask patterns."""
        # Resize ring mask to match k-space attention if needed
        if ring_mask.shape != kspace_attention.shape:
            ring_mask_resized = ndimage.zoom(ring_mask, 
                                           (kspace_attention.shape[0]/ring_mask.shape[0],
                                            kspace_attention.shape[1]/ring_mask.shape[1]), 
                                           order=0)
        else:
            ring_mask_resized = ring_mask
        
        # Analyze attention in sampled vs unsampled regions
        sampled_regions = ring_mask_resized > 0.5
        unsampled_regions = ring_mask_resized <= 0.5
        
        sampled_attention = np.mean(kspace_attention[sampled_regions])
        unsampled_attention = np.mean(kspace_attention[unsampled_regions])
        
        # Correlation coefficient
        correlation = np.corrcoef(ring_mask_resized.flatten(), 
                                 kspace_attention.flatten())[0, 1]
        
        return {
            'sampled_attention': sampled_attention,
            'unsampled_attention': unsampled_attention,
            'attention_ratio': sampled_attention / (unsampled_attention + 1e-8),
            'mask_correlation': correlation,
            'ring_mask_resized': ring_mask_resized
        }
    
    def _analyze_radial_frequencies(self, kspace_attention):
        """Analyze attention as a function of radial frequency."""
        H, W = kspace_attention.shape
        center_h, center_w = H // 2, W // 2
        
        y, x = np.ogrid[:H, :W]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Create radial profile
        max_radius = int(np.sqrt(center_h**2 + center_w**2))
        radial_profile = []
        radii = []
        
        for r in range(0, max_radius, 2):  # Sample every 2 pixels
            mask = (distances >= r) & (distances < r + 2)
            if np.any(mask):
                radial_profile.append(np.mean(kspace_attention[mask]))
                radii.append(r)
        
        return {
            'radii': np.array(radii),
            'radial_profile': np.array(radial_profile),
            'distance_map': distances
        }
    
    def _analyze_sampling_efficiency(self, kspace_attention, ring_mask):
        """Analyze how efficiently the model uses sampled k-space data."""
        ring_mask_resized = self.ring_correlation['ring_mask_resized'] if hasattr(self, 'ring_correlation') else ring_mask
        
        # Find ring boundaries (edges of sampled regions)
        ring_edges = ndimage.sobel(ring_mask_resized.astype(float))
        ring_edges = ring_edges > 0.1
        
        # Attention at ring boundaries vs centers
        edge_attention = np.mean(kspace_attention[ring_edges]) if np.any(ring_edges) else 0
        
        # Sample several rings and analyze attention patterns
        sampled_regions = ring_mask_resized > 0.5
        center_attention = np.mean(kspace_attention[sampled_regions]) if np.any(sampled_regions) else 0
        
        return {
            'edge_attention': edge_attention,
            'center_attention': center_attention,
            'edge_to_center_ratio': edge_attention / (center_attention + 1e-8),
            'ring_edges': ring_edges
        }
    
    def remove_hooks(self):
        """Clean up hooks and memory."""
        for handle in self.hook_handles:
            handle.remove()
        clear_gpu_memory()


def create_kspace_visualization(results, input_img, target_img, pred_img, ring_mask, ssim, save_path):
    """
    Create comprehensive k-space focused visualization.
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create grid layout
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Row 1: Original images
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
    ax4.set_title('Ring Mask\n(K-space Sampling)')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(results['image_attention'], cmap='jet')
    ax5.set_title('Final Layer Attention\n(Image Space)')
    ax5.axis('off')
    
    # Row 2: K-space analysis
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.imshow(results['kspace_attention'], cmap='jet')
    ax6.set_title('K-space Attention\n(Frequency Domain)')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 1])
    # Show frequency bands
    freq_masks = results['frequency_analysis']['frequency_masks']
    combined_mask = np.zeros_like(results['kspace_attention'])
    combined_mask[freq_masks['low']] = 1    # Red for low freq
    combined_mask[freq_masks['mid']] = 0.5  # Green for mid freq  
    combined_mask[freq_masks['high']] = 0.2 # Blue for high freq
    ax7.imshow(combined_mask, cmap='RdYlBu_r')
    ax7.set_title('Frequency Bands\nRed=Low, Yellow=Mid, Blue=High')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 2])
    # Ring mask correlation
    ring_corr = results['ring_correlation']['ring_mask_resized']
    ax8.imshow(ring_corr, cmap='gray', alpha=0.7)
    ax8.imshow(results['kspace_attention'], cmap='jet', alpha=0.5)
    ax8.set_title(f'Attention vs Ring Mask\nCorrelation: {results["ring_correlation"]["mask_correlation"]:.3f}')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[1, 3])
    # Radial profile
    radial = results['radial_analysis']
    ax9.plot(radial['radii'], radial['radial_profile'], 'b-', linewidth=2)
    ax9.set_xlabel('Radial Distance (pixels)')
    ax9.set_ylabel('Attention Importance')
    ax9.set_title('Radial Frequency Profile')
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[1, 4])
    # Frequency band pie chart
    freq_analysis = results['frequency_analysis']
    sizes = [freq_analysis['low_freq_percentage'], 
             freq_analysis['mid_freq_percentage'],
             freq_analysis['high_freq_percentage']]
    labels = ['Low Freq\n(Structure)', 'Mid Freq\n(Anatomy)', 'High Freq\n(Details)']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax10.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax10.set_title('Frequency Importance')
    
    # Row 3: Advanced k-space analysis
    ax11 = fig.add_subplot(gs[2, 0])
    # Sampling efficiency
    sampling = results['sampling_analysis']
    edge_map = sampling['ring_edges']
    ax11.imshow(edge_map, cmap='Reds', alpha=0.8)
    ax11.imshow(results['kspace_attention'], cmap='jet', alpha=0.4)
    ax11.set_title(f'Ring Edge Analysis\nEdge/Center Ratio: {sampling["edge_to_center_ratio"]:.2f}')
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[2, 1])
    # Attention distribution histogram
    attention_flat = results['kspace_attention'].flatten()
    ax12.hist(attention_flat, bins=50, alpha=0.7, color='steelblue')
    ax12.axvline(np.mean(attention_flat), color='red', linestyle='--', label=f'Mean: {np.mean(attention_flat):.3f}')
    ax12.set_xlabel('Attention Value')
    ax12.set_ylabel('Frequency')
    ax12.set_title('K-space Attention Distribution')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # Row 4: Statistics and insights
    ax13 = fig.add_subplot(gs[3, :])
    
    # Comprehensive statistics
    ring_corr = results['ring_correlation']
    freq_analysis = results['frequency_analysis']
    
    stats_text = f"""
    K-SPACE GRAD-CAM ANALYSIS - FINAL LAYER (model.final_conv)
    {'='*80}

    🎯 LAYER ANALYZED: Final convolution layer (64 → 1 channels) - DIRECTLY GENERATES OUTPUT!

    📊 FREQUENCY ANALYSIS:
    • Low Freq (Structure):  {freq_analysis['low_freq_percentage']:.1f}% importance
    • Mid Freq (Anatomy):    {freq_analysis['mid_freq_percentage']:.1f}% importance  
    • High Freq (Details):   {freq_analysis['high_freq_percentage']:.1f}% importance

    ⭕ RING MASK CORRELATION:
    • Sampled regions attention:    {ring_corr['sampled_attention']:.3f}
    • Unsampled regions attention:  {ring_corr['unsampled_attention']:.3f}
    • Attention ratio (sampled/unsampled): {ring_corr['attention_ratio']:.2f}
    • Mask correlation coefficient: {ring_corr['mask_correlation']:.3f}

    🌊 K-SPACE INSIGHTS:
    • Peak k-space attention: {results['kspace_attention'].max():.3f}
    • Mean k-space attention: {results['kspace_attention'].mean():.3f}
    • Attention spread (std):  {results['kspace_attention'].std():.3f}

    💡 CLINICAL INTERPRETATION:
    • Model focuses most on: {"Low frequencies (structure)" if freq_analysis['low_freq_percentage'] > 50 else "Mid frequencies (anatomy)" if freq_analysis['mid_freq_percentage'] > 40 else "Mixed frequency approach"}
    • Sampling efficiency: {"Good - focuses on sampled regions" if ring_corr['attention_ratio'] > 1.2 else "Moderate - mixed attention" if ring_corr['attention_ratio'] > 0.8 else "Poor - ignores sampled data"}
    • Ring mask usage: {"Strong correlation" if abs(ring_corr['mask_correlation']) > 0.3 else "Moderate correlation" if abs(ring_corr['mask_correlation']) > 0.1 else "Weak correlation"}
    """
    
    ax13.text(0.02, 0.98, stats_text, transform=ax13.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax13.axis('off')
    
    plt.suptitle('K-space Grad-CAM Analysis: Final Layer Focus', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"K-space analysis saved to: {save_path}")

def analyze_final_layer_kspace_only(model, X_for_gradcam, device, tg, zf, pred, ssim, psnr, 
                                  ring_mask_path, exp_path):
    """
    Complete k-space focused analysis of the very final layer.
    
    Args:
        model: U-Net model
        X_for_gradcam: Input tensor for analysis
        device: CUDA device
        tg: Ground truth image
        zf: Zero-filled image  
        pred: Predicted reconstruction
        ssim: SSIM score
        psnr: PSNR score
        ring_mask_path: Path to ring mask .npy file
        exp_path: Experiment path for saving results
    """
    print("K-SPACE GRAD-CAM ANALYSIS - FINAL LAYER ONLY")
    print("="*60)
    
    # Load ring mask
    ring_mask = np.load(ring_mask_path)
    print(f"Loaded ring mask: {ring_mask.shape}")
    
    try:
        # Initialize k-space Grad-CAM
        kspace_gradcam = KSpaceGradCAM(model, device)
        
        # Perform k-space analysis
        results = kspace_gradcam.compute_kspace_gradcam(X_for_gradcam, ring_mask)
        
        # Create k-space visualization
        create_kspace_visualization(
            results=results,
            input_img=zf[0].cpu().numpy() if torch.is_tensor(zf) else zf,
            target_img=tg[0].cpu().numpy() if torch.is_tensor(tg) else tg,
            pred_img=pred[0].cpu().numpy() if torch.is_tensor(pred) else pred,
            ring_mask=ring_mask,
            ssim=ssim,
            save_path=exp_path / "XAI" / "kspace_gradcam_final_layer.png"
        )
        
        # Print key insights
        print("\nKEY K-SPACE INSIGHTS:")
        freq_analysis = results['frequency_analysis']
        ring_corr = results['ring_correlation']
        
        print(f"   • Model focuses most on: {['Low frequencies (structure)', 'Mid frequencies (anatomy)', 'High frequencies (details)'][np.argmax([freq_analysis['low_freq_percentage'], freq_analysis['mid_freq_percentage'], freq_analysis['high_freq_percentage']])]}")
        print(f"   • Ring mask correlation: {ring_corr['mask_correlation']:.3f}")
        print(f"   • Sampling efficiency: {ring_corr['attention_ratio']:.2f}x more attention on sampled regions")
        
        # Cleanup
        kspace_gradcam.remove_hooks()
        
        return results
        
    except Exception as e:
        print(f"K-space analysis failed: {e}")
        if 'kspace_gradcam' in locals():
            kspace_gradcam.remove_hooks()
        return None

    

