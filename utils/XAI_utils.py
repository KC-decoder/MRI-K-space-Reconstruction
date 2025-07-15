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
import torch.nn.functional as F
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

    

