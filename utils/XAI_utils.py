import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastmri
from utils.testing_utils import recon_kspace_cold_diffusion_from_perturbed_data
from diffusion.kspace_diffusion import mask_sequence_sample
from tqdm import tqdm
import torch
import torch.nn.functional as F
import fastmri
from fastmri.data import subsample, transforms, mri_data
from utils.sample_mask import RandomMaskGaussianDiffusion
import numpy as np
import matplotlib.pyplot as plt
from fastmri import fft2c, ifft2c, complex_abs


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
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register forward and backward hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, target=None):
        # Forward pass
        output = self.model(input_tensor)
        
        # If no specific target output is given, take mean of output
        if target is None:
            target = output.mean()

        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target.backward()

        # Compute weights: Global Average Pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)  # keep only positive influences
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    
    
    
    
def generate_ring_masks(H=320, W=320, num_masks=5, max_radius_fraction=0.15, save_dir="./ring_masks"):
    """
    Generates thicker binary ring masks where the total diameter of the outermost ring
    does not exceed a fixed maximum radius, and all rings have equal radial thickness.

    Args:
        H, W: height and width of the mask.
        num_masks: number of concentric rings (including central disc).
        max_radius_fraction: fraction of H to set as the max radius.
        save_dir: path to save the masks.
    """
    os.makedirs(save_dir, exist_ok=True)
    center = (H // 2, W // 2)
    max_radius = H * max_radius_fraction  # set total limit (e.g., 0.15 * 320 = 48 pixels)
    step_r = max_radius / num_masks

    y, x = np.ogrid[:H, :W]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    for i in range(num_masks):
        inner = i * step_r
        outer = (i + 1) * step_r
        ring_mask = ((distance >= inner) & (distance < outer)).astype(np.uint8)

        filename = os.path.join(save_dir, f"ring_mask_{i+1}.npy")
        np.save(filename, ring_mask)
        print(f"Saved: {filename} with shape {ring_mask.shape}")
        
        
        
def plot_ring_masks(save_dir, num_masks=5, output_path="ring_mask_grid.png"):
    """
    Plots the saved ring masks side-by-side in black and white.

    Args:
        save_dir (str): Directory containing ring_mask_*.npy files.
        num_masks (int): Number of masks to plot.
        output_path (str): Path to save the output plot.
    """
    fig, axs = plt.subplots(1, num_masks, figsize=(3 * num_masks, 3))

    for i in range(num_masks):
        mask_file = os.path.join(save_dir, f"ring_mask_{i+1}.npy")
        mask = np.load(mask_file)
        axs[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axs[i].axis('off')
        axs[i].set_title(f"Ring {i+1}")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ring mask visualization to: {output_path}")
    



    

