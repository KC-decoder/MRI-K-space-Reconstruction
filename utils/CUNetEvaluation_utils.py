import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

def load_cunet_model(model_path, device, base_features=32, use_data_consistency=False):
    """
    Load a trained CU-Net model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: Device to load the model on
        base_features: Base features used in the model architecture
        use_data_consistency: Whether the model uses data consistency
    
    Returns:
        Loaded CU-Net model
    """
    from net.unet.complex_Unet import CUNet  # Import your CU-Net class
    
    # Create model with same architecture
    model = CUNet(
        in_channels=2, 
        out_channels=1, 
        base_features=base_features,
        use_data_consistency=use_data_consistency
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print(f"Model loaded from: {model_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.6f}")
    
    return model

def kspace_to_image_magnitude(kspace_tensor):
    """
    Convert undersampled k-space tensor to zero-filled reconstruction magnitude image.
    
    Args:
        kspace_tensor: Undersampled complex k-space tensor (B, 2, H, W) where dim 1 is [real, imag]
                      This should be the undersampled k-space (with zeros at unsampled locations)
    
    Returns:
        Zero-filled reconstruction magnitude image tensor (B, 1, H, W)
    """
    batch_size, channels, H, W = kspace_tensor.shape
    
    if channels != 2:
        raise ValueError(f"Expected 2 channels (real, imag), got {channels}")
    
    # Split real and imaginary parts
    real = kspace_tensor[:, 0, :, :]  # (B, H, W)
    imag = kspace_tensor[:, 1, :, :]  # (B, H, W)
    
    # Create complex tensor
    complex_kspace = torch.complex(real, imag)  # (B, H, W)
    
    # Apply 2D IFFT to get zero-filled reconstruction
    # The input k-space should already have zeros at unsampled locations
    complex_image = torch.fft.ifft2(complex_kspace, norm='ortho')
    
    # Get magnitude - this is the zero-filled reconstruction
    magnitude = torch.abs(complex_image)  # (B, H, W)
    
    return magnitude.unsqueeze(1)  # (B, 1, H, W)

def _to_gray2d(img_array):
    """
    Convert various image formats to 2D grayscale for display.
    Handles: (H,W), (1,H,W), (2,H,W), (H,W,1), (H,W,2)
    """
    if img_array.ndim == 2:
        return img_array
    elif img_array.ndim == 3:
        if img_array.shape[0] == 1:  # (1,H,W)
            return img_array[0]
        elif img_array.shape[0] == 2:  # (2,H,W) - complex, take magnitude
            real, imag = img_array[0], img_array[1]
            return np.sqrt(real**2 + imag**2)
        elif img_array.shape[-1] == 1:  # (H,W,1)
            return img_array[:, :, 0]
        elif img_array.shape[-1] == 2:  # (H,W,2) - complex, take magnitude
            real, imag = img_array[:, :, 0], img_array[:, :, 1]
            return np.sqrt(real**2 + imag**2)
        else:
            # Take first channel as fallback
            return img_array[0] if img_array.shape[0] < img_array.shape[-1] else img_array[:, :, 0]
    else:
        raise ValueError(f"Unexpected image dimensions: {img_array.shape}")

def _normalize(img_array, percentile=99):
    """Normalize image for display using percentile normalization."""
    vmax = np.percentile(img_array, percentile)
    vmin = np.percentile(img_array, 100 - percentile)
    return np.clip((img_array - vmin) / (vmax - vmin + 1e-8), 0, 1)

def normalize_image(img_tensor, percentile=99):
    """
    Normalize image tensor for visualization using the same approach as visualize_data_sample.
    
    Args:
        img_tensor: Image tensor (B, 1, H, W) or (H, W)
        percentile: Percentile for normalization
    
    Returns:
        Normalized image array
    """
    img_np = img_tensor.detach().cpu().numpy().squeeze()
    
    # Use the same normalization as _normalize function
    return _normalize(img_np, percentile)

def get_zero_filled_reconstruction(kspace_data):
    """
    Get zero-filled reconstruction from undersampled k-space data.
    This should match what your dataloader provides as the undersampled input.
    
    Args:
        kspace_data: Undersampled k-space tensor (B, 2, H, W)
    
    Returns:
        Zero-filled reconstruction (B, 1, H, W)
    """
    # The k-space data should already be undersampled (zeros at unsampled locations)
    return kspace_to_image_magnitude(kspace_data)

def calculate_image_metrics(pred_img, target_img):
    """
    Calculate NMSE, PSNR, and SSIM for single images.
    
    Args:
        pred_img: Predicted image (H, W)
        target_img: Target image (H, W)
    
    Returns:
        Dictionary with metrics
    """
    # Ensure images are numpy arrays
    if torch.is_tensor(pred_img):
        pred_img = pred_img.detach().cpu().numpy().squeeze()
    if torch.is_tensor(target_img):
        target_img = target_img.detach().cpu().numpy().squeeze()
    
    # NMSE
    mse = np.mean((pred_img - target_img) ** 2)
    target_var = np.var(target_img)
    nmse = mse / target_var if target_var > 0 else float('inf')
    
    # PSNR
    data_range = target_img.max() - target_img.min()
    psnr = psnr_metric(target_img, pred_img, data_range=data_range)
    
    # SSIM
    ssim_val = ssim_metric(target_img, pred_img, data_range=data_range)
    
    return {
        'NMSE': nmse,
        'PSNR': psnr,
        'SSIM': ssim_val
    }

def evaluate_and_visualize_cunet(
    model_path, 
    dataloader, 
    device, 
    save_dir=None,
    num_samples=5,
    base_features=32,
    use_data_consistency=False,
    figsize=(15, 5)):
    """
    Evaluate CU-Net model and create visualizations with correct zero-filled reconstruction.
    
    Args:
        model_path: Path to the trained model checkpoint
        dataloader: DataLoader with test data
        device: Device to run inference on
        save_dir: Directory to save plots (if None, just display)
        num_samples: Number of samples to visualize
        base_features: Base features used in model architecture
        use_data_consistency: Whether model uses data consistency
        figsize: Figure size for each plot
    
    Returns:
        Dictionary with average metrics
    """
    # Load model
    model = load_cunet_model(model_path, device, base_features, use_data_consistency)
    
    # Create save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {save_dir}")
    
    # Collect metrics
    all_metrics = {'NMSE': [], 'PSNR': [], 'SSIM': []}
    
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            X, y, mask = data
            X = X.to(device).float()      # Undersampled k-space (B, 2, H, W)
            y = y.to(device).float()      # Target magnitude image (B, 1, H, W)
            mask = mask.to(device).float() # Undersampling mask (B, 1, H, W)
            
            # Get model prediction
            y_pred = model(X, mask)  # Reconstructed magnitude image (B, 1, H, W)
            
            # Get zero-filled reconstruction from undersampled k-space
            # The k-space X should already be undersampled (with zeros at unsampled locations)
            zero_filled_recon = get_zero_filled_reconstruction(X)  # (B, 1, H, W)
            
            # Process each image in the batch
            batch_size = X.shape[0]
            for b in range(batch_size):
                if idx * batch_size + b >= num_samples:
                    break
                
                # Extract single images
                zero_filled_img = zero_filled_recon[b, 0]  # (H, W) - zero-filled reconstruction
                recon_img = y_pred[b, 0]                   # (H, W) - CU-Net reconstruction  
                target_img = y[b, 0]                       # (H, W) - fully sampled target
                
                # Calculate metrics (compare CU-Net reconstruction to target)
                metrics = calculate_image_metrics(recon_img, target_img)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                # Also calculate metrics for zero-filled vs target for comparison
                zf_metrics = calculate_image_metrics(zero_filled_img, target_img)
                
                # Normalize images for visualization using same method as visualize_data_sample
                zero_filled_norm = normalize_image(zero_filled_img)
                recon_norm = normalize_image(recon_img)
                target_norm = normalize_image(target_img)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=figsize)
                
                # Zero-filled reconstruction (undersampled input)
                im1 = axes[0].imshow(zero_filled_norm, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title(f'Zero-filled Reconstruction\n(Undersampled Input)\nSSIM: {zf_metrics["SSIM"]:.3f}')
                axes[0].axis('off')
                
                # CU-Net reconstruction
                im2 = axes[1].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title(f'CU-Net Reconstruction\nSSIM: {metrics["SSIM"]:.3f}')
                axes[1].axis('off')
                
                # Ground truth
                im3 = axes[2].imshow(target_norm, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('Fully Sampled\nGround Truth')
                axes[2].axis('off')
                
                # Add metrics as text
                metrics_text = f'CU-Net: NMSE: {metrics["NMSE"]:.4f} | PSNR: {metrics["PSNR"]:.2f} dB | SSIM: {metrics["SSIM"]:.3f}'
                zf_text = f'Zero-filled: NMSE: {zf_metrics["NMSE"]:.4f} | PSNR: {zf_metrics["PSNR"]:.2f} dB | SSIM: {zf_metrics["SSIM"]:.3f}'
                
                fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=9, weight='bold')
                fig.text(0.5, -0.01, zf_text, ha='center', fontsize=9, style='italic')
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.1)
                
                # Save or show
                if save_dir:
                    plt.savefig(save_dir / f'reconstruction_sample_{idx * batch_size + b + 1}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    print("\nCU-Net Average Metrics:")
    print(f"NMSE: {avg_metrics['NMSE']:.6f}")
    print(f"PSNR: {avg_metrics['PSNR']:.2f} dB")
    print(f"SSIM: {avg_metrics['SSIM']:.4f}")
    
    return avg_metrics

def create_comparison_grid(
    model_path,
    dataloader,
    device,
    save_path=None,
    sample_indices=[0, 1, 2],
    base_features=32,
    use_data_consistency=False):
    """
    Create a comparison grid showing multiple samples in one figure with correct zero-filled reconstruction.
    
    Args:
        model_path: Path to trained model
        dataloader: DataLoader with test data
        device: Device for inference
        save_path: Path to save the comparison grid
        sample_indices: List of sample indices to include
        base_features: Model base features
        use_data_consistency: Whether model uses data consistency
    """
    # Load model
    model = load_cunet_model(model_path, device, base_features, use_data_consistency)
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    all_data = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx in sample_indices:
                all_data.append(data)
            if len(all_data) >= num_samples:
                break
    
    for plot_idx, data in enumerate(all_data):
        X, y, mask = data
        X = X.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device).float()
        
        # Get prediction
        y_pred = model(X, mask)
        
        # Get zero-filled reconstruction
        zero_filled_recon = get_zero_filled_reconstruction(X)
        
        # Take first image from batch and normalize using correct method
        zero_filled_norm = normalize_image(zero_filled_recon[0, 0])
        recon_norm = normalize_image(y_pred[0, 0])
        target_norm = normalize_image(y[0, 0])
        
        # Calculate metrics
        zf_metrics = calculate_image_metrics(zero_filled_recon[0, 0], y[0, 0])
        cunet_metrics = calculate_image_metrics(y_pred[0, 0], y[0, 0])
        
        # Plot images
        axes[plot_idx, 0].imshow(zero_filled_norm, cmap='gray')
        axes[plot_idx, 0].set_title(f'Sample {sample_indices[plot_idx] + 1}\nZero-filled (SSIM: {zf_metrics["SSIM"]:.3f})')
        axes[plot_idx, 0].axis('off')
        
        axes[plot_idx, 1].imshow(recon_norm, cmap='gray')
        axes[plot_idx, 1].set_title(f'CU-Net Reconstruction\n(SSIM: {cunet_metrics["SSIM"]:.3f})')
        axes[plot_idx, 1].axis('off')
        
        axes[plot_idx, 2].imshow(target_norm, cmap='gray')
        axes[plot_idx, 2].set_title('Ground Truth')
        axes[plot_idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to: {save_path}")
    else:
        plt.show()

def validate_zero_filled_reconstruction(dataloader, device, sample_idx=0):
    """
    Validate that our zero-filled reconstruction matches what it should look like.
    This function helps verify that the k-space to image conversion is correct.
    
    Args:
        dataloader: Test dataloader
        device: Device for computation
        sample_idx: Which sample to validate
    """
    print("Validating zero-filled reconstruction...")
    
    # Get one batch of data
    X, y, mask = next(iter(dataloader))
    X = X.to(device).float()
    y = y.to(device).float()
    mask = mask.to(device).float()
    
    # Get zero-filled reconstruction using our function
    zero_filled = get_zero_filled_reconstruction(X)
    
    # Extract sample
    zf_img = zero_filled[sample_idx, 0].detach().cpu().numpy()
    target_img = y[sample_idx, 0].detach().cpu().numpy()
    mask_img = mask[sample_idx, 0].detach().cpu().numpy()
    
    # Normalize for display
    zf_norm = _normalize(zf_img)
    target_norm = _normalize(target_img)
    
    # Calculate metrics
    metrics = calculate_image_metrics(zf_img, target_img)
    
    # Create validation plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(zf_norm, cmap='gray')
    axes[0].set_title(f'Zero-filled Reconstruction\nSSIM: {metrics["SSIM"]:.3f}')
    axes[0].axis('off')
    
    axes[1].imshow(target_norm, cmap='gray')
    axes[1].set_title('Fully Sampled Target')
    axes[1].axis('off')
    
    axes[2].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('k-space Mask')
    axes[2].axis('off')
    
    plt.suptitle(f'Validation: Zero-filled vs Target (NMSE: {metrics["NMSE"]:.4f}, PSNR: {metrics["PSNR"]:.2f} dB)')
    plt.tight_layout()
    plt.show()
    
    print(f"Zero-filled reconstruction metrics:")
    print(f"  NMSE: {metrics['NMSE']:.6f}")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB") 
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print("\nThe zero-filled reconstruction should look like a degraded version of the target.")
    print("If it looks very different or noisy, there may be an issue with the k-space data.")
    
    return metrics

# Example usage functions
def quick_evaluate(model_path, dataloader, device, num_samples=3):
    """
    Quick evaluation and visualization function with correct zero-filled reconstruction.
    
    Args:
        model_path: Path to trained CU-Net model
        dataloader: Test dataloader
        device: Device for inference
        num_samples: Number of samples to visualize
    """
    print(f"Evaluating CU-Net model: {model_path}")
    
    # First validate that zero-filled reconstruction looks correct
    print("\n1. Validating zero-filled reconstruction...")
    validate_zero_filled_reconstruction(dataloader, device, sample_idx=0)
    
    # Run evaluation and visualization
    print("\n2. Running CU-Net evaluation...")
    metrics = evaluate_and_visualize_cunet(
        model_path=model_path,
        dataloader=dataloader,
        device=device,
        num_samples=num_samples,
        save_dir=None  # Display plots instead of saving
    )
    
    return metrics

def evaluate_and_save(model_path, dataloader, device, output_dir, num_samples=10):
    """
    Evaluate model and save all plots to directory.
    
    Args:
        model_path: Path to trained CU-Net model
        dataloader: Test dataloader  
        device: Device for inference
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to evaluate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating CU-Net model: {model_path}")
    print(f"Saving results to: {output_dir}")
    
    # Run evaluation and save individual plots
    metrics = evaluate_and_visualize_cunet(
        model_path=model_path,
        dataloader=dataloader,
        device=device,
        num_samples=num_samples,
        save_dir=output_dir / "individual_plots"
    )
    
    # Create comparison grid
    create_comparison_grid(
        model_path=model_path,
        dataloader=dataloader,
        device=device,
        save_path=output_dir / "comparison_grid.png",
        sample_indices=[0, 1, 2, 3, 4]
    )
    
    # Save metrics to file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("CU-Net Evaluation Metrics\n")
        f.write("=" * 25 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Samples evaluated: {num_samples}\n\n")
        f.write(f"NMSE: {metrics['NMSE']:.6f}\n")
        f.write(f"PSNR: {metrics['PSNR']:.2f} dB\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
    
    print("Evaluation complete!")
    return metrics

# # Example usage
# if __name__ == "__main__":
#     # Example usage
#     model_path = "/path/to/your/trained_model.pt"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # IMPORTANT: First validate that zero-filled reconstruction is correct
#     print("=== Validating Zero-filled Reconstruction ===")
#     # validate_zero_filled_reconstruction(test_dataloader, device)
    
#     # Quick evaluation (displays plots)
#     print("\n=== Quick Evaluation ===") 
#     # metrics = quick_evaluate(model_path, test_dataloader, device, num_samples=3)
    
#     # Full evaluation with saved results
#     print("\n=== Full Evaluation ===")
#     # metrics = evaluate_and_save(model_path, test_dataloader, device, 
#     #                           output_dir="./evaluation_results", num_samples=10)
    
    pass