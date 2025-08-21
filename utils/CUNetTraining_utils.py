import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from utils.visualize_utils import visualize_kspace_sample

def get_latest_checkpoint(models_path, logger):
    """
    Get the latest checkpoint from the models directory.
    """
    ckpt_files = list(models_path.glob("model_ck*.pt"))
    if not ckpt_files:
        return None
    
    # Sort by epoch number
    ckpt_files.sort(key=lambda x: int(x.stem.split("model_ck")[1]))
    latest_ckpt = ckpt_files[-1]
    return latest_ckpt

def calculate_metrics(pred_img, target_img):
    """
    Calculate NMSE, PSNR, and SSIM metrics.
    
    Args:
        pred_img: Predicted image tensor (B, 1, H, W)
        target_img: Target image tensor (B, 1, H, W)
    
    Returns:
        nmse, psnr, ssim values
    """
    # Convert to numpy and remove batch/channel dimensions
    pred_np = pred_img.detach().cpu().numpy().squeeze()
    target_np = target_img.detach().cpu().numpy().squeeze()
    
    # Handle batch dimension
    if pred_np.ndim == 3:  # Multiple images in batch
        nmse_vals, psnr_vals, ssim_vals = [], [], []
        for i in range(pred_np.shape[0]):
            pred_single = pred_np[i]
            target_single = target_np[i]
            
            # NMSE
            mse = np.mean((pred_single - target_single) ** 2)
            target_var = np.var(target_single)
            nmse = mse / target_var if target_var > 0 else float('inf')
            nmse_vals.append(nmse)
            
            # PSNR
            data_range = target_single.max() - target_single.min()
            psnr = psnr_metric(target_single, pred_single, data_range=data_range)
            psnr_vals.append(psnr)
            
            # SSIM
            ssim_val = ssim_metric(target_single, pred_single, data_range=data_range)
            ssim_vals.append(ssim_val)
        
        return np.mean(nmse_vals), np.mean(psnr_vals), np.mean(ssim_vals)
    
    else:  # Single image
        # NMSE
        mse = np.mean((pred_np - target_np) ** 2)
        target_var = np.var(target_np)
        nmse = mse / target_var if target_var > 0 else float('inf')
        
        # PSNR
        data_range = target_np.max() - target_np.min()
        psnr = psnr_metric(target_np, pred_np, data_range=data_range)
        
        # SSIM
        ssim_val = ssim_metric(target_np, pred_np, data_range=data_range)
        
        return nmse, psnr, ssim_val

def test_cunet(test_dataloader, net, device, logger,loss_fn, num_test_samples=None):
    """
    Test CU-Net and calculate metrics (after de-normalizing if scale is provided).
    """
    net.eval()
    total_nmse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    logger.log("Starting CU-Net evaluation...")
    
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            if len(data) == 4:
                X, y, mask, _ = data
            else:
                X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()
            mask = mask.to(device).float()

            # normalize input; keep target scale so we can denorm for metrics
            X_n, _ = minmax_norm(X)
            y_n, (y_min, y_scale) = minmax_norm(y)

            y_pred_n = net(X_n, mask)

            # validation loss (normalized space) – this is the one to log/compare
            val_loss = loss_fn(y_pred_n, y_n)

            # de-normalize for metrics/plots
            y_pred = denorm(y_pred_n, y_min, y_scale)
            y_true = denorm(y_n,     y_min, y_scale)

            # now compute PSNR/SSIM/NMSE on y_pred vs y_true (original scale)
            nmse, psnr, ssim_val = calculate_metrics(y_pred, y_true)

            total_nmse += nmse
            total_psnr += psnr
            total_ssim += ssim_val
            num_samples += 1
    
    avg_nmse = total_nmse / num_samples if num_samples > 0 else float('inf')
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
    
    logger.log(f"Test Results ({num_samples} samples):")
    logger.log(f"Test loss: {val_loss:.6f}")
    # logger.log(f"  NMSE: {avg_nmse:.6f}")
    # logger.log(f"  PSNR: {avg_psnr:.2f} dB")
    # logger.log(f"  SSIM: {avg_ssim:.4f}")
    
    return avg_nmse, avg_psnr, avg_ssim



def minmax_norm(x, dims=(2,3), eps=1e-8):
    # per-sample min/max over H,W (keeps B,C)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    scale = (x_max - x_min).clamp_min(eps)
    x_n = (x - x_min) / scale
    return x_n, (x_min, scale)

def denorm(x_n, x_min, scale):
    return x_n * scale + x_min


def train_cunet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_fn,  # CUNetLoss instance
        net,      # CUNet instance
        scheduler,
        device,
        logger,
        PATH_MODEL,
        NUM_EPOCH=5,
        save_every=500,
        show_test=False,
        use_data_consistency=False):
    '''
    Train the CU-Net with resume-from-checkpoint support.
    
    Args:
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        optimizer: Optimizer (e.g., Adam)
        loss_fn: CUNetLoss instance
        net: CUNet model instance
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Logger for output
        PATH_MODEL: Path to save models and logs
        NUM_EPOCH: Number of epochs to train
        save_every: Save checkpoint every N epochs
        show_test: Whether to run test after training
        use_data_consistency: Whether to use data consistency in CU-Net
    
    Returns:
        Trained CU-Net model
    '''
    # Move model to device
    net = net.to(device)
    
    # Setup paths
    PATH_MODEL = Path(PATH_MODEL)
    MODELS_PATH = PATH_MODEL / "models"
    LOGS_PATH = PATH_MODEL / "logs"
    VIZ_PATH = PATH_MODEL / "VISUALIZATIONS"
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    VIZ_PATH.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    logger.log(f"Vizualizing sample from inside training function")
    visualize_kspace_sample(train_dataloader, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    

    logger.log(f"Training CU-Net for {NUM_EPOCH} epochs")
    logger.log(f"Model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    logger.log(f"Using data consistency: {use_data_consistency}")
    logger.log(f"Models will be saved to: {MODELS_PATH}")

    # Load latest checkpoint if exists
    start_epoch = 0
    latest_ckpt = get_latest_checkpoint(MODELS_PATH, logger)
    if latest_ckpt:
        logger.log(f"Loading checkpoint from {latest_ckpt}")
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = int(latest_ckpt.stem.split("model_ck")[1])
            logger.log(f"Successfully resumed from epoch {start_epoch}")
        except Exception as e:
            logger.log(f"Error loading checkpoint: {e}")
            logger.log("Starting training from scratch")
            start_epoch = 0

    # Training loop
    pbar = tqdm(range(start_epoch, NUM_EPOCH), desc="Training CU-Net")
    best_loss = float('inf')
    
    for epoch in pbar:
        net.train()
        running_loss = 0.0
        epoch_losses = []

        for idx, data in enumerate(train_dataloader):
            if len(data) == 4:
                X, y, mask, _ = data
            else:
                X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()
            mask = mask.to(device).float()

            # Normalize inputs/targets (independently is fine)
            X_n, _ = minmax_norm(X)          # you typically don't need X scale later
            y_n, _ = minmax_norm(y)          # train loss in normalized space

            y_pred_n = net(X_n, mask)
            loss_train = loss_fn(y_pred_n, y_n)

            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # Track loss
            loss_value = loss_train.item()
            running_loss += loss_value
            epoch_losses.append(loss_value)
            
            # Update progress bar
            if idx % 10 == 0:  # Update every 10 batches
                pbar.set_postfix({
                    'Loss': f'{loss_value:.6f}',
                    'Avg': f'{running_loss/(idx+1):.6f}'
                })

        # Calculate epoch statistics
        avg_loss = running_loss / len(train_dataloader)
        epoch_std = np.std(epoch_losses) if len(epoch_losses) > 1 else 0.0
        
        # Update learning rate scheduler
        if scheduler:
            if hasattr(scheduler, 'step'):
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Log epoch results
        logger.log(f"Epoch {epoch + 1}/{NUM_EPOCH}")
        logger.log(f"  Average Loss: {avg_loss:.6f} ± {epoch_std:.6f}")
        # logger.log(f"  Learning Rate: {current_lr:.2e}")
        
        # Update progress bar
        pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCH} | Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or avg_loss < best_loss:
            checkpoint_data = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save regular checkpoint
            if (epoch + 1) % save_every == 0:
                torch.save(checkpoint_data, MODELS_PATH / f'model_ck{epoch + 1}.pt')
                logger.log(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint_data, MODELS_PATH / 'model_best.pt')
                logger.log(f"Best model saved at epoch {epoch + 1} with loss {avg_loss:.6f}")

        # # Run validation periodically
        # if (epoch + 1) % max(1, save_every // 5) == 0 and test_dataloader:
        #     logger.log("Running validation...")
        #     nmse, psnr, ssim_val = test_cunet(test_dataloader, net, device, logger, num_test_samples=5)
        #     logger.log(f"Validation at epoch {epoch + 1}: NMSE={nmse:.4f}, PSNR={psnr:.2f}, SSIM={ssim_val:.4f}")
        #     net.train()  # Switch back to training mode

    # Final test after training
    if show_test and test_dataloader:
        logger.log("Running final evaluation...")
        nmse, psnr, ssim_val = test_cunet(test_dataloader, net, device, logger)
        logger.log(f"Final Test Results:")
        logger.log(f"  NMSE: {nmse:.6f}")
        logger.log(f"  PSNR: {psnr:.2f} dB")
        logger.log(f"  SSIM: {ssim_val:.4f}")

    # Save final model
    final_checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': NUM_EPOCH,
        'final_loss': avg_loss,
    }
    if scheduler:
        final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(final_checkpoint, MODELS_PATH / 'model_final.pt')
    logger.log(f'Final model saved in {MODELS_PATH}/model_final.pt')
    logger.log(f'Best model saved in {MODELS_PATH}/model_best.pt')

    return net

def load_cunet_checkpoint(model, checkpoint_path, device, logger):
    """
    Load a CU-Net checkpoint.
    
    Args:
        model: CUNet instance
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        logger: Logger for output
    
    Returns:
        Loaded model
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        
        logger.log(f"Loaded checkpoint from {checkpoint_path}")
        logger.log(f"  Epoch: {epoch}")
        logger.log(f"  Loss: {loss}")
        
        return model
    except Exception as e:
        logger.log(f"Error loading checkpoint: {e}")
        return model

# Example usage function
def setup_cunet_training(
    train_dataloader, 
    test_dataloader, 
    device, 
    logger, 
    PATH_MODEL,
    base_features=32,
    learning_rate=1e-4,
    use_data_consistency=False):
    """
    Setup CU-Net training with default parameters.
    
    Returns:
        model, optimizer, scheduler, loss_fn ready for training
    """
    from net.unet.complex_Unet import CUNet , CUNetLoss
    
    # Create model
    model = CUNet(
        in_channels=2, 
        out_channels=1, 
        base_features=base_features,
        use_data_consistency=use_data_consistency
    ).to(device)
    
    # Create loss function
    loss_fn = CUNetLoss(alpha=1.0, beta=0.1)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0
    )
    
    # Create scheduler
    scheduler = None
    
    logger.log("CU-Net training setup complete:")
    logger.log(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.log(f"  Base features: {base_features}")
    logger.log(f"  Learning rate: {learning_rate}")
    logger.log(f"  Data consistency: {use_data_consistency}")
    
    return model, optimizer, scheduler, loss_fn

# # Example training script
# if __name__ == "__main__":
#     # Example usage (you would replace these with your actual data loaders)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Setup training
#     model, optimizer, scheduler, loss_fn = setup_cunet_training(
#         train_dataloader=None,  # Your train dataloader
#         test_dataloader=None,   # Your test dataloader
#         device=device,
#         logger=None,            # Your logger
#         PATH_MODEL="./checkpoints",
#         base_features=32,
#         learning_rate=1e-4,
#         use_data_consistency=False
#     )
    
    # Train the model
    # trained_model = train_cunet(
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     net=model,
    #     scheduler=scheduler,
    #     device=device,
    #     logger=logger,
    #     PATH_MODEL="./checkpoints",
    #     NUM_EPOCH=100,
    #     save_every=10,
    #     show_test=True
    # )