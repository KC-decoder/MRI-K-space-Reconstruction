import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
from net.unet.KIKI_unet import KIKI

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from debug.CUNet_debug import *
from utils.kiki_helpers import _get_autocast , _align_pred_target, _mkdir, fftshift2, load_checkpoint, save_checkpoint


from contextlib import nullcontext

class KIKIConfig:
    def __init__(self, iters, k, i, in_ch, out_ch, fm):
        self.iters = iters
        self.k = k
        self.i = i
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fm = fm


def build_model(iters,k,i,in_ch,out_ch,fm ) -> nn.Module:
    kcfg = KIKIConfig(
        iters=iters,
        k=k,
        i=i,
        in_ch=in_ch,
        out_ch=out_ch,
        fm=fm,
    )
    return KIKI(kcfg), kcfg



@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    logger=None,
    global_step: int = 0,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0

    start = time.time()
    # Use autocast in eval too (safe & faster on CUDA)
    autocast_ctx = _get_autocast(device, enabled=True)
    for batch in dataloader:
        x, y, m = batch
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        m = m.to(device, non_blocking=True).float()

        if x.dim() == 4 and x.size(1) == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        with autocast_ctx:
            pred = model(x, m)
            pred, y_ = _align_pred_target(pred, y)
            loss = loss_fn(pred, y_)

        running_loss += loss.item()
        n_batches += 1

    epoch_time = time.time() - start
    avg_loss = running_loss / max(1, n_batches)
    if logger is not None:
        logger.log(f"[val]   avg_loss={avg_loss:.6f}, batches={n_batches}, time={epoch_time:.1f}s")
        try:
            logger.log({"val/avg_loss": avg_loss, "val/epoch_time_sec": epoch_time}, step=global_step)
        except Exception:
            pass
    return avg_loss



@torch.no_grad()
def save_reconstruction_image(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: Path,
    epoch: int,
    sample_idx: int = 0,
):
    
    model.eval()
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx == sample_idx:
            x, y_target, m = batch
            break
    
    x = x.to(device).float()
    y_target = y_target.to(device).float()
    m = m.to(device).float()
    
    if x.size(1) == 1:
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    
    # Generate reconstruction
    pred = model(x, m)
    pred_mag = complex_magnitude(pred)
    if pred_mag.dim() == 3:
        pred_mag = pred_mag.unsqueeze(1)
    
    # Create zero-filled baseline using correct helper
    zero_filled_mag = create_zero_filled_baseline(x)
    
    # Convert to numpy for visualization
    y_vis = y_target[0].squeeze().cpu().numpy()
    zero_filled_vis = zero_filled_mag[0].squeeze().cpu().numpy()
    pred_vis = pred_mag[0].squeeze().cpu().numpy()
    m_vis = m[0].squeeze().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'MRI Reconstruction at Epoch {epoch} (UPDATED)', fontsize=16)
    
    # Plot all components
    im1 = axes[0,0].imshow(y_vis, cmap='gray', vmin=0, vmax=np.percentile(y_vis, 99))
    axes[0,0].set_title('Ground Truth')
    axes[0,0].axis('off')
    plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    im2 = axes[0,1].imshow(m_vis, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title('Undersampling Mask')
    axes[0,1].axis('off')
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    im3 = axes[1,0].imshow(zero_filled_vis, cmap='gray', vmin=0, vmax=np.percentile(zero_filled_vis, 99))
    axes[1,0].set_title('Zero-Filled Reconstruction')
    axes[1,0].axis('off')
    plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    im4 = axes[1,1].imshow(pred_vis, cmap='gray', vmin=0, vmax=np.percentile(pred_vis, 99))
    axes[1,1].set_title('KIKI Reconstruction')
    axes[1,1].axis('off')
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
    
    # Compute metrics
    l1_kiki = np.mean(np.abs(pred_vis - y_vis))
    l1_zero_filled = np.mean(np.abs(zero_filled_vis - y_vis))
    mse_kiki = np.mean((pred_vis - y_vis)**2)
    
    fig.text(0.02, 0.02, f'L1 KIKI: {l1_kiki:.4f} | L1 Zero-filled: {l1_zero_filled:.4f} | MSE KIKI: {mse_kiki:.4f}', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    image_filename = f"Reconstruction at epoch {epoch}.png"
    full_path = save_path / image_filename
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    return full_path

# -----------------------------
# Fit / Resume orchestration
# -----------------------------
def calculate_global_step(epoch: int, batch_idx: int, batches_per_epoch: int, start_epoch: int = 0) -> int:
    """
    Calculate global step from epoch and batch information.
    
    Args:
        epoch: Current epoch (0-based)
        batch_idx: Current batch index within epoch (0-based)
        batches_per_epoch: Total number of batches per epoch
        start_epoch: Starting epoch (for resumed training)
    
    Returns:
        global_step: Total number of gradient steps taken
    """
    return (epoch - start_epoch) * batches_per_epoch + batch_idx

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,                                    # NEW: Current epoch number
    start_epoch: int = 0,                         # NEW: Starting epoch (for resume)
    logger=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = 1.0,
    log_every_n_steps: int = 100,                 # NEW: Log every N global steps
    log_every_n_batches: int = None,              # NEW: Alternative - log every N batches
) -> tuple[float, int]:
    """
    Train for one epoch with explicit step tracking and relationship logging.
    
    Args:
        epoch: Current epoch number (0-based)
        start_epoch: Starting epoch number (for resumed training)
        log_every_n_steps: Log every N global steps
        log_every_n_batches: Alternative to log_every_n_steps - log every N batches within epoch
    
    Returns:
        avg_loss: Average loss for this epoch
        final_global_step: Global step after this epoch
    """
    model.train()
    running_loss = 0.0
    n_batches = len(dataloader)
    
    # Log epoch/step relationship at start
    initial_global_step = calculate_global_step(epoch, 0, n_batches, start_epoch)
    final_expected_global_step = calculate_global_step(epoch, n_batches - 1, n_batches, start_epoch)

    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) != 3:
            raise ValueError(f"Each batch must be (X, Y, M). Got {len(batch)} elements.")
        
        # Calculate explicit global step
        current_global_step = calculate_global_step(epoch, batch_idx, n_batches, start_epoch)
        
        x, y, m = batch
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        m = m.to(device, non_blocking=True).float()

        if x.dim() == 4 and x.size(1) == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)


        # ============ ADD DEBUG CODE HERE ============
        if epoch == 0 and batch_idx == 0:  # Debug first batch of first epoch


            with torch.no_grad():
                # Test FFT consistency 
                test_image = torch.randn(1, 2, 320, 320).to(device) * 0.01
                print(f"Test image range: [{test_image.min():.6f}, {test_image.max():.6f}]")
                
                # Forward: image -> k-space
                test_k = fft2(test_image)
                print(f"After fft2: [{test_k.min():.6f}, {test_k.max():.6f}]")
                
                test_k_shifted = fftshift2(test_k)  
                print(f"After fftshift2: [{test_k_shifted.min():.6f}, {test_k_shifted.max():.6f}]")
                
                # Inverse: k-space -> image
                test_k_unshifted = fftshift2(test_k_shifted)
                print(f"After fftshift2 (inverse): [{test_k_unshifted.min():.6f}, {test_k_unshifted.max():.6f}]")
                
                test_image_recovered = ifft2(test_k_unshifted)
                print(f"After ifft2: [{test_image_recovered.min():.6f}, {test_image_recovered.max():.6f}]")
                
                # Check round-trip error
                roundtrip_error = (test_image - test_image_recovered).abs().mean()
                print(f"Round-trip error: {roundtrip_error:.6f}")
                
                # Check if the problem is the scale mismatch with input k-space
                actual_kspace_scale = x[0:1].abs().mean()
                fft_result_scale = test_k.abs().mean()
                print(f"Input k-space scale: {actual_kspace_scale:.6f}")
                print(f"FFT result scale: {fft_result_scale:.6f}")
                print(f"Scale ratio: {fft_result_scale/actual_kspace_scale:.6f}")


            print("\n=== DEBUGGING FIRST BATCH ===")
            
            # Debug data flow
            debug_results = debug_kiki_pipeline(x, y, m, model, device, "debug_kiki_pipeline.png")
            
            # Debug forward pass step-by-step  
            debug_output = debug_kiki_forward_pass(model, x, m, device)
            
            print("\n=== DEBUG COMPLETE ===")

        # ============ END DEBUG CODE ============

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = _get_autocast(device, enabled=(scaler is not None))
        with autocast_ctx:
            pred = model(x, m)
            pred, y_ = _align_pred_target(pred, y)
            loss = loss_fn(pred, y_)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item()
        
        # Explicit logging with both local and global step information
        should_log = False
        if log_every_n_batches is not None:
            should_log = batch_idx % log_every_n_batches == 0
        else:
            should_log = current_global_step % log_every_n_steps == 0
        

        # Log additional metrics every 1000 steps or at epoch boundaries
        if logger is not None and (current_global_step % 1000 == 0 or batch_idx == 0 or batch_idx == n_batches - 1):
            try:
                # Log learning rate if scheduler is being used
                current_lr = optimizer.param_groups[0]['lr']
            except Exception:
                pass

    # Final epoch summary
    epoch_time = time.time() - start_time
    avg_loss = running_loss / max(1, n_batches)
    final_global_step = calculate_global_step(epoch, n_batches - 1, n_batches, start_epoch)
    
    if logger is not None:
        # logger.log(f"Epoch {epoch + 1} Complete:")
        logger.log(f"  - Average loss: {avg_loss:.6f}")
        # logger.log(f"  - Batches processed: {n_batches}")
        # logger.log(f"  - Time: {epoch_time:.1f}s ({epoch_time/n_batches:.2f}s/batch)")
        # logger.log(f"  - Final global step: {final_global_step}")
        # logger.log(f"  - Total gradient updates: {final_global_step + 1}")
        
        # Log epoch-level metrics
        try:
            logger.log({
                "train/avg_loss": avg_loss 
            #     "train/epoch_time_sec": epoch_time,
            #     "train/batches_per_epoch": n_batches,
            #     "train/final_global_step": final_global_step,
             }, step=final_global_step)
        except Exception:
            pass

    return avg_loss, final_global_step

# Updated fit function to use explicit step tracking
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    config: Optional[KIKIConfig] = None,
    # optimization
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas=(0.9, 0.999),
    loss_name: str = "l1",
    # schedule
    use_cosine_decay: bool = False,
    T_max: Optional[int] = None,
    # run control
    num_epochs: int = 50,
    save_every: int = 5,
    test_every: Optional[int] = None,
    test_loader: Optional[DataLoader] = None,
    save_reconstructions: bool = False,
    reconstruction_save_path: Optional[Path] = None,
    test_sample_idx: int = 0,
    ckpt_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    mixed_precision: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    log_every_n_steps: int = 100,          # NEW: Global step logging frequency
) -> Dict[str, Any]:
    """
    Train/validate KIKI with explicit step tracking and comprehensive logging.
    """
    if ckpt_dir is None:
        raise ValueError("Please provide ckpt_dir (Path) where checkpoints will be saved.")
    
    # Validate test parameters
    if test_every is not None:
        if test_loader is None:
            raise ValueError("test_loader must be provided when test_every is specified")
        if test_every <= 0:
            raise ValueError("test_every must be positive")
    
    # Validate reconstruction parameters  
    if save_reconstructions:
        if test_every is None or test_loader is None:
            raise ValueError("save_reconstructions requires test_every and test_loader to be specified")
        if reconstruction_save_path is None:
            reconstruction_save_path = ckpt_dir / "reconstructions"
        reconstruction_save_path = Path(reconstruction_save_path)
    
    Model_path = ckpt_dir / "models"
    _mkdir(Model_path)

    # Loss
    if loss_name.lower() == "l1":
        loss_fn = nn.L1Loss()
    elif loss_name.lower() in ("l2", "mse"):
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("loss_name must be 'l1' or 'l2'")

    # Optimizer / Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if use_cosine_decay:
        tmax = T_max if T_max is not None else num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=0.0)
    else:
        scheduler = None

    # AMP
    scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    # Resume logic with explicit step calculation
    start_epoch = 0
    best_val = math.inf
    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    if resume_from is not None and Path(resume_from).is_file():
        chk = load_checkpoint(Path(resume_from), model, optimizer, scheduler, map_location="cpu")
        start_epoch = chk.get("epoch", 0)
        best_val = chk.get("best_val", math.inf)
        if scaler is not None and "scaler" in chk and chk["scaler"] is not None:
            try:
                scaler.load_state_dict(chk["scaler"])
            except Exception:
                pass
        logger.log(f"Resumed from {resume_from} at epoch {start_epoch}, best_val={best_val:.6f}")

    if config is not None and logger is not None:
        logger.log("=== Model Configuration ===")
        logger.log(f"KIKI iterations: {config.iters}")
        logger.log(f"K-space conv layers: {config.k}")
        logger.log(f"Image conv layers: {config.i}")
        logger.log(f"Input channels: {config.in_ch}")
        logger.log(f"Output channels: {config.out_ch}")
        logger.log(f"Feature maps: {config.fm}")
        logger.log(f"Config: iters={config.iters}, k={config.k}, i={config.i}, in_ch={config.in_ch}, out_ch={config.out_ch}, fm={config.fm}")
        logger.log(f"Acceleration Factor :6")

    # Move to device
    model.to(device)

    # Calculate training metrics
    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * (num_epochs - start_epoch)
    
    # if logger is not None:
    #     logger.log("=== Training Configuration ===")
    #     logger.log(f"Epochs: {start_epoch} -> {num_epochs} ({num_epochs - start_epoch} epochs to train)")
    #     logger.log(f"Batches per epoch: {batches_per_epoch}")
    #     logger.log(f"Total batches to process: {total_batches}")
    #     logger.log(f"Expected final global step: {calculate_global_step(num_epochs - 1, batches_per_epoch - 1, batches_per_epoch, start_epoch)}")
    #     logger.log(f"Logging every {log_every_n_steps} global steps")

    # Create test results directory if testing enabled
    test_results_dir = None
    if test_every is not None:
        test_results_dir = ckpt_dir / "test_results"
        _mkdir(test_results_dir)

    # Training loop with explicit step tracking
    history = {"train_loss": [], "val_loss": []}
    test_epochs = []
    reconstruction_paths = []
    
    for epoch in range(start_epoch, num_epochs):
        if logger is not None:
            logger.log(f"===== Epoch {epoch + 1}/{num_epochs} =====")

        # Train with explicit step tracking
        train_loss, final_global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            start_epoch=start_epoch,
            logger=logger,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            log_every_n_steps=log_every_n_steps,
        )
        history["train_loss"].append(train_loss)

        # Validation
        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            global_step=final_global_step,
        )
        history["val_loss"].append(val_loss)

        # Periodic testing (same as before)
        if test_every is not None and (epoch + 1) % test_every == 0:
            if logger is not None:
                logger.log(f"===== Running Test at Epoch {epoch+1} (Global Step {final_global_step}) =====")
            
            test_recons_dir = test_results_dir / f"epoch_{epoch+1:04d}"
            
            test_loop(
                model=model,
                test_loader=test_loader,
                device=device,
                logger=logger,
                ckpt_path=None,
                save_recons_dir=test_recons_dir,
                save_n_first=5,
            )
            
            if save_reconstructions:
                try:
                    img_path = save_reconstruction_image(
                        model=model,
                        test_loader=test_loader,
                        device=device,
                        save_path=reconstruction_save_path,
                        epoch=epoch + 1,
                        sample_idx=test_sample_idx,
                    )
                    reconstruction_paths.append(str(img_path))
                    if logger is not None:
                        logger.log(f"Saved reconstruction image: {img_path}")
                except Exception as e:
                    if logger is not None:
                        logger.log(f"Failed to save reconstruction image: {e}")
            
            test_epochs.append(epoch + 1)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoints with global step info
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val": best_val,
            "global_step": final_global_step,
            "batches_per_epoch": batches_per_epoch,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        save_checkpoint(state, ckpt_dir, "last.pt")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(state, ckpt_dir, f"epoch_{epoch+1:04d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(state, ckpt_dir, "best.pt")
            if logger is not None:
                logger.log(f"[best] val_loss improved to {best_val:.6f} at epoch {epoch+1}, global step {final_global_step}")

    final_state = {
        "history": history,
        "best_val": best_val,
        "last_ckpt": str(last_ckpt_path),
        "best_ckpt": str(best_ckpt_path),
        "epochs_run": num_epochs - start_epoch,
        "total_batches": total_batches,
        "final_global_step": final_global_step,
        "batches_per_epoch": batches_per_epoch,
        "test_epochs": test_epochs,
        "reconstruction_images": reconstruction_paths,
    }
    
    if logger is not None:
        logger.log("=== Training Complete ===")
        logger.log(f"Final statistics:")
        logger.log(f"  - Best validation loss: {best_val:.6f}")
        logger.log(f"  - Total epochs trained: {num_epochs - start_epoch}")
        logger.log(f"  - Total batches processed: {total_batches}")
        logger.log(f"  - Final global step: {final_global_step}")
        if test_epochs:
            logger.log(f"  - Testing performed at epochs: {test_epochs}")
        if reconstruction_paths:
            logger.log(f"  - Saved {len(reconstruction_paths)} reconstruction images")
    
    return final_state

# -----------------------------
# Testing / Inference
# -----------------------------
@torch.no_grad()
def test_loop(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger=None,
    ckpt_path: Optional[Path] = None,
    save_recons_dir: Optional[Path] = None,
    save_n_first: Optional[int] = None,
):
    """
    Evaluates (forward-only) the model on a test dataloader.
    If ckpt_path is provided, loads it before testing.
    Optionally saves the first N reconstructions as torch tensors (for later visualization).
    """
    if ckpt_path is not None and Path(ckpt_path).is_file():
        load_checkpoint(Path(ckpt_path), model, optimizer=None, scheduler=None, map_location="cpu")
        if logger is not None:
            logger.log(f"Loaded checkpoint for test: {ckpt_path}")

    model.to(device)
    model.eval()

    n_saved = 0
    if save_recons_dir is not None:
        _mkdir(save_recons_dir)

    total_batches = 0
    start = time.time()
    for batch_idx, batch in enumerate(test_loader):
        x, y, m = batch
        x = x.to(device, non_blocking=True).float()
        m = m.to(device, non_blocking=True).float()

        pred = model(x, m)  # raw prediction (complex or real)
        # Save a few predictions if requested
        if save_recons_dir is not None and (save_n_first is None or n_saved < save_n_first):
            # store as-is to keep exact model output; user can visualize later
            out_path = save_recons_dir / f"recon_batch{batch_idx:04d}.pt"
            torch.save(pred.cpu(), out_path)
            n_saved += 1

        total_batches += 1

    elapsed = time.time() - start
    if logger is not None:
        logger.log(f"[test] batches={total_batches}, time={elapsed:.1f}s, saved={n_saved} preds")




