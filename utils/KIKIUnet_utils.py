import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
from net.unet.KIKI_unet import KIKI

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from contextlib import nullcontext

def _get_autocast(device: torch.device, enabled: bool = True):
    """
    Returns a proper autocast context for the given device.
    """
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        # modern API (silences FutureWarning)
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if device.type == "cpu":
        # safe on recent PyTorch; else falls back below
        try:
            return torch.amp.autocast("cpu", dtype=torch.bfloat16)
        except Exception:
            return nullcontext()
    return nullcontext()

# -----------------------------
# Utility: make dirs & device
# -----------------------------
def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)




iters = 3
k = 3
i = 3
in_ch = 2
out_ch = 1
fm = 32
class KikiConfig:
    def __init__(self, iters, k, i, in_ch, out_ch, fm):
        self.iters = iters
        self.k = k
        self.i = i
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fm = fm


def build_model(cfg: dict) -> nn.Module:
    kcfg = KikiConfig(
        iters=iters,
        k=k,
        i=i,
        in_ch=in_ch,
        out_ch=out_ch,
        fm=fm,
    )
    return KIKI(kcfg)



# -----------------------------
# Prediction/target alignment
# -----------------------------
def _align_pred_target(pred: torch.Tensor, target: torch.Tensor):
    # pred: (N,2,H,W) complex image, target: (N,1,H,W) magnitude
    if pred.dim()!=4 or target.dim()!=4:
        raise ValueError(f"Expected 4D NCHW, got pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.size(1)==2 and target.size(1)==1:
        # complex -> magnitude
        mag = torch.sqrt(pred[:,0:1].pow(2) + pred[:,1:1+1].pow(2) + 1e-12)
        return mag, target
    if pred.size(1)==target.size(1):
        return pred, target
    raise ValueError(f"Channel mismatch: pred C={pred.size(1)}, target C={target.size(1)}")

# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(state: Dict[str, Any], ckpt_dir: Path, name: str):
    _mkdir(ckpt_dir)
    path = ckpt_dir / name
    torch.save(state, path)
    return path

def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None, map_location: Optional[str] = None) -> Dict[str, Any]:
    chk = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(chk["model"])
    if optimizer is not None and "optimizer" in chk and chk["optimizer"] is not None:
        optimizer.load_state_dict(chk["optimizer"])
    if scheduler is not None and "scheduler" in chk and chk["scheduler"] is not None:
        scheduler.load_state_dict(chk["scheduler"])
    return chk

# -----------------------------
# Train / Val loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    logger=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = 1.0,
    global_step: int = 0,
):
    model.train()
    running_loss = 0.0
    n_batches = 0

    start = time.time()
    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError(f"Each batch must be (X, Y, M). Got {len(batch)} elements.")
        x, y, m = batch
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        m = m.to(device, non_blocking=True).float()

        if x.dim() == 4 and x.size(1) == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = _get_autocast(device, enabled=(scaler is not None))
        with autocast_ctx:
            pred = model(x, m)
            pred, y_ = _align_pred_target(pred, y)
            loss = loss_fn(pred, y_)

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
        n_batches += 1
        global_step += 1

        if logger is not None:
            try:
                train = float(loss.item())
                fmt = f"{train:.6f}".rstrip("0").rstrip(".")
                logger.info(f"loss at train step {global_step}: {fmt}")
            except Exception:
                logger.log(f"[train step {global_step}] loss={loss.item():.6f}")

    epoch_time = time.time() - start
    avg_loss = running_loss / max(1, n_batches)
    if logger is not None:
        logger.log(f"[train] avg_loss={avg_loss:.6f}, batches={n_batches}, time={epoch_time:.1f}s")
        try:
            logger.log({"train/avg_loss": avg_loss, "train/epoch_time_sec": epoch_time}, step=global_step)
        except Exception:
            pass

    return avg_loss, global_step


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

# -----------------------------
# Fit / Resume orchestration
# -----------------------------
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    # optimization
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas=(0.9, 0.999),
    loss_name: str = "l1",          # "l1" or "l2"
    # schedule
    use_cosine_decay: bool = False,
    T_max: Optional[int] = None,    # if None, set to num_epochs
    # run control
    num_epochs: int = 50,
    save_every: int = 5,
    ckpt_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    mixed_precision: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, Any]:
    """
    Train/validate KIKI with checkpoints + resume, logging to your provided logger.

    Returns a run summary dict containing best metrics and final checkpoint path.
    """
    if ckpt_dir is None:
        raise ValueError("Please provide ckpt_dir (Path) where checkpoints will be saved.")
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
    
    # after:
    scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    # Resume
    start_epoch = 0
    best_val = math.inf
    global_step = 0
    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    if resume_from is not None and Path(resume_from).is_file():
        chk = load_checkpoint(Path(resume_from), model, optimizer, scheduler, map_location="cpu")
        start_epoch = chk.get("epoch", 0)
        best_val = chk.get("best_val", math.inf)
        global_step = chk.get("global_step", 0)
        # restore scaler if present
        if scaler is not None and "scaler" in chk and chk["scaler"] is not None:
            try:
                scaler.load_state_dict(chk["scaler"])
            except Exception:
                pass
        logger.log(f"Resumed from {resume_from} at epoch {start_epoch}, best_val={best_val:.6f}, step={global_step}")

    # Move to device
    model.to(device)

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(start_epoch, num_epochs):
        if logger is not None:
            logger.log(f"===== Epoch {epoch+1}/{num_epochs} =====")

        # Train
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            global_step=global_step,
        )
        history["train_loss"].append(train_loss)

        # Val
        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            logger=logger,
            global_step=global_step,
        )
        history["val_loss"].append(val_loss)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save "last" checkpoint every epoch
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val": best_val,
            "global_step": global_step,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        save_checkpoint(state, ckpt_dir, "last.pt")

        # Save periodic checkpoints
        if (epoch + 1) % save_every == 0:
            save_checkpoint(state, ckpt_dir, f"epoch_{epoch+1:04d}.pt")

        # Track best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(state, ckpt_dir, "best.pt")
            if logger is not None:
                logger.log(f"[best] val_loss improved to {best_val:.6f} -> saved best.pt")

    final_state = {
        "history": history,
        "best_val": best_val,
        "last_ckpt": str(last_ckpt_path),
        "best_ckpt": str(best_ckpt_path),
        "epochs_run": num_epochs - start_epoch,
        "global_step": global_step,
    }
    if logger is not None:
        logger.log(f"Training finished. best_val={best_val:.6f}")
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