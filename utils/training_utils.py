import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

import fastmri
from fastmri.data import transforms

from utils.evaluation_utils import *
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from pathlib import Path
import copy


def cycle(dl):
    while True:
        for data in dl:
            yield data

def train_unet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        scheduler,
        device,
        logger,
        PATH_MODEL,
        NUM_EPOCH=5,
        save_every=500,
        show_test=False):
    '''
    Train the U-Net.
    :param train_dataloader: training dataloader.
    :param test_dataloader: test dataloader.
    :param optimizer: optimizer.
    :param loss: loss function object.
    :param net: network object.
    :param device: device, gpu or cpu.
    :param NUM_EPOCH: number of epoch, default=5.
    :param show_step: int, default=-1. Steps to show intermediate loss during training. -1 for not showing.
    :param show_test: flag. Whether to show test after training.
    '''

    net = net.to(device)
    net.train()
    PATH_MODEL = Path(PATH_MODEL) 
    MODELS_PATH = PATH_MODEL / "models"
    LOGS_PATH = PATH_MODEL / "logs"
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(NUM_EPOCH))
    for epoch in pbar:
        running_loss = 0.0
        pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCH} | Avg Loss: {running_loss / len(train_dataloader):.6f}")
        for idx, data in enumerate(train_dataloader):   
            X, y, mask = data
            X = X.to(device).float()
            y = y.to(device).float()

            y_pred = net(X)
            loss_train = loss(y_pred, y)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()


        avg_loss = running_loss / len(train_dataloader)
        logger.log(f"Epoch {epoch + 1}/{NUM_EPOCH} | Avg Loss: {avg_loss:.6f}")
        # # update outer pbar with average loss info
        # pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCH} | Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODELS_PATH / f'model_ck{epoch + 1}.pt')

    # Final test after training
    if show_test:
        nmse, psnr, ssim = test_unet(test_dataloader, net, device,logger)
        logger.log(f"Final Test -- SSIM: {ssim}\n")

    # Save final model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, MODELS_PATH / 'model_final.pt')

    logger.log(f'MODEL SAVED in {MODELS_PATH}.')

    return net


def test_unet(
        test_dataloader,
        net,
        device,
        logger):
    '''
    Test the reconstruction performance. U-Net.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            X, y, mask = data

            X = X.to(device).float()  #[B,1,H,W]
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach()  # [B,1,H,W]
            pred = y_pred.detach()
            tg = tg.squeeze(1)  # [B,H,W]
            pred = pred.squeeze(1)
            max = torch.amax(X, dim=(1, 2, 3)).detach()
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            if idx==0:
                print('tg.shape:', tg.shape)
                print('pred.shape:', pred.shape)

            nmseb = 0
            psnrb = 0
            ssimb = 0
            B = tg.shape[0]
            for idxs in range(B):
                nmseb += calc_nmse_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
                psnrb += calc_psnr_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
                ssimb += calc_ssim_tensor(tg[idxs].unsqueeze(0), pred[idxs].unsqueeze(0))
            nmseb /= B
            psnrb /= B
            ssimb /= B
            nmse += nmseb
            psnr += psnrb
            ssim += ssimb

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    logger.log('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    logger.log('----------------------------------------------------------------------')

    return nmse, psnr, ssim

class MultiScaleLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4, 8], weights=None, loss_fn=F.l1_loss):
        """
        Multi-scale loss function for image reconstruction.

        Args:
            scales (list): Downscaling factors for multi-scale loss computation.
            weights (list, optional): Weighting for each scale. Defaults to equal weighting.
            loss_fn (callable): Loss function (e.g., L1 or SSIM).
        """
        super().__init__()
        self.scales = scales
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0 / len(scales)] * len(scales)

    def forward(self, y_pred, y_true):
        """
        Compute multi-scale loss.

        Args:
            y_pred (Tensor): Predicted MRI reconstruction.
            y_true (Tensor): Ground truth MRI image.

        Returns:
            Tensor: Multi-scale weighted loss.
        """
        loss = 0.0
        for i, scale in enumerate(self.scales):
            if scale > 1:
                y_pred_scaled = F.avg_pool2d(y_pred, kernel_size=scale, stride=scale)
                y_true_scaled = F.avg_pool2d(y_true, kernel_size=scale, stride=scale)
            else:
                y_pred_scaled = y_pred
                y_true_scaled = y_true

            loss += self.weights[i] * self.loss_fn(y_pred_scaled, y_true_scaled)

        return loss



class UNetTrainer:
    """
    Trainer for U-Net model with Exponential Moving Average (EMA) and step-based training.
    """

    def __init__(
            self,
            model,
            *,
            ema_decay=0.995,
            train_dataloader,
            test_dataloader,
            logger,
            optimizer= None,
            loss_fn= None,
            device="cuda",
            train_lr=2e-5,
            train_num_steps=5000,
            gradient_accumulate_every=1,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=2000,
            eval_every=500,
            exp_path: Path,
            load_path=None
    ):
        """
        Args:
            exp_path (Path): Root experiment directory where checkpoints/logs are stored.
        """
        self.device = device
        self.model = model.to(device)
        self.ema_model = copy.deepcopy(model).to(device)
        self.ema_decay = ema_decay
        self.update_ema_every = update_ema_every
        self.save_and_sample_every = save_and_sample_every
        self.step_start_ema = step_start_ema
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.eval_every = eval_every  # Evaluate every few steps

        self.train_dataloader = cycle(train_dataloader)  # Convert dataloader to iterator
        self.test_dataloader = test_dataloader
        self.logger = logger  # Logger instance

        self.dl = iter(train_dataloader)  # Convert to iterator
        
        self.optimizer = optimizer if optimizer else Adam(model.parameters(), lr=train_lr)
        # self.loss_fn = MultiScaleLoss(scales=[1, 2, 4, 8], loss_fn=F.l1_loss)

        self.scaler = GradScaler()

        # Experiment Path Setup
        self.exp_path = exp_path
        self.model_path = self.exp_path / "models"
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.best_loss = float('inf')
        self.step = 0

        # Load checkpoint if available
        if load_path:
            self.load(load_path)
            
    def reset_ema(self):
        """Reset EMA model to match the current model."""
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_ema(self):
        """Update the EMA model using exponential moving average."""
        if self.step < self.step_start_ema:
            self.reset_ema()
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = ema_param.data * self.ema_decay + param.data * (1 - self.ema_decay)

    def save_model(self, step=None, best=False, final=False):
        """Save model and EMA model checkpoints."""
        if best:
            filename = self.model_path / "best_model.pt"
        elif final:
            filename = self.model_path / "final_model.pt"
        else:
            filename = self.model_path / f"checkpoint_step_{step}.pt"

        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        self.logger.log(f"Model saved at {filename}")

    def load(self, load_path):
        """Load a checkpoint for model and EMA model."""
        self.logger.log(f"Loading model from {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self):
        """ Train the U-Net model with step-based training, EMA, and periodic evaluation. """
        self.logger.log(f"Starting training for {self.train_num_steps} steps...")

        acc_loss = 0.0  # Accumulate loss for logging
        self.model.train()
        pbar = tqdm(range(self.step, self.train_num_steps), desc="Training Progress")

        for step in pbar:
            self.step = step
            u_loss = 0.0  # Track per-step loss
            
            for _ in range(self.gradient_accumulate_every):
                # Get next batch (masked input, ground truth, mask)
                image_masked, image_full, mask = next(self.train_dataloader)

                # Move tensors to GPU
                image_masked = image_masked.to(self.device).float()  # Shape: (B, 1, 2, H, W)
                image_full = image_full.to(self.device).float()  # Shape: (B, 1, H, W)

                self.optimizer.zero_grad()
                
                # print(f"Shape of image_full: {image_full.shape}")
                # print(f"Shape of image_masked: {image_masked.shape}")
                

                # Mixed precision training
                
                y_pred = self.model(image_masked)  # Predict reconstruction
                # print(f"Shape of y_pred_real: {y_pred.shape}")
                loss = F.l1_loss(y_pred, image_full) # Compute loss

                # Backpropagation with gradient accumulation
                self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                if (step + 1) % self.gradient_accumulate_every == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                u_loss += loss.item()

            # Average loss over gradient accumulation steps
            avg_loss = u_loss / self.gradient_accumulate_every
            acc_loss += avg_loss
            pbar.set_description(f"Loss={avg_loss:.6f}")

            # EMA Update
            if self.step % self.update_ema_every == 0:
                self.update_ema()

            # Periodic evaluation
            if self.step % self.eval_every == 0 and self.step > 0:
                mean_loss = acc_loss / self.eval_every
                self.logger.log(f"Step {self.step}: Training Loss = {mean_loss:.6f}")
                self.evaluate()
                acc_loss = 0.0  # Reset accumulated loss

            # Save model periodically
            if self.step % self.save_and_sample_every == 0 and self.step > 0:
                self.save_model(step)

        # Save final model
        self.save_model(final=True)
        self.logger.log("Training completed.")
        
    def test(self):
        """ Evaluate the U-Net using EMA model and print metrics. """
        self.ema_model.eval()
        nmse, psnr, ssim = 0.0, 0.0, 0.0
        num_batches = len(self.test_dataloader)

        self.logger.log("Starting evaluation on test dataset...")

        with torch.no_grad():
            pbar = tqdm(self.test_dataloader, desc="Testing")
            for idx, (image_masked, target, mask) in enumerate(pbar):
                image_masked, target = image_masked.to(self.device).float(), target.to(self.device).float()

                # Forward pass using EMA model
                y_pred = self.ema_model(image_masked)

                # Compute batch-wise metrics
                # nmse += calc_nmse_tensor(target, y_pred).item()
                # psnr += calc_psnr_tensor(target, y_pred).item()
                ssim += calc_ssim_tensor(target, y_pred).item()

        # nmse /= num_batches
        # psnr /= num_batches
        # NMSE={nmse:.6f}, PSNR={psnr:.4f},
        ssim /= num_batches

        self.logger.log(f"Final Test Metrics: SSIM={ssim:.4f}")
        self.logger.log("Evaluation completed.")

    def evaluate(self):
        """Evaluate model performance on test set."""
        self.logger.log(f"Step {self.step}: Running Evaluation...")
        self.test()







def train_wnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_mid,
        loss_img,
        alpha,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the WNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = alpha * loss_mid(k_pred_mid, yk) + loss_img(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

        if i == 0 or (i + 1) % 5 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL+'model_ck'+str(i + 1)+'.pt')

    # test model
    if show_test:
        nmse, psnr, ssim = test_wnet(
            test_dataloader,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL+'model_ck.pt')
    print('MODEL SAVED.')

    return net


def test_wnet(
        test_dataloader,
        net,
        device):
    '''
    Test the reconstruction performance. WNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def train_varnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the VarNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = loss(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

        if i == 0 or (i + 1) % 5 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL+'model_ck'+str(i + 1)+'.pt')

    # test model
    if show_test:
        nmse, psnr, ssim = test_varnet(
            test_dataloader,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL+'model_ck.pt')
    print('MODEL SAVED.')

    return net


def test_varnet(
        test_dataloader,
        net,
        device):
    '''
    Test the reconstruction performance. VarNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            X = fastmri.complex_abs(fastmri.ifft2c(Xk.detach()))  # [B,Nc,H,W]
            max = torch.amax(X, dim=(1, 2, 3))
            scale_coeff = 1. / max  # [B,]
            tg = torch.einsum('ijk, i -> ijk', tg, scale_coeff)
            pred = torch.einsum('ijk, i -> ijk', pred, scale_coeff)

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### TEST NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim
