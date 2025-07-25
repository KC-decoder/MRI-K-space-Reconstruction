import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import torch
import matplotlib.pyplot as plt
import io
import gc
from contextlib import redirect_stdout
import fastmri
from datetime import datetime
from torchsummary import summary
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.optim import Adam , RMSprop

from utils.mri_data import SliceDataset
from torch.optim.lr_scheduler import StepLR
from utils.data_transform import DataTransform_Diffusion , DataTransform_UNet , XAITransform
from utils.sample_mask import RingMaskFunc, RandomMaskGaussian
from utils.misc import *
from utils.XAI_utils import *
from help_func import print_var_detail

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.training_utils import UNetTrainer, train_unet
from utils.diffusion_train import Trainer
from utils.testing_utils import reconstruct_multicoil, recon_slice_unet
from net.unet.unet_supermap import Unet 
from net.unet.improved_unet import UnetModel
from utils.logger_utils import Logger
from utils.evaluation_utils import *

from utils.visualize_utils import plot_reconstruction_vs_ground_truth, plot_full_reconstruction_4panel, visualize_data_sample,generate_ring_masks_fixed_step, plot_ring_masks
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_memory_efficient():
    """Set up memory-efficient environment."""
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    print("Memory environment optimized")

# https://claude.ai/public/artifacts/9cc4caec-3b72-4560-a8ba-f7ec32fe220a

def l1_image_loss(pred, target):
    """
    Computes L1 (Mean Absolute Error) loss between predicted and target images.
    Args:
        pred: [B, 1, H, W] - Predicted image
        target: [B, 1, H, W] - Ground truth image
    Returns:
        Scalar L1 loss
    """
    return F.l1_loss(pred, target)

print(torch.__version__)
gpu = 0
# Check if specified GPU is available, else default to CPU
if torch.cuda.is_available():
    try:
        device = torch.device(f"cuda:{gpu}")
        # Test if the specified GPU index is valid
        _ = torch.cuda.get_device_name(device)
    except AssertionError:
        print(f"GPU {gpu} is not available. Falling back to GPU 0.")
        device = torch.device("cuda:0")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")





# Define Huber Loss
# huber_loss = nn.HuberLoss(delta=1.0)



def main():
    # ****** TRAINING SETTINGS ******
    # dataset settings
    idx_case = 8 # Select the case you want to visualize
    num_rings = 20
    batch_no = 0
    n_perturbations = 100
    path_dir_train = '/data2/users/koushani/FAST_MRI_data/MRI_Knee/singlecoil_train'
    # # # save settings
    exp_id = datetime.now().strftime("%m%d-%H-%M-%S")  # "0711-09-17-42"
    PATH_MODEL = f'/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask'
    save_folder=PATH_MODEL
    create_path(PATH_MODEL)
    ring_mask_path = pathlib.Path(PATH_MODEL) / "ring_mask" 
    
    
    # Call this in your main() function:
    setup_memory_efficient()
    
    
    # generate_ring_masks_fixed_step(save_dir=ring_mask_path)

    # # Step 2: Plot them
    # plot_ring_masks(save_dir=ring_mask_path)
    
    
    
    EXP_PATH = pathlib.Path(PATH_MODEL) / exp_id  # Full path with timestamp

    # # Ensure experiment directory exists
    EXP_PATH.mkdir(parents=True, exist_ok=True)

    

    # Define subfolders inside the experiment path
    LOGS_PATH = EXP_PATH / "logs"
    MODELS_PATH = EXP_PATH / "models" 

    # Create necessary subdirectories
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    create_path(PATH_MODEL)
    
    # model_load_path = EXP_PATH / "models" / "model_final.pt"
    # # # construct diffusion model
    # perturbations_output_dir= f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s300_lr_1e-05/0418-15-26-15/XAI/PERTURBATIONS_REVERSE_{num_rings}/"
    # test_output_dir= f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s300_lr_1e-05/0418-15-26-15/XAI/GRADCAM_VISUALIZATION_PROGRESIVE_{num_rings}/"
    # EXP_PATH.mkdir(parents=True, exist_ok=True)

    # Now safe to pass to Logger
    logger = Logger(logging_level="INFO", exp_path=EXP_PATH, use_wandb=False)
        
    path_dir_test = '/data2/users/koushani/FAST_MRI_data/MRI_Knee/singlecoil_test'
    img_mode = 'fastmri'  # 'fastmri' or 'B1000'
    bhsz = 16
    NUM_EPOCH = 100
    img_size = 320
   

    # root=pathlib.Path(path_dir_train)
    # print(root)
    
    # ====== Construct dataset ======
    # initialize mask
   # Define the shape of your images
    image_shape = (320, 320)

    # Create a fixed random Gaussian mask generator
    # mask_func = RandomMaskGaussian(
    #     acceleration=4,
    #     center_fraction=0.08,
    #     size=(1, *image_shape),  # (1, H, W)
    #     seed=42,                 # Fix seed for reproducibility and consistency
    #     mean=(0, 0),
    #     cov=[[1, 0], [0, 1]],
    #     concentration=3,
    #     patch_size=4,
    # )

    mask_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/ring_mask/ring_mask_9.npy"
    mask_func = RingMaskFunc(mask_path)
    
        
    
    transform = DataTransform_UNet(mask_func=mask_func, combine_coil = False)


    
    # training set
    dataset_train = SliceDataset(
        root=pathlib.Path(path_dir_train),
        transform=transform,
        challenge='singlecoil',
        num_skip_slice=5,
    )

   # test set
    dataset_test = SliceDataset(
        root=pathlib.Path(path_dir_test),
        transform=transform,
        challenge='singlecoil',
        num_skip_slice=5,
    )

    # 90/10 split
    n_total = len(dataset_train)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # DataLoaders
    dataloader_train = DataLoader(train_dataset, batch_size=bhsz, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=bhsz, shuffle=False)
    
    logger.log(f"Using device: {device}")
    logger.log(f"len dataloader train: {len(dataloader_train)}")
    logger.log(f"len dataloader test: {len(dataloader_val)}")
    
    
    logger.log("\n----------------TRAINING DATA--------------------")
    for i, (x, y, m) in enumerate(dataloader_train):
        logger.log(f"\nSample {i+1}:")
        logger.log(f"  Input (x) shape : {x.shape}")
        logger.log(f"  Target (y) shape: {y.shape}")
        logger.log(f"  Mask shape      : {m.shape}")
        break
    

    
    VIZ_PATH = EXP_PATH / "VISUALIZATIONS"
    VIZ_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    SUMMARY_FILE = VIZ_PATH / f"ring_mask_summary.png"
    plot_ring_masks(save_dir = ring_mask_path, output_path=SUMMARY_FILE)
    
    sample_idx = 4
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    
    
    sample_idx = 6
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    
    
    sample_idx = 8
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    
    
    sample_idx = 12
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)



    # model_load_path =  f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/0711-09-17-42/models/model_ck20.pt"
    # output_dir =f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/0711-09-17-42/VISUALIZATIONS"
    # npy_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_final.npy"
    # png_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_final.png"
    # npy_path = os.path.join(output_dir, npy_filename)
    # png_save_path = os.path.join(output_dir, png_filename)


    model = Unet(
    dim=64,
    channels=1,         # input is single-channel masked image
    out_dim=1,          # output is single-channel reconstructed image
    dim_mults=(1, 2, 3, 4),
    self_condition=False
    ).to(device)
    
    
    # checkpoint = torch.load(model_load_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    weight_decay = 0.0



    logger.log('model size: %.3f MB' % (calc_model_size(model)))
    logger.log(f"Results will be saved in: {save_folder}")

    # # Create a buffer to capture output
    # buffer = io.StringIO()
    # with redirect_stdout(buffer):
    #     summary(model, input_size=(1, 320, 320), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")

    # # Get the string output
    # summary_str = buffer.getvalue()
    # channels = 1
    # H = 320
    # W = 320
    # # Now log it
    # logger.log("Model Summary:\n" + summary_str)
    # input_size=(channels, H, W)
    # summary(model, input_size=(1, 320, 320), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # 2. Optimizer
    # --------------------------
    
    
    learning_rate = 1e-5  # start here
    # use RMSprop as optimizer
    optimizer = Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # --------------------------
    # 3. Loss function
    # --------------------------


    loss_fn = l1_image_loss  # expects [B, 1, H, W]


    # --------------------------
    # 3. Scheduler
    # --------------------------
    step_size = 12
    lr_gamma = 0.1 # change in learning rate
    scheduler = StepLR(optimizer, step_size, lr_gamma)
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # # # # # --- XAI ANALYSIS ---
    

    # --------------------------
    # 4. Train model
    # --------------------------
    train_unet(
        train_dataloader=dataloader_train,
        test_dataloader=dataloader_val,
        optimizer=optimizer,
        loss=loss_fn,
        net=model,
        scheduler=scheduler,
        device=device,
        logger = logger,
        PATH_MODEL=EXP_PATH,        # e.g., "/checkpoints/NormUNet/"
        NUM_EPOCH=NUM_EPOCH,                 # or any number of epochs
        save_every=20,                  # print every 5 epochs
        show_test=True                # run test after training
    )
    
    
    # model_load_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/0711-09-17-42/models/model_final.pt"
    # grad_cam_path= EXP_PATH / "XAI" / f"ring_7_Top16Channel_Grad-CAM_ups_plot.png"
    # grad_cam_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists


    # Get one reconstruction
    pred, zf, tg, i_nmse, i_psnr, i_ssim, mask, X_for_gradcam = recon_slice_unet(
        dataloader=dataloader_val,  # Define this earlier
        net=model,
        device=device,
        idx_case=idx_case,
    )


    # plot_reconstruction_vs_ground_truth(pred, tg, ssim_value=i_ssim, save_path=f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/ring_1_model_reconstruction_plot(normalized).png")

    # plot_full_reconstruction_4panel(
    # pred=pred[0],
    # zf=zf[0],
    # gt=tg[0],
    # mask=mask[0][0],  # assuming shape is [B, 1, H, W]
    # ssim_value=i_ssim, save_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/exp/ring_1_model_FullPanel_reconstruction_plot.png")

    print("FINAL LAYER GRADCAM ANALYSIS")
    results = analyze_final_layer_kspace_only(
    model=model,
    X_for_gradcam=X_for_gradcam,
    device=device,
    tg=tg,
    zf=zf,
    pred=pred,
    ssim=i_ssim,
    psnr=i_psnr,
    ring_mask_path=mask_path,  # Your ring mask path
    exp_path=EXP_PATH
    )
    print("COMPLETE IMPLEMENTATION READY:")
    
        
        
        
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
