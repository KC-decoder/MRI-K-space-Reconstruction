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
from torchsummaryX import summary
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.optim import Adam , RMSprop

from utils.mri_data import SliceDataset
from torch.optim.lr_scheduler import StepLR
from utils.data_transform import DataTransform_UNet_Kspace , DataTransform_UNet , XAITransform
from utils.sample_mask import RingMaskFunc, RandomMaskGaussian
from utils.misc import *
from utils.XAI_utils import *
from help_func import print_var_detail
from debug.CUNet_debug import *

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.training_utils import UNetTrainer, train_unet, train_KIKI
from utils.diffusion_train import Trainer
from utils.testing_utils import reconstruct_multicoil, recon_slice_unet
from net.unet.complex_Unet import CUNet ,count_parameters, CUNetLoss, quick_cunet_check, test_cunet_step_by_step, verify_cunet_fix
from net.unet.KIKI_unet import KIKI
from net.unet.complex_KIKI import KIKIRecon
from utils.KIKIUnet_utils import *
from utils.CUNetTraining_utils import *
from utils.CUNetEvaluation_utils import *
from utils.logger_utils import Logger
from utils.evaluation_utils import *

from utils.visualize_utils import visualize_kspace_sample, visualize_cunet_recon, visualize_kiki_recon
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

# print(torch.__version__)
# gpu = 0
# # Check if specified GPU is available, else default to CPU
# if torch.cuda.is_available():
#     try:
#         device = torch.device(f"cuda:{gpu}")
#         # Test if the specified GPU index is valid
#         _ = torch.cuda.get_device_name(device)
#     except AssertionError:
#         print(f"GPU {gpu} is not available. Falling back to GPU 0.")
#         device = torch.device("cuda:0")
# else:
#     print("CUDA is not available. Using CPU.")
#     device = torch.device("cpu")





# Define Huber Loss
# huber_loss = nn.HuberLoss(delta=1.0)



def main():
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
    # ****** TRAINING SETTINGS ******
    # dataset settings
    idx_case = 0 # Select the case you want to visualize
    num_rings = 20
    batch_no = 0
    n_perturbations = 100
    path_dir_train = '/data2/users/koushani/FAST_MRI_data/MRI_Knee/singlecoil_train'
    # # # save settings
    exp_id = datetime.now().strftime("%m%d-%H-%M-%S") 
    # exp_id = "0821-19-20-04"
    PATH_MODEL = f'/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKINet_RandomGaussianMask'
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
    bhsz = 32
    NUM_EPOCH = 100
    img_size = 320
   

    # root=pathlib.Path(path_dir_train)
    # print(root)
    
    # ====== Construct dataset ======
    # initialize mask
   # Define the shape of your images
    image_shape = (320, 320)

    #Create a fixed random Gaussian mask generator
    mask_func = RandomMaskGaussian(
        acceleration=8,
        center_fraction=0.08,
        size=(1, *image_shape),  # (1, H, W)
        seed=42,                 # Fix seed for reproducibility and consistency
        mean=(0, 0),
        cov=[[1, 0], [0, 1]],
        concentration=3,
        patch_size=4,
    )

    # mask_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/ring_mask/ring_mask_2.npy"
    # mask_func = RandomMaskGaussian()
    
        
    
    transform = DataTransform_UNet_Kspace(mask_func=mask_func, combine_coil = False)


    
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



    # Get one batch
    batch = next(iter(dataloader_train))
    x_batch, y_batch, mask_batch = batch

    # Select a single sample from the batch
    x_sample = x_batch[20].unsqueeze(0) # Shape: [1, 1, 320, 320]
    y_sample = y_batch[20].unsqueeze(0)
    mask_sample = mask_batch[20].unsqueeze(0)


    dummy_dataset = TensorDataset(x_sample, y_sample, mask_sample)

    # Step 3: Create DataLoader with batch_size=1
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1)
    
    # logger.log(f"Using device: {device}")
    # logger.log(f"len dataloader train: {len(dataloader_train)}")
    # logger.log(f"len dataloader test: {len(dataloader_val)}")
    
    
    logger.log("\n----------------TRAINING DATA--------------------")
    for i, (x, y, m) in enumerate(dummy_dataloader):
        logger.log(f"\nSample {i+1}:")
        logger.log(f"  Input (x) shape : {x.shape}")
        logger.log(f"  Target (y) shape: {y.shape}")
        logger.log(f"  Mask shape      : {m.shape}")
        break


    # logger.log("\n----------------TESTING DATA--------------------")
    # for i, (x, y, m) in enumerate(dataloader_val):
    #     logger.log(f"\nSample {i+1}:")
    #     logger.log(f"  Input (x) shape : {x.shape}")
    #     logger.log(f"  Target (y) shape: {y.shape}")
    #     logger.log(f"  Mask shape      : {m.shape}")
    #     break
    

    
    VIZ_PATH = EXP_PATH / "VISUALIZATIONS"
    VIZ_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    # SUMMARY_FILE = VIZ_PATH / f"ring_mask_summary.png"
    # plot_ring_masks(save_dir = ring_mask_path, output_path=SUMMARY_FILE)
    
    sample_idx = 0
    VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    visualize_kspace_sample(dataloader_train, sample_idx, f"K-Space Training Sample Visualization_{sample_idx}", VIZ_FILE)

    # """Main function - integrate this into your workflow"""
    
    # print("Choose debugging approach:")
    # print("1. Debug existing trained model")
    # print("2. Test with synthetic data")
    # print("3. Test identity mapping")
    # print("4. Step-by-step guide")
    # print("5. Complete diagnosis")
    
    # choice = input("Enter choice (1-5): ").strip()
    
    # if choice == "1":
    #     model_path = input("Enter path to trained model: ").strip()
    #     # You need to provide your dataloader here
    #     print("You need to provide your test dataloader")
    #     # debug_cunet_training_issues(model_path, your_dataloader)
        
    # elif choice == "2":
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     synthetic_dataloader = create_synthetic_test_data(device=device)
    #     debug_cunet_training_issues(None, synthetic_dataloader)
        
    # elif choice == "3":
    #     test_with_identity_mapping()
        
    # elif choice == "4":
    #     step_by_step_debug_guide()
        
    # elif choice == "5":
    #     print("Running complete diagnosis...")
    #     # This would run everything
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     synthetic_dataloader = create_synthetic_test_data(device=device)
    #     debug_cunet_training_issues(None, synthetic_dataloader)
    #     test_with_identity_mapping()
    #     step_by_step_debug_guide()
    
    # else:
    #     print("Invalid choice")

    # verify_cunet_fix()

    # sample_idx = 15
    # VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    # visualize_kspace_sample(dataloader_train, sample_idx, f"K-Space Training Sample Visualization_{sample_idx}", VIZ_FILE)


    # sample_idx = 20
    # VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    # visualize_kspace_sample(dataloader_val, sample_idx, f"K-Space Testing Sample Visualization_{sample_idx}", VIZ_FILE)
    # logger.log(f"Train losses: {history['train_losses']}")
    # logger.log(f"Val losses:   {history['val_losses']}")

     # --------------------------
    # 2. Optimizer
    # --------------------------
    
    # weight_decay = 0.0
    # learning_rate = 1e-5  # start here
    # # # use RMSprop as optimizer
    # optimizer = Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # --------------------------
    # 3. Loss function
    # --------------------------


    # loss_fn = l1_image_loss  # expects [B, 1, H, W]




    # Run step-by-step tests first
    # print("Choose testing mode:")
    # print("1. Full step-by-step tests (recommended)")
    # print("2. Quick sanity check only")
    
    # choice = input("Enter choice (1 or 2): ").strip()
    
    # if choice == "2":
    #     quick_cunet_check()
    # else:
    #     # Run full diagnostic tests
    #     results = test_cunet_step_by_step()
        
    #     # Optional: Save results
    #     save_results = input("\nSave test results to file? (y/n): ").strip().lower()
    #     if save_results == 'y':
    #         with open("cunet_test_results.txt", "w") as f:
    #             f.write("CUNet Step-by-Step Test Results\n")
    #             f.write("=" * 40 + "\n\n")
    #             for test_name, result in results.items():
    #                 status = "PASSED" if result else "FAILED"
    #                 f.write(f"{test_name}: {status}\n")
    #         print(" Results saved to cunet_test_results.txt")
    
    # print("CU-Net test completed successfully!")



    # model, optimizer, scheduler, loss_fn = setup_cunet_training(
    #     train_dataloader=dummy_dataloader,  # Your train dataloader
    #     test_dataloader=dummy_dataloader,   # Your test dataloader
    #     device=device,
    #     logger=logger,            # Your logger
    #     loss_fn = loss_fn,
    #     optimizer = optimizer,
    #     PATH_MODEL=EXP_PATH,
    #     base_features=32,
    #     learning_rate=1e-4
    # )

    # # Train the model
    # trained_model = train_cunet(
    #     train_dataloader=dummy_dataloader,
    #     test_dataloader=dummy_dataloader,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     net=model,
    #     scheduler=scheduler,
    #     device=device,
    #     logger=logger,
    #     PATH_MODEL=EXP_PATH,
    #     NUM_EPOCH=1000,
    #     save_every=200,
    #     show_test_every=200
    # )

    # logger.log("COMPLETED TRAINING CUNET")

    # model_load_path = MODELS_PATH / "model_final.pth"
    # output_dir = VIZ_PATH / "Recon.png"
    
    iters = 5
    k_layers = 3
    i_layers = 3
    in_ch = 2                  # single-coil complex => 2 (real, imag)
    out_ch = 2
    features = 32

    cfg = KikiConfig(
        iters=iters,
        k=k,
        i=i,
        in_ch=in_ch,
        out_ch=out_ch,
        fm=fm,
    )

    
    model = build_model(cfg)



    summary = fit(
    model=model,
    train_loader=dataloader_train,
    val_loader=dataloader_val,
    device=device,
    logger=logger,
    lr=1e-4,
    weight_decay=0.0,
    loss_name="l1",                 # or "l2"
    use_cosine_decay=True,
    T_max=None,                     # None -> num_epochs
    num_epochs=NUM_EPOCH,           # from your settings
    save_every=5,                   # save every 5 epochs
    ckpt_dir=EXP_PATH,
    resume_from=None,               # or a .pt path to resume
    mixed_precision=True,
    grad_clip_norm=1.0,
)

    logger.log("Training complete!")

    model_load_path =  f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKINet_RandomGaussianMask/0824-23-03-36/models/model_final.pt" # ring 2
    output_dir = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKINet_RandomGaussianMask/0824-23-03-36/VISUALIZATIONS/Recon_KIKINet.png"     # f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKIUnet_RandomGaussianMask/0821-15-49-25/VISUALIZATIONS"


#     # Run full debug
#     model = full_debug_pipeline(model_load_path, dummy_dataloader, device)
    
#     # Quick visualization
#     visualize_debug_output(model, dummy_dataloader, device)

#     visualize_cunet_recon(
#     model_load_path=model_load_path,
#     dataloader=dataloader_val,          # must yield (X, Y, M)
#     device=device,
#     sample_idx=0,
#     save_path=output_dir,
#     target_is_kspace=False,             # set True if Y is k-space (2ch)
#     show_mask=True,
#     base_features=32,                   # match what you trained with
#     use_data_consistency=True           # match what you trained with
# )
    # visualize_kiki_recon(model_load_path = model_load_path,
    #                      dataloader=dummy_dataloader,
    #                      device = device,
    #                      sample_idx= 0,
    #                      save_path = output_dir,
    #                      )
    





if __name__ == "__main__":
    main()

























    # sample_idx = 6
    # VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    # visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    
    
    # sample_idx = 8
    # VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    # visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)
    
    
    # sample_idx = 12
    # VIZ_FILE = VIZ_PATH / f"dataset_sample_{sample_idx}.png"
    # visualize_data_sample(dataloader_train, sample_idx, f"K-Space Sample Visualization_{sample_idx}", VIZ_FILE)



    # model_load_path =  f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKIUnet_RandomGaussianMask/0821-13-24-18/models/model_final.pt" # ring 2
    # output_dir =f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKIUnet_RandomGaussianMask/0821-13-24-18/VISUALIZATIONS"
    # npy_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_final.npy"
    # png_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_final.png"
    # npy_path = os.path.join(output_dir, npy_filename)
    # png_save_path = os.path.join(output_dir, png_filename)

   
    
    # checkpoint = torch.load(model_load_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # model = CUNet(in_channels=2, out_channels=1, base_features=32)
    
    # model = Unet(
    # dim=64,
    # channels=1,         # input is single-channel masked image
    # out_dim=1,          # output is single-channel reconstructed image
    # dim_mults=(1, 2, 3, 4),
    # self_condition=False
    # ).to(device)

    # weight_decay = 0.0

    



    # logger.log('model size: %.3f MB' % (calc_model_size(model)))
    # logger.log(f"Results will be saved in: {save_folder}")

    # # Create a buffer to capture output
    # buffer = io.StringIO()
    # with redirect_stdout(buffer):
    #     dummy_k = torch.randn(1, 2, 320, 320, device=device)  # (B,2,H,W)
    #     dummy_m = torch.ones(1, 1, 320, 320, device=device)   # (B,1,H,W)

    #     summary(model, dummy_k, dummy_m)

    # # Get the string output
    # summary_str = buffer.getvalue()
    # channels = 2
    # H = 320
    # W = 320
    # # Now log it
    # logger.log("Model Summary:\n" + summary_str)
    # input_size=(channels, H, W)
    # summary(model, dummy_k, dummy_m)

    # # --------------------------
    # # 2. Optimizer
    # # --------------------------
    
    # weight_decay = 0.0
    # learning_rate = 1e-5  # start here
    # # # use RMSprop as optimizer
    # optimizer = Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # # --------------------------
    # # 3. Loss function
    # # --------------------------


    # loss_fn = l1_image_loss  # expects [B, 1, H, W]


    # # --------------------------
    # # 3. Scheduler
    # # --------------------------
    # step_size = 12
    # lr_gamma = 0.1 # change in learning rate
    # scheduler = StepLR(optimizer, step_size, lr_gamma)
    # # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # # # # # # --- XAI ANALYSIS ---
    

    # # --------------------------
    # # 4. Train model
    # # --------------------------
    # train_unet(
    #     train_dataloader=dummy_dataloader,
    #     test_dataloader=dummy_dataloader,
    #     optimizer=optimizer,
    #     loss=loss_fn,
    #     net=model,
    #     scheduler=scheduler,
    #     device=device,
    #     logger = logger,
    #     PATH_MODEL=EXP_PATH,        # e.g., "/checkpoints/NormUNet/"
    #     NUM_EPOCH=NUM_EPOCH,                 # or any number of epochs
    #     save_every=20,                  # print every 5 epochs
    #     show_test=True                # run test after training
    # )
    

    # # --------------------------
    # # 4. Train KIKI model
    # # --------------------------

    # train_KIKI(
    #         train_dataloader=dummy_dataloader,
    #         test_dataloader=dummy_dataloader,
    #         optimizer=optimizer,
    #         loss_fn=loss_fn,
    #         net=model,
    #         scheduler=scheduler,
    #         device=device,
    #         logger = logger,
    #         PATH_MODEL=EXP_PATH,        # e.g., "/checkpoints/NormUNet/"
    #         NUM_EPOCH=NUM_EPOCH,                 # or any number of epochs
    #         save_every=20,                  # print every 5 epochs
    #         show_test=False                # run test after training
    #     )
    
    # # model_load_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/KIKIUnet_RandomGaussianMask/0819-16-15-55/models/model_ck40.pt"
    # # grad_cam_path= EXP_PATH / "XAI" / f"ring_7_Top16Channel_Grad-CAM_ups_plot.png"
    # # grad_cam_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists


    # Get one reconstruction
    # pred, zf, tg, i_nmse, i_psnr, i_ssim, mask, X_for_gradcam = recon_slice_unet(
    #     dataloader=dummy_dataloader,  # Define this earlier
    #     net=model,
    #     device=device,
    #     idx_case=idx_case,
    # )

    # out_files = evaluate_and_plot(
    #     model=model,
    #     test_dataloader=dataloader_val,
    #     device=device,
    #     save_dir=VIZ_PATH,
    #     batches=2,                # visualize first 2 batches
    #     samples_per_batch=2       # 4 samples per batch
    # )
    # print("Saved:", out_files)

    # plot_reconstruction_vs_ground_truth(pred, tg, ssim_value=i_ssim, save_path=VIZ_PATH/"Unet_RandomGaussian_model_reconstruction_plot(normalized).png")

    # plot_full_reconstruction_4panel(
    # pred=pred[0],
    # zf=zf[0],
    # gt=tg[0],
    # mask=mask[0][0],  # assuming shape is [B, 1, H, W]
    # ssim_value=i_ssim, save_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_RandomGaussian_Mask/0607-19-19-45/VISUALIZATIONS/ring_2_model_FullPanel_reconstruction_plot.png")

    #  # After training is complete, add this:
    # print("\n" + "="*60)
    # print("STARTING ENHANCED GRAD-CAM PLUS PLUS ANALYSIS")
    # print("="*60)
    
    # # Single sample analysis
    # idx_case = 8  # Change this to analyze different samples
    # enhanced_results = analyze_final_layer_kspace_gradcam_plus_plus(model,
    #                                                                 X_for_gradcam,
    #                                                                 device, tg, zf, pred,i_ssim, i_psnr,
    #                                                                 ring_mask_path = mask_path,
    #                                                                 exp_path = EXP_PATH)
    
        
        
        
    
    

    
    
    
    
    
    
