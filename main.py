import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random

from torch.utils.data import DataLoader

from utils.mri_data import SliceDataset
from utils.data_transform import DataTransform_Diffusion
from utils.sample_mask import RandomMaskGaussianDiffusion, RandomMaskDiffusion, RandomMaskDiffusion2D
from utils.misc import *
from help_func import print_var_detail

from diffusion.kspace_diffusion import KspaceDiffusion
from utils.diffusion_train import Trainer
from net.u_net_diffusion import Unet

from utils.visualize_utils import Visualizer_Kspace_ColdDiffusion, plot_intermediate_kspace_results, save_image_from_kspace

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)






def main():
    # ****** TRAINING SETTINGS ******
    # dataset settings
    acc = 4  # acceleration factor
    frac_c = 0.08  # center fraction
    path_dir_train = '/data2/users/koushani/FAST_MRI_data/singlecoil_train'
    path_dir_test = '/data2/users/koushani/FAST_MRI_data/singlecoil_test'
    img_mode = 'fastmri'  # 'fastmri' or 'B1000'
    bhsz = 16
    img_size = 320


    # ====== Construct dataset ======
    # initialize mask
    mask_func = RandomMaskGaussianDiffusion(
        acceleration=acc,
        center_fraction=frac_c,
        size=(1, img_size, img_size),
    )

    # initialize dataset
    data_transform = DataTransform_Diffusion(
        mask_func,
        img_size=img_size,
        combine_coil=True,
        flag_singlecoil=True,
    )

    # training set
    dataset_train = SliceDataset(
        root=pathlib.Path(path_dir_train),
        transform=data_transform,
        challenge='singlecoil',
        num_skip_slice=5,
    )

    # test set
    dataset_test = SliceDataset(
        root=pathlib.Path(path_dir_test),
        transform=data_transform,
        challenge='singlecoil',
        num_skip_slice=5,
    )

    dataloader_train = DataLoader(dataset_train, batch_size=bhsz, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=bhsz, shuffle=True)
    
    
    save_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/recon_results/"
    
    # # Choose a random batch from the dataloader
    # random_batch = random.randint(0, len(dataloader_test) - 1)

    # # Iterate through the dataloader
    # for i, batch in enumerate(dataloader_test):
    #     if i == random_batch:
    #         # Extract the batch
    #         print(f"Batch {i}:")
    #         kspace, target, x = batch
            
    #         # Print the shapes and data
    #         print(f"kspace shape: {kspace.shape}")   # Shape of the k-space tensor
    #         print(f"target shape: {target.shape}")       # Shape of the mask
    #         print(f"x shape: {x.shape}")   # Shape of the reconstructed image (if available)
    #         save_image_from_kspace(kspace, i, save_path)
    #         break
    
    print('len dataloader train:', len(dataloader_train))
    print('len dataloader test:', len(dataloader_test))
    
    
    
    # model settings
    CH_MID = 64
    # training settings
    NUM_EPOCH = 10
    learning_rate = 2e-5
    time_steps = 1000
    train_steps = NUM_EPOCH * len(dataloader_train) # can be customized to a fixed number, however, it should reflect the dataset size.
    train_steps = max(train_steps, 700000)
    print('train_steps:',train_steps)
    # save settings
    PATH_MODEL = '/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_'+str(img_mode)+'_'+str(acc)+'x_T'+str(time_steps)+'_S'+str(train_steps)+'/'
    create_path(PATH_MODEL)
    
    
    
    
    # construct diffusion model
    save_folder=PATH_MODEL
    load_path=None
    blur_routine='Constant'
    train_routine='Final'
    sampling_routine='x0_step_down'
    discrete=False

    model = Unet(
        dim=CH_MID,
        dim_mults=(1, 2, 4, 8),
        channels=2,
    ).to(device)
    print('model size: %.3f MB' % (calc_model_size(model)))

    diffusion = KspaceDiffusion(
        model,
        image_size=img_size,
        device_of_kernel='cuda',
        channels=2,
        timesteps=time_steps,  # number of steps
        loss_type='l1',  # L1 or L2
        blur_routine=blur_routine,
        train_routine=train_routine,
        sampling_routine=sampling_routine,
        discrete=discrete,
    ).to(device)
    
    # # construct trainer and train

    # # trainer = Trainer(
    # #     diffusion,
    # #     image_size=img_size,
    # #     train_batch_size=bhsz,
    # #     train_lr=learning_rate,
    # #     train_num_steps=train_steps,  # total training steps
    # #     gradient_accumulate_every=2,  # gradient accumulation steps
    # #     ema_decay=0.995,  # exponential moving average decay
    # #     fp16=False,  # turn on mixed precision training with apex
    # #     save_and_sample_every=50000,
    # #     results_folder=save_folder,
    # #     load_path=load_path,
    # #     dataloader_train=dataloader_train,
    # #     dataloader_test=dataloader_test,
    # # )
    # # trainer.train()
    
    load_path = '/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/model_50000.pt'
    save_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/recon_results/intermediate_kspace_visualization_50000_trainloader.png"
    # Initialize the visualizer
    visualizer = Visualizer_Kspace_ColdDiffusion(
        diffusion_model=diffusion,
        ema_decay=0.995,
        dataloader_test=dataloader_train,  # The test dataloader created in the main function
        load_path=load_path,  # Path to the model checkpoint
    ) 
        


    idx_case = 50  # Select the case you want to visualize
    t = 50000       # Time step in the diffusion process
    
    # Visualize intermediate k-space and reconstructions
    xt, kspacet, gt_imgs_abs, direct_recons_abs, sample_imgs_abs, kspace = visualizer.show_intermediate_kspace_cold_diffusion(
        t=t, 
        idx_case=idx_case
    )
    
    plot_intermediate_kspace_results(
        xt, kspacet, gt_imgs_abs, direct_recons_abs, sample_imgs_abs, kspace, save_path
    )


if __name__ == "__main__":
    main()
    