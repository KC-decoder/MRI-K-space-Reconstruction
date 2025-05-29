# #     # # Get one batch
# #     # batch = next(iter(dataloader_train))
# #     # x_batch, y_batch, mask_batch = batch

# #     # # Select a single sample from the batch
# #     # x_sample = x_batch[10].unsqueeze(0) # Shape: [1, 1, 320, 320]
# #     # y_sample = y_batch[10].unsqueeze(0)
# #     # mask_sample = mask_batch[10].unsqueeze(0)


# #     # dummy_dataset = TensorDataset(x_sample, y_sample, mask_sample)

# #     # # Step 3: Create DataLoader with batch_size=1
# #     # dummy_dataloader = DataLoader(dummy_dataset, batch_size=1)




# #     # Save path for visualization


# # #     train_save_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/recon_results/multicoil/train_sample_multicoil.png"
# # #     test_save_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/recon_results/multicoil/test_sample_multicoil.png"    
# # #     visualize_kspace_sample(dataloader_train, "Multicoil Training Sample", train_save_path)
# # #     visualize_kspace_sample(dataloader_test, "Multicoil Test Sample", test_save_path)

# # model settings
# CH_MID = 64
# # training settings
# NUM_EPOCH = 300
# learning_rate = 1e-5  # start here
# time_steps = 100
# train_steps = NUM_EPOCH * len(dataloader_train) # can be customized to a fixed number, however, it should reflect the dataset size.
# train_steps = max(train_steps, 5000)



# # # save settings
# # exp_id = datetime.now().strftime("%m%d-%H-%M-%S")
# # PATH_MODEL = '/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s'+str(NUM_EPOCH)+'_lr_'+str(learning_rate)+'/'
# # create_path(PATH_MODEL)
# EXP_PATH = pathlib.Path(PATH_MODEL) / exp_id  # Full path with timestamp

# # # Ensure experiment directory exists
# # EXP_PATH.mkdir(parents=True, exist_ok=True)

# # sample_idx = 6
# # VIZ_PATH = EXP_PATH / "VISUALIZATIONS"
# # VIZ_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# # VIZ_FILE = VIZ_PATH / f"kspace_sample_{sample_idx}.png"

# # # Print some basic statistics
# # visualize_data_sample(dataloader_train, sample_idx, "K-Space Sample Visualization", VIZ_FILE)

# # Define subfolders inside the experiment path
# LOGS_PATH = EXP_PATH / "logs"
# MODELS_PATH = EXP_PATH / "models"

# # Create necessary subdirectories
# LOGS_PATH.mkdir(parents=True, exist_ok=True)
# MODELS_PATH.mkdir(parents=True, exist_ok=True)
# create_path(PATH_MODEL)



# # Initialize Logger
# logger = Logger(logging_level="INFO", exp_path=EXP_PATH, use_wandb=False)


# sample_viz_path = VIZ_PATH / f"sample_{sample_idx}_debug_visualization.png"

# # Use your existing visualization function
# # visualize_data_sample(dataloader_train, sample_idx=sample_idx, title="Debug: Single Sample", save_path=sample_viz_path)

# #  # Step 4: Log sample details
# # logger.log("\n----------------SINGLE SAMPLE DEBUG--------------------")
# # for i, (x, y, m) in enumerate(dummy_dataloader):
# #     logger.log(f"\nSample {i+1}:")
# #     logger.log(f"  Input (x) shape : {x.shape}")
# #     logger.log(f"  Target (y) shape: {y.shape}")
# #     logger.log(f"  Mask shape      : {m.shape}")
# #     break


# logger.log(f"Using device: {device}")
# logger.log(f"train_steps:{train_steps}")
# logger.log(f"len dataloader train: {len(dataloader_train)}")
# logger.log(f"len dataloader test: {len(dataloader_val)}")
# logger.log("\n----------------TRAINING DATA--------------------")
# for i, (masked_image) in enumerate(dataloader_train):
#     logger.log(f"\nBatch {i+1}:")
#     logger.log(f"  masked image shape      : {masked_image.shape}")
#     break



# logger.log("\n----------------TESTING DATA--------------------")
# for i, (masked_image) in enumerate(dataloader_val):
#     logger.log(f"\nBatch {i+1}:")
#     logger.log(f"  masked image shape      : {masked_image.shape}")
#     break



# # construct diffusion model
# save_folder=PATH_MODEL
# load_path=None
# blur_routine='Constant'
# train_routine='Final'
# sampling_routine='x0_step_down'
# discrete=False

# # model = Unet().to(device)



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


# # input_size=(channels, H, W)
# # summary(model, input_size=(1, 320, 320), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------
# # 2. Optimizer
# # --------------------------
# # use RMSprop as optimizer
# optimizer = Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

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



# # --------------------------
# # 4. Train model
# # --------------------------
# # train_unet(
# #     train_dataloader=dataloader_train,
# #     test_dataloader=dataloader_val,
# #     optimizer=optimizer,
# #     loss=loss_fn,
# #     net=model,
# #     scheduler=scheduler,
# #     device=device,
# #     logger = logger,
# #     PATH_MODEL=EXP_PATH,        # e.g., "/checkpoints/NormUNet/"
# #     NUM_EPOCH=NUM_EPOCH,                 # or any number of epochs
# #     save_every=50,                  # print every 5 epochs
# #     show_test=True                # run test after training
# # )










# # #     # """XAI, RECONSTRUCTION, SALIENCY ANALYSIS"""



# idx_case = 4 # Select the case you want to visualize
# batch_no = 0
# n_perturbations = 100
# #     # t = 20000       # Time step in the diffusion process
# load_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s300_lr_1e-05/0418-15-26-15/models/model_final.pt"
# test_output_dir= f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s300_lr_1e-05/0418-15-26-15/XAI/PERTURBATION_GRAD_CAM_VISUALIZATIONS_sample_{idx_case}_batch_{batch_no}_accFactorDifference_1"
# perturbation_save_dir = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/SuperMap_l1_Adam_s300_lr_1e-05/0418-15-26-15/XAI/PERTURBATIONS" # 1. Pull a batch
# npy_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_300.npy"
# png_filename = f"test_sample_singlecoil_reconstruction_idx_{idx_case}_model_300.png"
# npy_path = os.path.join(test_output_dir, npy_filename)
# png_save_path = os.path.join(test_output_dir, png_filename)

# os.makedirs(test_output_dir, exist_ok=True)
# os.makedirs(perturbation_save_dir, exist_ok=True)
# # #     # perturbation_save_path = f"/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/XAI_results/multicoil/"



# checkpoint = torch.load(load_path, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])


# # Create the generator
# perturb_gen = XAIPerturbationGenerator(
#     dataloader=dataloader_train,
#     mask_function_class = RandomMaskGaussianDiffusion,
#     n_perturbations=n_perturbations,
#     acceleration_start=4,
#     acceleration_end=4.5,
#     save_path= os.path.join(perturbation_save_dir, f"perturbations_case_batch_{batch_no}_slice_number_{idx_case}_num_perturbation_{n_perturbations}.npy")
# )

# # Generate perturbations for slice 2 in batch 0
# perturbations = perturb_gen.generate(batch_idx=batch_no, slice_idx=idx_case)
# print(f"Perturbations shape: {perturbations.shape}")
# perturbations = torch.tensor(perturbations, dtype=torch.float32).to(device)

# #     # Reconstruct full batch
# #     with torch.no_grad():
# #         y_pred_batch = model(X_batch)  # [B, 1, H, W]

# #     # --- Set up Grad-CAM ---
# gradcam = GradCAM(model=model, target_layer=model.final_res_block)

# analyze_perturbations_with_gradcam(
# perturbations=perturbations,
# model=model,
# gradcam=gradcam,
# test_output_dir=test_output_dir,
# device = device
# )

# #     save_gradcam_per_sample_visualizations(
# #     X_batch=X_batch,
# #     y_pred_batch=y_pred_batch,
# #     cams=cams,
# #     save_dir=test_output_dir,
# # )

# # # --- Reconstruct ---
# # pred, zf, tg, i_nmse, i_psnr, i_ssim = recon_slice_unet(dataloader_val, model, device, idx_case)

# # # --- Save Numpy Arrays ---
# # np.save(npy_path, {
# #     "masked_input": zf.numpy(),
# #     "reconstruction": pred.numpy(),
# #     "ground_truth": tg.numpy()
# # })
# # print(f"Saved NPY to: {npy_path}")

# # # --- Save Visualization ---

# # ssim_val = i_ssim


# # plt.figure(figsize=(12, 4))
# # plt.subplot(1, 3, 1)
# # plt.imshow(zf[0].numpy(), cmap='gray')
# # plt.title("Masked Input")
# # plt.axis('off')

# # plt.subplot(1, 3, 2)
# # plt.imshow(pred[0].numpy(), cmap='gray')
# # plt.title(f"Reconstructed Output\nSSIM: {ssim_val:.4f}")
# # plt.axis('off')

# # plt.subplot(1, 3, 3)
# # plt.imshow(tg[0].numpy(), cmap='gray')
# # plt.title("Ground Truth")
# # plt.axis('off')

# # plt.tight_layout()
# # plt.savefig(png_save_path)
# # plt.close()

# # print(f"Saved PNG visualization to: {png_save_path}")

# #     # # construct trainer and train
# #     # save_path = "/data2/users/koushani/FAST_MRI_data/checkpoint_dir/Axial/diffusion_Gmask_fastmri_4x_T1000_S700000/recon_results/"
    
# #     # # Choose a random batch from the dataloader
# #     # random_batch = random.randint(0, len(dataloader_test) - 1)

# #     # # Iterate through the dataloader
# #     # for i, batch in enumerate(dataloader_test):
# #     #     if i == random_batch:
# #     #         # Extract the batch
# #     #         print(f"Batch {i}:")
# #     #         kspace, target, x = batch
            
# #     #         # Print the shapes and data
# #     #         print(f"kspace shape: {kspace.shape}")   # Shape of the k-space tensor
# #     #         print(f"target shape: {target.shape}")       # Shape of the mask
# #     #         print(f"x shape: {x.shape}")   # Shape of the reconstructed image (if available)
# #     #         save_image_from_kspace(kspace, i, save_path)
# #     #         break



# #      # diffusion = KspaceDiffusion(
# #     #     model,
# #     #     image_size=img_size,
# #     #     device_of_kernel= device,
# #     #     channels=2,
# #     #     timesteps=time_steps,  # number of steps
# #     #     loss_type='l1',  # L1 or L2
# #     #     blur_routine=blur_routine,
# #     #     train_routine=train_routine,
# #     #     sampling_routine=sampling_routine,
# #     #     discrete=discrete,
# #     # ).to(device)



# #     # # trainer = Trainer(
# #     # #     diffusion,
# #     # #     image_size=img_size,
# #     # #     train_batch_size=bhsz,
# #     # #     train_lr=learning_rate,
# #     # #     train_num_steps=train_steps,  # total training steps
# #     # #     gradient_accumulate_every=2,  # gradient accumulation steps
# #     # #     ema_decay=0.995,  # exponential moving average decay
# #     # #     fp16=False,  # turn on mixed precision training with apex
# #     # #     save_and_sample_every=50000,
# #     # #     results_folder=save_folder,
# #     # #     load_path=load_path,
# #     # #     dataloader_train=dataloader_train,
# #     # #     dataloader_test=dataloader_test,
# #     # # )
# #     # # trainer.train()
    
   
# #     # # # Initialize the visualizer
# #     # # visualizer = Visualizer_Kspace_ColdDiffusion(
# #     # #     diffusion_model=diffusion,
# #     # #     ema_decay=0.995,
# #     # #     dataloader_test=dataloader_test,  # The test dataloader created in the main function
# #     # #     load_path=load_path,  # Path to the model checkpoint
# #     # #     device = device,
# #     # #     output_dir = test_output_dir,
# #     # # ) 
# #     # # # Visualize intermediate k-space and reconstructions
# #     # # # Save the extracted data
# #     # # visualizer.save_intermediate_kspace_npy(idx_case=idx_case, t=t, filename=npy_filename)  
# #     # # plot_reconstruction_results_from_npy(npy_path, png_save_path)


# #     # idx_case = 100  # Select the case you want to visualize
# #     # t = 50000       # Time step in the diffusion process
    
# #     # # Visualize intermediate k-space and reconstructions
# #     # xt, kspacet, gt_imgs_abs, direct_recons_abs, sample_imgs_abs, kspace = visualizer.show_intermediate_kspace_cold_diffusion(
# #     #     t=t, 
# #     #     idx_case=idx_case
# #     # )
    
# #     # plot_intermediate_kspace_results(
# #     #     xt, kspacet, gt_imgs_abs, direct_recons_abs, sample_imgs_abs, kspace, save_path
# #     # )
    
# #      # num_perturbations = 10
    
# #     # # Instantiate and run the class
# #     # # Initialize EditColdDiffusion
# #     # edit_cd = EditColdDiffusion(
# #     #     model=diffusion,
# #     #     model_path=load_path,
# #     #     npy_dir=test_output_dir,
# #     #     sample_id=idx_case,
# #     #     timesteps=t,
# #     #     num_perturbations=num_perturbations,
# #     #     output_dir=perturbation_save_path,
# #     #     npy_filename = npy_filename,
# #     #     device=device
# #     # )

# #     # # Run the perturbation analysis
# #     # edit_cd.run()
    
# #     # diffusion = KspaceDiffusion(
# #     #     model,
# #     #     image_size=img_size,
# #     #     device_of_kernel= device,
# #     #     channels=2,
# #     #     timesteps=time_steps,  # number of steps
# #     #     loss_type='l1',  # L1 or L2
# #     #     blur_routine=blur_routine,
# #     #     train_routine=train_routine,
# #     #     sampling_routine=sampling_routine,
# #     #     discrete=discrete,
# #     # ).to(device)
    
    
    
# # import torch
# # print(torch.__version__)
# # print(torch.version.cuda)
# # print(torch.cuda.is_available())