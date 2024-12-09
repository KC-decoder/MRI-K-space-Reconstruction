import matplotlib.pyplot as plt
import torch

def test_trained_model(model, data_loader, device='cuda'):
    """
    Test the trained model on a sample from the data_loader, get the output k-space, 
    and reconstruct the image from it.
    
    :param model: The trained model (UNet).
    :param data_loader: DataLoader to sample from.
    :param device: Device to run the model on ('cuda' or 'cpu').
    """
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # No need to compute gradients during testing
        for image_batch, kspace_batch in data_loader:
            # Move data to the appropriate device
            image_batch = image_batch.to(device)
            kspace_batch = kspace_batch.to(device)
            
            # Get the first sample from the batch
            image_sample = image_batch[0].unsqueeze(0)  # Shape: (1, 1, height, width)
            kspace_target_sample = kspace_batch[0].unsqueeze(0)  # Shape: (1, 2, height, width)
            
            # Pass the image through the trained model to get the predicted k-space
            predicted_kspace = model(image_sample)  # Shape: (1, 2, height, width)
            
            # Reconstruct the image from the predicted k-space
            # Convert the real and imaginary channels into a complex tensor
            complex_kspace = torch.view_as_complex(predicted_kspace.permute(0, 2, 3, 1))  # Shape: (1, height, width)
            
            # Apply inverse Fourier transform to get the reconstructed image
            reconstructed_image = fastmri.ifft2c(complex_kspace)  # Shape: (1, height, width)
            
            # Get the magnitude of the complex image
            reconstructed_image_abs = fastmri.complex_abs(reconstructed_image).cpu().squeeze(0)
            
            # Get the ground truth image (from the input)
            original_image = image_sample.cpu().squeeze(0)  # Shape: (height, width)
            
            # Plot and compare the original image and the reconstructed image
            plt.figure(figsize=(12, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(original_image.squeeze().numpy(), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image_abs.squeeze().numpy(), cmap='gray')
            plt.title('Reconstructed Image')
            plt.axis('off')
            
            plt.show()
            
            break  # Only test on one sample

# # Example of running the function
# test_trained_model(trained_model, train_loader, device='cuda')