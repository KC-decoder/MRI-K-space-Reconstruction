import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_model_training(model_path, device):
    """Debug if the model trained properly."""
    print("=== Model Training Debug ===")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check training info
    print(f"Final loss: {checkpoint.get('loss', 'Not found')}")
    print(f"Epoch: {checkpoint.get('epoch', 'Not found')}")
    
    # Load model and check weights
    from net.unet.complex_Unet import CUNet
    model = CUNet(in_channels=2, out_channels=1, base_features=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Check for problematic weights
    total_params = 0
    zero_params = 0
    nan_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param.abs() < 1e-8).sum().item()
        nan_params += torch.isnan(param).sum().item()
        
        print(f"{name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
        
        if torch.isnan(param).any():
            print(f"  ⚠️  WARNING: {name} contains NaN values!")
        if param.std() < 1e-8:
            print(f"  ⚠️  WARNING: {name} has very small variance (might be dead neurons)!")
    
    print(f"\nWeight Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Near-zero parameters: {zero_params:,} ({100*zero_params/total_params:.1f}%)")
    print(f"  NaN parameters: {nan_params:,}")
    
    return model

def debug_model_output(model, dataloader, device):
    """Debug what the model is actually outputting."""
    print("\n=== Model Output Debug ===")
    
    model.eval()
    with torch.no_grad():
        X, y, mask = next(iter(dataloader))
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        
        print(f"Input shapes: X={X.shape}, y={y.shape}, mask={mask.shape}")
        
        # Check input data
        print(f"\nInput k-space stats:")
        print(f"  Min: {X.min():.6f}, Max: {X.max():.6f}")
        print(f"  Mean: {X.mean():.6f}, Std: {X.std():.6f}")
        
        print(f"\nTarget image stats:")
        print(f"  Min: {y.min():.6f}, Max: {y.max():.6f}")
        print(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}")
        
        # Forward pass
        try:
            pred = model(X, mask)
            print(f"\nModel output stats:")
            print(f"  Shape: {pred.shape}")
            print(f"  Min: {pred.min():.6f}, Max: {pred.max():.6f}")
            print(f"  Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")
            
            # Check for problematic outputs
            if pred.max() < 1e-6:
                print("  🚨 PROBLEM: Output is essentially zero!")
            if torch.isnan(pred).any():
                print("  🚨 PROBLEM: Output contains NaN values!")
            if torch.isinf(pred).any():
                print("  🚨 PROBLEM: Output contains Inf values!")
                
            return pred, y
            
        except Exception as e:
            print(f"  🚨 ERROR during forward pass: {e}")
            return None, y

def debug_intermediate_outputs(model, X, mask, device):
    """Debug intermediate outputs in the model."""
    print("\n=== Intermediate Outputs Debug ===")
    
    model.eval()
    with torch.no_grad():
        # Hook to capture intermediate outputs
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        model.encoder1.register_forward_hook(get_activation('encoder1'))
        model.bottleneck.register_forward_hook(get_activation('bottleneck'))
        model.decoder1.register_forward_hook(get_activation('decoder1'))
        model.final_conv.register_forward_hook(get_activation('final_conv'))
        
        # Forward pass
        pred = model(X, mask)
        
        # Check intermediate activations
        for name, activation in activations.items():
            print(f"{name}: shape={activation.shape}")
            print(f"  Min: {activation.min():.6f}, Max: {activation.max():.6f}")
            print(f"  Mean: {activation.mean():.6f}, Std: {activation.std():.6f}")
            
            if activation.max() < 1e-6:
                print(f"  ⚠️  {name} outputs are essentially zero!")
            if torch.isnan(activation).any():
                print(f"  🚨 {name} contains NaN values!")

def test_simple_forward_pass(device):
    """Test if the model can do a simple forward pass with random data."""
    print("\n=== Simple Forward Pass Test ===")
    
    from net.unet.complex_Unet import CUNet
    
    # Create fresh model
    model = CUNet(in_channels=2, out_channels=1, base_features=16).to(device)
    
    # Random input
    X = torch.randn(1, 2, 64, 64).to(device)
    mask = torch.ones(1, 1, 64, 64).to(device)
    
    try:
        with torch.no_grad():
            pred = model(X, mask)
            print(f"✅ Forward pass successful!")
            print(f"Input shape: {X.shape}")
            print(f"Output shape: {pred.shape}")
            print(f"Output range: [{pred.min():.4f}, {pred.max():.4f}]")
            
            # Check if output is reasonable
            if pred.std() > 0.01:
                print("✅ Model produces varied outputs (good sign)")
            else:
                print("⚠️  Model outputs have very low variance")
                
    except Exception as e:
        print(f"🚨 Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def suggest_fixes():
    """Suggest potential fixes based on common issues."""
    print("\n=== Suggested Fixes ===")
    print("1. 🔍 Check training loss curve - did loss actually decrease?")
    print("2. 🔍 Verify data normalization - ensure training/inference use same scaling")
    print("3. 🔍 Check learning rate - might be too high causing instability")
    print("4. 🔍 Verify loss function - CUNetLoss might not be appropriate")
    print("5. 🔍 Check complex operations - numerical instabilities possible")
    print("6. 🔍 Try training a simpler model first (regular U-Net) to verify data")
    print("7. 🔍 Check gradient flow - gradients might be vanishing/exploding")
    print("8. 🔍 Verify target data - ensure targets are magnitude images, not k-space")

def full_debug_pipeline(model_path, dataloader, device):
    """Run complete debugging pipeline."""
    print("🔧 Starting CU-Net Debug Pipeline...")
    
    # 1. Test simple forward pass
    test_simple_forward_pass(device)
    
    # 2. Debug trained model
    model = debug_model_training(model_path, device)
    
    # 3. Debug model outputs
    pred, target = debug_model_output(model, dataloader, device)
    
    # 4. Debug intermediate outputs if model runs
    if pred is not None:
        X, y, mask = next(iter(dataloader))
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        debug_intermediate_outputs(model, X, mask, device)
    
    # 5. Suggest fixes
    suggest_fixes()
    
    return model

# Quick visualization for debugging
def visualize_debug_output(model, dataloader, device):
    """Visualize what's actually happening."""
    model.eval()
    with torch.no_grad():
        X, y, mask = next(iter(dataloader))
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        pred = model(X, mask)
        
        pred = model(X, mask)
        
        # Convert to numpy for visualization
        pred_np = pred[0, 0].cpu().numpy()
        target_np = y[0, 0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Model output (with extreme scaling to see any pattern)
        im1 = axes[0].imshow(pred_np, cmap='gray')
        axes[0].set_title(f'Model Output\nRange: [{pred_np.min():.2e}, {pred_np.max():.2e}]')
        plt.colorbar(im1, ax=axes[0])
        
        # Target
        im2 = axes[1].imshow(target_np, cmap='gray')
        axes[1].set_title(f'Target\nRange: [{target_np.min():.2f}, {target_np.max():.2f}]')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

# if __name__ == "__main__":
#     # Usage example:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_path = "/path/to/your/model.pt"
    
#     # Run full debug
#     # model = full_debug_pipeline(model_path, test_dataloader, device)
    
#     # Quick visualization
#     # visualize_debug_output(model, test_dataloader, device)
#     pass