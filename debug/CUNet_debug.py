import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
import json
from utils.kiki_helpers import DataConsist , create_zero_filled_baseline , complex_magnitude, ifft2, fft2, fftshift2

# =============================================================================
# STEP 3: Update debug functions - replace your debug functions with these
# =============================================================================

import matplotlib.pyplot as plt


def debug_kiki_pipeline(x, y, m, model, device, save_path="debug_output.png"):
    """Updated debug function using correct helpers."""
    print("=== KIKI Pipeline Debug (UPDATED) ===")
    
    print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
    print(f"Target y shape: {y.shape}, dtype: {y.dtype}")  
    print(f"Mask m shape: {m.shape}, dtype: {m.dtype}")
    print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"y range: [{y.min():.6f}, {y.max():.6f}]")
    print(f"m range: [{m.min():.6f}, {m.max():.6f}]")
    
    x_sample = x[0:1]
    y_sample = y[0:1]
    m_sample = m[0:1]
    
    # Create zero-filled baseline using correct helper
    print("\n=== Creating Zero-Filled Baseline (UPDATED) ===")
    zero_filled_mag = create_zero_filled_baseline(x_sample)  # Uses correct ifft2
    print(f"Zero-filled magnitude range: [{zero_filled_mag.min():.6f}, {zero_filled_mag.max():.6f}]")
    
    # Run KIKI model  
    print("\n=== Running KIKI Model ===")
    model.eval()
    with torch.no_grad():
        kiki_output = model(x_sample, m_sample)
    
    print(f"KIKI output shape: {kiki_output.shape}")
    print(f"KIKI output range: [{kiki_output.min():.6f}, {kiki_output.max():.6f}]")
    
    # Convert to magnitude using correct helper
    kiki_mag = complex_magnitude(kiki_output)
    if kiki_mag.dim() == 3:
        kiki_mag = kiki_mag.unsqueeze(1)
    print(f"KIKI magnitude range: [{kiki_mag.min():.6f}, {kiki_mag.max():.6f}]")
    
    # Sanity checks
    print("\n=== Sanity Checks ===")
    target_mean = y_sample.mean().item()
    zero_filled_mean = zero_filled_mag.mean().item()
    kiki_mean = kiki_mag.mean().item()
    
    print(f"Target mean: {target_mean:.6f}")
    print(f"Zero-filled mean: {zero_filled_mean:.6f}")
    print(f"KIKI mean: {kiki_mean:.6f}")
    print(f"KIKI/Target ratio: {kiki_mean/target_mean:.6f}")
    print(f"KIKI/Zero-filled ratio: {kiki_mean/zero_filled_mean:.6f}")
    
    # Check for improvement
    if abs(kiki_mean/target_mean - 1.0) < abs(zero_filled_mean/target_mean - 1.0):
        print("SUCCESS: KIKI scale closer to target than zero-filled!")
    else:
        print("WARNING: KIKI scale still off")
    
    return {
        'target': y_sample.squeeze().cpu().numpy(),
        'zero_filled': zero_filled_mag.squeeze().cpu().numpy(),
        'kiki': kiki_mag.squeeze().cpu().numpy(),
    }

def debug_kiki_forward_pass(model, x, m, device):
    """Updated forward pass debug using correct helpers."""
    print("\n=== KIKI Forward Pass Debug (UPDATED) ===")
    
    model.eval()
    x_sample = x[0:1]
    m_sample = m[0:1]
    
    print(f"Input to KIKI: shape={x_sample.shape}, range=[{x_sample.min():.6f}, {x_sample.max():.6f}]")
    
    with torch.no_grad():
        rec = fftshift2(x_sample)
        print(f"After initial fftshift2: range=[{rec.min():.6f}, {rec.max():.6f}]")
        
        for i in range(model.n_iter):
            print(f"\n--- KIKI Iteration {i+1} (UPDATED) ---")
            
            # K-space processing
            rec = model.conv_blocks_K[i](rec)
            print(f"After K-block {i}: range=[{rec.min():.6f}, {rec.max():.6f}]")
            
            # Go to image space
            rec = fftshift2(rec)
            rec = ifft2(rec)
            print(f"After ifft2: range=[{rec.min():.6f}, {rec.max():.6f}]")
            
            # Image space processing
            rec = rec + model.conv_blocks_I[i](rec)
            print(f"After I-block {i}: range=[{rec.min():.6f}, {rec.max():.6f}]")
            
            # Data consistency
            rec = DataConsist(rec, x_sample, m_sample)
            print(f"After DataConsist: range=[{rec.min():.6f}, {rec.max():.6f}]")
            
            # Back to k-space for next iteration (if not last)
            if i < model.n_iter - 1:
                rec = fftshift2(fft2(rec))  # NO SCALING with ortho norm
                print(f"Back to k-space: range=[{rec.min():.6f}, {rec.max():.6f}]")
    
    return rec

# Add this to your training loop for debugging
def add_debug_to_training():
    """
    Add this code to your training loop to debug the first batch.
    """
    debug_code = '''
    # Add this right after loading your first batch in training:
    if global_step == 1:  # Debug first batch only
        print("\\n=== DEBUGGING FIRST BATCH ===")
        
        # Debug data flow
        debug_results = debug_kiki_pipeline(x, y, m, model, device, "debug_kiki_pipeline.png")
        
        # Debug forward pass  
        debug_output = debug_kiki_forward_pass(model, x, m, device)
        
        # Compare with actual model call
        model_output = model(x[0:1], m[0:1])
        print(f"\\nModel output matches debug: {torch.allclose(debug_output, model_output, atol=1e-6)}")
        
        print("\\n=== DEBUG COMPLETE ===\\n")
        
        # Optional: exit after first batch for analysis
        # import sys; sys.exit()
    '''
    return debug_code

print("Debug functions created. Use debug_kiki_pipeline() and debug_kiki_forward_pass() to analyze your model.")


def debug_dataconsist(input_, k, m, is_k=False):
    """
    Debug version of DataConsist to identify the problem.
    """
    print(f"\n=== DataConsist Debug ===")
    print(f"Input shape: {input_.shape}, range: [{input_.min():.6f}, {input_.max():.6f}]")
    print(f"k shape: {k.shape}, range: [{k.min():.6f}, {k.max():.6f}]")
    print(f"m shape: {m.shape}, range: [{m.min():.6f}, {m.max():.6f}]")
    print(f"is_k: {is_k}")
    
    # Store original for comparison
    orig_input = input_.clone()
    
    # Ensure device/dtype consistency
    if input_.dtype == torch.float16:
        input_work = input_.float()
        k_work = k.float() 
        m_work = m.float()
    else:
        input_work = input_
        k_work = k.to(input_.device, input_.dtype)
        m_work = m.to(input_.device, input_.dtype)
    
    print(f"After dtype conversion - k range: [{k_work.min():.6f}, {k_work.max():.6f}]")
    
    # Handle mask broadcasting
    if m_work.size(1) == 1 and k_work.size(1) == 2:
        m_work = m_work.repeat(1, 2, 1, 1)
        print(f"Broadcasted mask shape: {m_work.shape}")
    
    if is_k:
        print("Using k-space mode")
        result = input_work * m_work + k_work * (1 - m_work)
    else:
        print("Using image space mode")
        
        # Convert to (N,H,W,2) format for FFT
        input_p = input_work.permute(0, 2, 3, 1)  # (N,2,H,W) -> (N,H,W,2)
        k_p = k_work.permute(0, 2, 3, 1)
        m_p = m_work.permute(0, 2, 3, 1)
        
        print(f"Permuted input range: [{input_p.min():.6f}, {input_p.max():.6f}]")
        print(f"Permuted k range: [{k_p.min():.6f}, {k_p.max():.6f}]")
        
        # Convert input to complex for FFT
        input_complex = torch.complex(input_p[..., 0], input_p[..., 1])
        print(f"Input complex range: [{input_complex.abs().min():.6f}, {input_complex.abs().max():.6f}]")
        
        # Apply 2D FFT 
        input_k_complex = torch.fft.fft2(input_complex, norm='backward')
        print(f"After FFT range: [{input_k_complex.abs().min():.6f}, {input_k_complex.abs().max():.6f}]")
        
        # Convert back to real/imag format
        input_k_p = torch.stack([input_k_complex.real, input_k_complex.imag], dim=-1)
        print(f"FFT real/imag range: [{input_k_p.min():.6f}, {input_k_p.max():.6f}]")
        
        # Data consistency in k-space - THIS IS THE KEY STEP
        dc_k_p_before = input_k_p.clone()
        dc_k_p = input_k_p * m_p + k_p * (1 - m_p)
        print(f"DC operation:")
        print(f"  input_k_p * m_p range: [{(input_k_p * m_p).min():.6f}, {(input_k_p * m_p).max():.6f}]")
        print(f"  k_p * (1-m_p) range: [{(k_p * (1 - m_p)).min():.6f}, {(k_p * (1 - m_p)).max():.6f}]")
        print(f"  Final dc_k_p range: [{dc_k_p.min():.6f}, {dc_k_p.max():.6f}]")
        
        # Check how much k_p is affecting the result
        mask_ratio = m_p.mean().item()
        print(f"Mask ratio (fraction of k-space kept): {mask_ratio:.4f}")
        
        # Convert back to complex for IFFT
        dc_k_complex = torch.complex(dc_k_p[..., 0], dc_k_p[..., 1])
        print(f"DC complex range: [{dc_k_complex.abs().min():.6f}, {dc_k_complex.abs().max():.6f}]")
        
        # Apply 2D IFFT
        result_complex = torch.fft.ifft2(dc_k_complex, norm='backward')
        print(f"After IFFT range: [{result_complex.abs().min():.6f}, {result_complex.abs().max():.6f}]")
        
        # Convert back to (N,2,H,W) format
        result_p = torch.stack([result_complex.real, result_complex.imag], dim=-1)
        result = result_p.permute(0, 3, 1, 2)  # (N,H,W,2) -> (N,2,H,W)
    
    print(f"Final result range: [{result.min():.6f}, {result.max():.6f}]")
    print(f"Change magnitude: {(result - orig_input).abs().mean():.6f}")
    
    return result


def test_fixed_dataconsist(input_, k, m, is_k=False):
    """
    Test the fixed DataConsist and compare with original.
    """
    print(f"\n=== Testing Fixed DataConsist ===")
    print(f"Input range: [{input_.min():.6f}, {input_.max():.6f}], mean: {input_.abs().mean():.6f}")
    
    # Apply fixed data consistency
    result = DataConsist(input_, k, m, is_k)
    print(f"Fixed result range: [{result.min():.6f}, {result.max():.6f}], mean: {result.abs().mean():.6f}")
    print(f"Change magnitude: {(result - input_).abs().mean():.6f}")
    
    return result