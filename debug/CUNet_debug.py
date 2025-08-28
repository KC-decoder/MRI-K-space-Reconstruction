import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
import json





class CUNetTrainingDebugger:
    """Comprehensive debugging pipeline for CUNet training issues"""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.debug_results = {}
        
    def run_full_debug(self, dataloader, num_samples=5):
        """Run complete debugging pipeline"""
        print("🔧 STARTING CUNET TRAINING DEBUG PIPELINE")
        print("=" * 60)
        
        # Step 1: Test basic functionality
        print("\n TESTING BASIC FUNCTIONALITY")
        self.test_basic_forward_pass()
        
        # Step 2: Debug data pipeline
        print("\n DEBUGGING DATA PIPELINE")  
        self.debug_data_pipeline(dataloader, num_samples)
        
        # Step 3: Debug individual components
        print("\n TESTING INDIVIDUAL COMPONENTS")
        self.test_individual_components(dataloader)
        
        # Step 4: Debug trained model (if available)
        if self.model_path and Path(self.model_path).exists():
            print("\n DEBUGGING TRAINED MODEL")
            model = self.debug_trained_model()
            
            # Step 5: Test trained model inference
            print("\n TESTING TRAINED MODEL INFERENCE")
            self.test_trained_model_inference(model, dataloader, num_samples)
            
        else:
            print("\n NO TRAINED MODEL PROVIDED - Testing fresh model")
            model = self.create_fresh_model()
            
        # Step 6: Training simulation test
        print("\n TRAINING SIMULATION TEST")
        self.test_training_simulation(dataloader)
        
        # Step 7: Generate diagnostic report
        print("\n GENERATING DIAGNOSTIC REPORT")
        self.generate_diagnostic_report()
        
        return self.debug_results
    
    def test_basic_forward_pass(self):
        """Test if basic forward pass works with synthetic data"""
        print(" Testing basic forward pass...")
        
        try:
            from net.unet.complex_Unet import CUNet
            
            # Create minimal model
            model = CUNet(in_channels=2, out_channels=1, base_features=8).to(self.device)
            model.eval()
            
            # Create simple test data
            batch_size = 2
            H, W = 32, 32
            
            k_space = torch.randn(batch_size, 2, H, W).to(self.device)
            mask = torch.ones(batch_size, 1, H, W).to(self.device)
            
            print(f"   Input: {k_space.shape}, Mask: {mask.shape}")
            
            with torch.no_grad():
                output = model(k_space, mask)
                
            print(f"   Output: {output.shape}")
            print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"   Output std: {output.std():.6f}")
            
            # Check for common issues
            if torch.isnan(output).any():
                print("   🚨 OUTPUT CONTAINS NaN!")
                self.debug_results['basic_forward'] = 'FAILED - NaN'
            elif torch.isinf(output).any():
                print("   🚨 OUTPUT CONTAINS Inf!")
                self.debug_results['basic_forward'] = 'FAILED - Inf'
            elif output.std() < 1e-8:
                print("   ⚠️  OUTPUT HAS VERY LOW VARIANCE")
                self.debug_results['basic_forward'] = 'WARNING - Low variance'
            else:
                print("   ✅ Basic forward pass OK")
                self.debug_results['basic_forward'] = 'PASSED'
                
        except Exception as e:
            print(f"   🚨 BASIC FORWARD PASS FAILED: {e}")
            traceback.print_exc()
            self.debug_results['basic_forward'] = f'FAILED - {str(e)}'
    
    def debug_data_pipeline(self, dataloader, num_samples=5):
        """Debug the data pipeline thoroughly"""
        print("🔍 Debugging data pipeline...")
        
        data_stats = {
            'input_stats': [],
            'target_stats': [],
            'mask_stats': [],
            'issues': []
        }
        
        try:
            for i, (X, y, mask) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                print(f"\n   Sample {i+1}:")
                print(f"   Shapes: X={X.shape}, y={y.shape}, mask={mask.shape}")
                
                # Move to device for analysis
                X = X.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32) 
                mask = mask.to(self.device, dtype=torch.float32)
                
                # Analyze input k-space
                x_stats = {
                    'min': X.min().item(),
                    'max': X.max().item(),
                    'mean': X.mean().item(),
                    'std': X.std().item(),
                    'shape': list(X.shape)
                }
                data_stats['input_stats'].append(x_stats)
                
                print(f"   X (k-space): min={x_stats['min']:.4f}, max={x_stats['max']:.4f}, mean={x_stats['mean']:.4f}, std={x_stats['std']:.4f}")
                
                # Analyze target
                y_stats = {
                    'min': y.min().item(),
                    'max': y.max().item(), 
                    'mean': y.mean().item(),
                    'std': y.std().item(),
                    'shape': list(y.shape)
                }
                data_stats['target_stats'].append(y_stats)
                
                print(f"   y (target): min={y_stats['min']:.4f}, max={y_stats['max']:.4f}, mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}")
                
                # Analyze mask
                mask_stats = {
                    'min': mask.min().item(),
                    'max': mask.max().item(),
                    'coverage': mask.mean().item(),
                    'shape': list(mask.shape)
                }
                data_stats['mask_stats'].append(mask_stats)
                
                print(f"   mask: coverage={mask_stats['coverage']*100:.1f}%, min={mask_stats['min']:.1f}, max={mask_stats['max']:.1f}")
                
                # Check for data issues
                if torch.isnan(X).any():
                    data_stats['issues'].append(f"Sample {i+1}: X contains NaN")
                    print("   🚨 X CONTAINS NaN!")
                    
                if torch.isnan(y).any():
                    data_stats['issues'].append(f"Sample {i+1}: y contains NaN")
                    print("   🚨 y CONTAINS NaN!")
                    
                if X.std() < 1e-8:
                    data_stats['issues'].append(f"Sample {i+1}: X has no variance")
                    print("   ⚠️  X HAS NO VARIANCE!")
                    
                if y.std() < 1e-8:
                    data_stats['issues'].append(f"Sample {i+1}: y has no variance")
                    print("   ⚠️  y HAS NO VARIANCE!")
                    
                # Check scale mismatch
                if abs(X.std().item() - y.std().item()) > 10:
                    data_stats['issues'].append(f"Sample {i+1}: Large scale mismatch X vs y")
                    print(f"   ⚠️  LARGE SCALE MISMATCH: X.std={X.std():.4f} vs y.std={y.std():.4f}")
                
                # Test data preparation
                try:
                    from net.unet.complex_Unet import prepare_data_for_hard_dc
                    k_prep, y_prep, mask_prep = prepare_data_for_hard_dc(X, y, mask)
                    print(f"   ✅ Data preparation successful: k={k_prep.shape}, y={y_prep.shape}, mask={mask_prep.shape}")
                except Exception as e:
                    data_stats['issues'].append(f"Sample {i+1}: Data preparation failed - {str(e)}")
                    print(f"   🚨 DATA PREPARATION FAILED: {e}")
            
            self.debug_results['data_pipeline'] = data_stats
            
            # Summary
            if data_stats['issues']:
                print(f"\n   🚨 FOUND {len(data_stats['issues'])} DATA ISSUES!")
                for issue in data_stats['issues']:
                    print(f"     - {issue}")
            else:
                print("\n   ✅ Data pipeline looks good!")
                
        except Exception as e:
            print(f"   🚨 DATA PIPELINE DEBUG FAILED: {e}")
            traceback.print_exc()
            self.debug_results['data_pipeline'] = {'error': str(e)}
    
    def test_individual_components(self, dataloader):
        """Test individual model components"""
        print("🔍 Testing individual components...")
        
        component_results = {}
        
        # Get sample data
        X, y, mask = next(iter(dataloader))
        X = X.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)
        
        # Test 1: FFT operations
        print("   Testing FFT operations...")
        try:
            from net.unet.complex_Unet import fft2c_2ch, ifft2c_2ch
            
            # Convert to proper format if needed
            if X.shape[1] == 2:  # K-space input
                test_img = ifft2c_2ch(X)  # Convert to image
                test_k = fft2c_2ch(test_img)  # Convert back to k-space
                
                roundtrip_error = torch.mean((X - test_k) ** 2)
                print(f"      FFT roundtrip error: {roundtrip_error.item():.2e}")
                
                if roundtrip_error < 1e-10:
                    component_results['fft_ops'] = 'PASSED'
                    print("      ✅ FFT operations work correctly")
                else:
                    component_results['fft_ops'] = f'WARNING - High error: {roundtrip_error.item():.2e}'
                    print("      ⚠️  FFT operations have high roundtrip error")
            else:
                component_results['fft_ops'] = 'SKIPPED - Wrong input format'
                print("      ⚠️  Skipped - input not in k-space format")
                
        except Exception as e:
            component_results['fft_ops'] = f'FAILED - {str(e)}'
            print(f"      🚨 FFT operations failed: {e}")
        
        # Test 2: Complex operations
        print("   Testing complex operations...")
        try:
            from net.unet.complex_Unet import ComplexConv2d, ComplexBatchNorm2d
            
            # Test ComplexConv2d
            conv = ComplexConv2d(1, 4, 3, padding=1).to(self.device)
            test_input = torch.randn(1, 2, 32, 32).to(self.device)
            
            conv_out = conv(test_input)
            print(f"      ComplexConv2d: {test_input.shape} -> {conv_out.shape}")
            
            if torch.isnan(conv_out).any():
                component_results['complex_ops'] = 'FAILED - ComplexConv2d produces NaN'
                print("      🚨 ComplexConv2d produces NaN")
            elif conv_out.std() < 1e-8:
                component_results['complex_ops'] = 'WARNING - ComplexConv2d low variance'
                print("      ⚠️  ComplexConv2d has very low output variance")
            else:
                component_results['complex_ops'] = 'PASSED'
                print("      ✅ Complex operations work correctly")
                
        except Exception as e:
            component_results['complex_ops'] = f'FAILED - {str(e)}'
            print(f"      🚨 Complex operations failed: {e}")
        
        # Test 3: Data consistency
        print("   Testing data consistency...")
        try:
            from net.unet.complex_Unet import CUNet
            
            model = CUNet(in_channels=2, out_channels=1, base_features=8, use_data_consistency=True).to(self.device)
            
            with torch.no_grad():
                # Test with DC enabled
                out_with_dc = model(X[:1], mask[:1])  # Small batch
                
                # Test without DC
                model.use_data_consistency = False
                out_without_dc = model(X[:1], mask[:1])
                
                dc_diff = torch.mean((out_with_dc - out_without_dc) ** 2)
                print(f"      DC vs no-DC difference: {dc_diff.item():.6f}")
                
                if torch.isnan(out_with_dc).any():
                    component_results['data_consistency'] = 'FAILED - DC produces NaN'
                    print("      🚨 Data consistency produces NaN")
                elif dc_diff > 1e-6:
                    component_results['data_consistency'] = 'PASSED - DC makes difference'
                    print("      ✅ Data consistency is working (creates difference)")
                else:
                    component_results['data_consistency'] = 'WARNING - DC makes no difference'
                    print("      ⚠️  Data consistency makes no noticeable difference")
                    
        except Exception as e:
            component_results['data_consistency'] = f'FAILED - {str(e)}'
            print(f"      🚨 Data consistency test failed: {e}")
        
        self.debug_results['component_tests'] = component_results
    
    def debug_trained_model(self):
        """Debug the trained model weights and state"""
        print("🔍 Debugging trained model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check training metadata
            print(f"   Checkpoint info:")
            print(f"     Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"     Loss: {checkpoint.get('loss', 'N/A')}")
            
            # Load model
            from net.unet.complex_Unet import CUNet
            model = CUNet(in_channels=2, out_channels=1, base_features=32, use_data_consistency=True).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Analyze weights
            weight_analysis = self.analyze_model_weights(model)
            self.debug_results['trained_model_weights'] = weight_analysis
            
            return model
            
        except Exception as e:
            print(f"   🚨 Failed to load trained model: {e}")
            self.debug_results['trained_model_weights'] = {'error': str(e)}
            return self.create_fresh_model()
    
    def analyze_model_weights(self, model):
        """Analyze model weights for common issues"""
        print("   Analyzing model weights...")
        
        weight_stats = {
            'total_params': 0,
            'zero_params': 0,
            'nan_params': 0,
            'problematic_layers': [],
            'layer_stats': {}
        }
        
        for name, param in model.named_parameters():
            total_params = param.numel()
            zero_params = (param.abs() < 1e-8).sum().item()
            nan_params = torch.isnan(param).sum().item()
            inf_params = torch.isinf(param).sum().item()
            
            weight_stats['total_params'] += total_params
            weight_stats['zero_params'] += zero_params
            weight_stats['nan_params'] += nan_params
            
            layer_stat = {
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'zero_ratio': zero_params / total_params,
                'nan_count': nan_params,
                'inf_count': inf_params
            }
            
            weight_stats['layer_stats'][name] = layer_stat
            
            # Check for problems
            if nan_params > 0:
                weight_stats['problematic_layers'].append(f"{name}: {nan_params} NaN values")
                print(f"      🚨 {name}: Contains {nan_params} NaN values!")
                
            if inf_params > 0:
                weight_stats['problematic_layers'].append(f"{name}: {inf_params} Inf values") 
                print(f"      🚨 {name}: Contains {inf_params} Inf values!")
                
            if layer_stat['std'] < 1e-8:
                weight_stats['problematic_layers'].append(f"{name}: Very low variance ({layer_stat['std']:.2e})")
                print(f"      ⚠️  {name}: Very low variance ({layer_stat['std']:.2e})")
                
            if layer_stat['zero_ratio'] > 0.9:
                weight_stats['problematic_layers'].append(f"{name}: {layer_stat['zero_ratio']*100:.1f}% zeros")
                print(f"      ⚠️  {name}: {layer_stat['zero_ratio']*100:.1f}% zeros")
        
        print(f"   Weight summary:")
        print(f"     Total parameters: {weight_stats['total_params']:,}")
        print(f"     Near-zero parameters: {weight_stats['zero_params']:,} ({100*weight_stats['zero_params']/weight_stats['total_params']:.1f}%)")
        print(f"     NaN parameters: {weight_stats['nan_params']:,}")
        print(f"     Problematic layers: {len(weight_stats['problematic_layers'])}")
        
        return weight_stats
    
    def test_trained_model_inference(self, model, dataloader, num_samples=3):
        """Test trained model inference"""
        print("🔍 Testing trained model inference...")
        
        model.eval()
        inference_results = {'samples': [], 'issues': []}
        
        with torch.no_grad():
            for i, (X, y, mask) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                X = X.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                mask = mask.to(self.device, dtype=torch.float32)
                
                print(f"   Sample {i+1}:")
                
                try:
                    # Prepare data
                    from net.unet.complex_Unet import prepare_data_for_hard_dc
                    k_prep, y_prep, mask_prep = prepare_data_for_hard_dc(X, y, mask)
                    
                    # Forward pass
                    pred = model(k_prep, mask_prep)
                    
                    # Calculate basic metrics
                    mse = torch.mean((pred - y_prep) ** 2).item()
                    pred_std = pred.std().item()
                    target_std = y_prep.std().item()
                    
                    sample_result = {
                        'mse': mse,
                        'pred_std': pred_std,
                        'target_std': target_std,
                        'pred_range': [pred.min().item(), pred.max().item()],
                        'target_range': [y_prep.min().item(), y_prep.max().item()]
                    }
                    
                    inference_results['samples'].append(sample_result)
                    
                    print(f"     MSE: {mse:.6f}")
                    print(f"     Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}], std: {pred_std:.4f}")
                    print(f"     Target range: [{y_prep.min().item():.4f}, {y_prep.max().item():.4f}], std: {target_std:.4f}")
                    
                    # Check for issues
                    if torch.isnan(pred).any():
                        inference_results['issues'].append(f"Sample {i+1}: Prediction contains NaN")
                        print("     🚨 PREDICTION CONTAINS NaN!")
                        
                    if pred_std < 1e-6:
                        inference_results['issues'].append(f"Sample {i+1}: Prediction has no variance")
                        print("     🚨 PREDICTION HAS NO VARIANCE!")
                        
                    if mse < 1e-10:
                        inference_results['issues'].append(f"Sample {i+1}: Suspiciously low MSE")
                        print("     ⚠️  SUSPICIOUSLY LOW MSE (might be overfitting to zero)")
                        
                    if abs(pred.mean().item()) < 1e-8 and abs(pred.std().item()) < 1e-8:
                        inference_results['issues'].append(f"Sample {i+1}: Model outputs essentially zero")
                        print("     🚨 MODEL OUTPUTS ESSENTIALLY ZERO!")
                        
                except Exception as e:
                    inference_results['issues'].append(f"Sample {i+1}: Inference failed - {str(e)}")
                    print(f"     🚨 INFERENCE FAILED: {e}")
        
        self.debug_results['trained_model_inference'] = inference_results
        
        if inference_results['issues']:
            print(f"   🚨 Found {len(inference_results['issues'])} inference issues!")
        else:
            print("   ✅ Trained model inference looks reasonable")
    
    def test_training_simulation(self, dataloader):
        """Simulate a few training steps to check training dynamics"""
        print("🔍 Testing training simulation...")
        
        try:
            from net.unet.complex_Unet import CUNet, CUNetLoss
            
            # Create fresh model
            model = CUNet(in_channels=2, out_channels=1, base_features=16).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = CUNetLoss()
            
            model.train()
            
            training_results = {
                'losses': [],
                'grad_norms': [],
                'weight_changes': [],
                'issues': []
            }
            
            # Get initial weights for comparison
            initial_weights = {}
            for name, param in model.named_parameters():
                initial_weights[name] = param.clone().detach()
            
            # Simulate several training steps
            for step, (X, y, mask) in enumerate(dataloader):
                if step >= 5:  # Just test a few steps
                    break
                    
                X = X.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                mask = mask.to(self.device, dtype=torch.float32)
                
                print(f"   Training step {step + 1}:")
                
                try:
                    # Prepare data
                    from net.unet.complex_Unet import prepare_data_for_hard_dc
                    k_prep, y_prep, mask_prep = prepare_data_for_hard_dc(X, y, mask)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(k_prep, mask_prep)
                    loss = criterion(pred, y_prep)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Calculate gradient norm
                    grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2) ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    optimizer.step()
                    
                    training_results['losses'].append(loss.item())
                    training_results['grad_norms'].append(grad_norm)
                    
                    print(f"     Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}")
                    
                    # Check for training issues
                    if torch.isnan(loss):
                        training_results['issues'].append(f"Step {step+1}: Loss is NaN")
                        print("     🚨 LOSS IS NaN!")
                        break
                        
                    if grad_norm < 1e-8:
                        training_results['issues'].append(f"Step {step+1}: Gradients vanishing")
                        print("     🚨 GRADIENTS VANISHING!")
                        
                    if grad_norm > 100:
                        training_results['issues'].append(f"Step {step+1}: Gradients exploding")
                        print("     🚨 GRADIENTS EXPLODING!")
                        
                except Exception as e:
                    training_results['issues'].append(f"Step {step+1}: Training step failed - {str(e)}")
                    print(f"     🚨 TRAINING STEP FAILED: {e}")
                    break
            
            # Check weight changes
            total_weight_change = 0.0
            for name, param in model.named_parameters():
                if name in initial_weights:
                    change = torch.mean((param - initial_weights[name]) ** 2).item()
                    total_weight_change += change
            
            training_results['weight_changes'] = total_weight_change
            
            print(f"   Training simulation summary:")
            print(f"     Losses: {training_results['losses']}")
            print(f"     Average grad norm: {np.mean(training_results['grad_norms']):.6f}")
            print(f"     Total weight change: {total_weight_change:.6e}")
            
            if total_weight_change < 1e-10:
                training_results['issues'].append("Weights barely changed during training")
                print("     🚨 WEIGHTS BARELY CHANGED!")
            
            self.debug_results['training_simulation'] = training_results
            
        except Exception as e:
            print(f"   🚨 TRAINING SIMULATION FAILED: {e}")
            traceback.print_exc()
            self.debug_results['training_simulation'] = {'error': str(e)}
    
    def create_fresh_model(self):
        """Create a fresh model for testing"""
        from net.unet.complex_Unet import CUNet
        return CUNet(in_channels=2, out_channels=1, base_features=16).to(self.device)
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report with recommendations"""
        print("📋 GENERATING DIAGNOSTIC REPORT...")
        
        # Analyze all collected debug results
        issues = []
        warnings = []
        recommendations = []
        
        # Basic functionality issues
        if self.debug_results.get('basic_forward', '').startswith('FAILED'):
            issues.append("❌ Basic forward pass fails - fundamental model issue")
            recommendations.append("🔧 Check model architecture and dependencies")
        
        # Data pipeline issues
        data_pipeline = self.debug_results.get('data_pipeline', {})
        if isinstance(data_pipeline, dict) and data_pipeline.get('issues'):
            for issue in data_pipeline['issues']:
                issues.append(f"❌ Data issue: {issue}")
            recommendations.append("🔧 Fix data preprocessing and normalization")
        
        # Component issues
        component_tests = self.debug_results.get('component_tests', {})
        for component, result in component_tests.items():
            if result.startswith('FAILED'):
                issues.append(f"❌ {component} component failed: {result}")
                if component == 'fft_ops':
                    recommendations.append("🔧 Check FFT implementation and numerical precision")
                elif component == 'complex_ops':
                    recommendations.append("🔧 Check complex operations and weight initialization")
        
        # Training issues
        training_sim = self.debug_results.get('training_simulation', {})
        if isinstance(training_sim, dict) and training_sim.get('issues'):
            for issue in training_sim['issues']:
                issues.append(f"❌ Training issue: {issue}")
            recommendations.append("🔧 Adjust learning rate, loss function, or model architecture")
        
        # Trained model issues
        if 'trained_model_inference' in self.debug_results:
            inference = self.debug_results['trained_model_inference']
            if inference.get('issues'):
                for issue in inference['issues']:
                    issues.append(f"❌ Inference issue: {issue}")
                recommendations.append("🔧 Model may not have trained properly - retrain with fixes")
        
        # Generate report
        print("\n" + "="*60)
        print("🎯 DIAGNOSTIC REPORT")
        print("="*60)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Issues found: {len(issues)}")
        print(f"   Warnings: {len(warnings)}")
        print(f"   Recommendations: {len(recommendations)}")
        
        if issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in issues[:10]:  # Show top 10
                print(f"   {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues)-10} more issues")
        
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in warnings[:5]:
                print(f"   {warning}")
        
        if recommendations:
            print(f"\n🔧 RECOMMENDED FIXES:")
            for i, rec in enumerate(recommendations[:10], 1):
                print(f"   {i}. {rec}")
        
        # Additional specific recommendations
        print(f"\n💡 SPECIFIC DEBUGGING STEPS:")
        print(f"   1. Check if input k-space data is properly normalized")
        print(f"   2. Verify target images are magnitude (not complex)")
        print(f"   3. Test with simpler loss function (MSE only)")
        print(f"   4. Try training without data consistency first")
        print(f"   5. Use smaller model (base_features=8) for debugging")
        print(f"   6. Check for gradient clipping needs")
        print(f"   7. Verify FFT/IFFT operations preserve data range")
        print(f"   8. Test with real-valued U-Net first")
        
        # Save detailed report
        self.save_detailed_report()
        
        return {
            'issues': issues,
            'warnings': warnings, 
            'recommendations': recommendations
        }
    
    def save_detailed_report(self):
        """Save detailed debug results to file"""
        try:
            report_path = "cunet_debug_report.json"
            with open(report_path, 'w') as f:
                # Convert any tensor values to lists for JSON serialization
                json_safe_results = self._make_json_safe(self.debug_results)
                json.dump(json_safe_results, f, indent=2)
            print(f"   💾 Detailed report saved: {report_path}")
        except Exception as e:
            print(f"   ⚠️  Could not save report: {e}")
    
    def _make_json_safe(self, obj):
        """Convert tensors and other non-JSON types for saving"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

def visualize_debug_results(model, dataloader, device, save_path=None):
    """Create comprehensive visualization of debugging results"""
    print("🎨 Creating debug visualizations...")
    
    model.eval()
    with torch.no_grad():
        X, y, mask = next(iter(dataloader))
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        
        try:
            from net.unet.complex_Unet import prepare_data_for_hard_dc
            k_prep, y_prep, mask_prep = prepare_data_for_hard_dc(X, y, mask)
            pred = model(k_prep, mask_prep)
            
            # Convert to numpy
            pred_np = pred[0, 0].cpu().numpy()
            target_np = y_prep[0, 0].cpu().numpy()
            k_real = k_prep[0, 0].cpu().numpy()
            k_imag = k_prep[0, 1].cpu().numpy()
            k_mag = np.sqrt(k_real**2 + k_imag**2)
            mask_np = mask_prep[0, 0].cpu().numpy()
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Row 1: Input data
            im1 = axes[0, 0].imshow(k_mag, cmap='gray')
            axes[0, 0].set_title(f'K-space Magnitude\nRange: [{k_mag.min():.2e}, {k_mag.max():.2e}]')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Sampling Mask\nCoverage: {mask_np.mean()*100:.1f}%')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(target_np, cmap='gray')
            axes[0, 2].set_title(f'Target Image\nRange: [{target_np.min():.3f}, {target_np.max():.3f}]')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # Zero-filled reconstruction for comparison
            k_complex = k_real + 1j * k_imag
            zero_filled = np.abs(np.fft.ifft2(k_complex))
            im4 = axes[0, 3].imshow(zero_filled, cmap='gray')
            axes[0, 3].set_title(f'Zero-filled\nRange: [{zero_filled.min():.3f}, {zero_filled.max():.3f}]')
            plt.colorbar(im4, ax=axes[0, 3])
            
            # Row 2: Model outputs and analysis
            im5 = axes[1, 0].imshow(pred_np, cmap='gray')
            axes[1, 0].set_title(f'Model Output\nRange: [{pred_np.min():.2e}, {pred_np.max():.2e}]')
            plt.colorbar(im5, ax=axes[1, 0])
            
            # Difference from target
            diff = np.abs(pred_np - target_np)
            im6 = axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title(f'|Prediction - Target|\nMean: {diff.mean():.2e}')
            plt.colorbar(im6, ax=axes[1, 1])
            
            # Difference from zero-filled
            diff_zf = np.abs(pred_np - zero_filled)
            im7 = axes[1, 2].imshow(diff_zf, cmap='hot')
            axes[1, 2].set_title(f'|Prediction - Zero-filled|\nMean: {diff_zf.mean():.2e}')
            plt.colorbar(im7, ax=axes[1, 2])
            
            # Histogram comparison
            axes[1, 3].hist(pred_np.flatten(), bins=50, alpha=0.7, label='Prediction', density=True)
            axes[1, 3].hist(target_np.flatten(), bins=50, alpha=0.7, label='Target', density=True)
            axes[1, 3].set_title('Value Distribution')
            axes[1, 3].set_xlabel('Pixel Value')
            axes[1, 3].set_ylabel('Density')
            axes[1, 3].legend()
            axes[1, 3].set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   💾 Debug visualization saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"   🚨 Visualization failed: {e}")

# Usage function
def run_complete_cunet_debug(model_path, dataloader, device='cuda'):
    """Run the complete CUNet debugging pipeline"""
    debugger = CUNetTrainingDebugger(model_path, device)
    results = debugger.run_full_debug(dataloader, num_samples=3)
    
    # Create visualizations if model loads
    if model_path and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            from net.unet.complex_Unet import CUNet
            model = CUNet(in_channels=2, out_channels=1, base_features=32, use_data_consistency=True).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            visualize_debug_results(model, dataloader, device, "cunet_debug_visualization.png")
        except:
            print("   ⚠️  Could not create visualization")
    
    return results

# Quick debugging function
def quick_cunet_debug(dataloader, device='cuda'):
    """Quick debugging for immediate issues"""
    print("⚡ QUICK CUNET DEBUG")
    print("-" * 40)
    
    debugger = CUNetTrainingDebugger(None, device)
    
    # Test 1: Basic forward pass
    debugger.test_basic_forward_pass()
    
    # Test 2: Data pipeline (just 1 sample)
    debugger.debug_data_pipeline(dataloader, num_samples=1)
    
    # Test 3: Quick training simulation
    debugger.test_training_simulation(dataloader)
    
    print("\n⚡ Quick debug completed!")
    return debugger.debug_results


def debug_cunet_training_issues(model_path=None, dataloader=None):
    """
    Main function to debug CUNet training issues
    Integrate this into your existing training pipeline
    """
    
    print("🚀 DEBUGGING CUNET TRAINING ISSUES")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Method 1: Quick debug (if you just want to find immediate issues)
    if input("Run quick debug first? (y/n): ").lower() == 'y':
        print("\n🏃 Running quick debug...")
        quick_results = quick_cunet_debug(dataloader, device)
        
        # Check if there are critical issues that need fixing first
        basic_forward = quick_results.get('basic_forward', '')
        if basic_forward.startswith('FAILED'):
            print("🚨 CRITICAL: Basic forward pass fails!")
            print("Fix model architecture issues before proceeding.")
            return
    
    # Method 2: Complete debug (comprehensive analysis)
    print("\n🔬 Running complete debugging pipeline...")
    
    # Run the full debug
    results = run_complete_cunet_debug(model_path, dataloader, device)
    
    # Analyze results and provide specific guidance
    print("\n" + "="*50)
    print("🎯 PERSONALIZED RECOMMENDATIONS")
    print("="*50)
    
    provide_specific_recommendations(results)
    
    return results

def provide_specific_recommendations(debug_results):
        """Provide specific recommendations based on debug results"""
        
        # Check for the most common issues
        data_issues = debug_results.get('data_pipeline', {}).get('issues', [])
        training_issues = debug_results.get('training_simulation', {}).get('issues', [])
        component_issues = debug_results.get('component_tests', {})
        
        print("Based on your results, here's what to fix first:")
        
        # Priority 1: Data issues
        if data_issues:
            print("\n🥇 PRIORITY 1: Fix Data Pipeline Issues")
            for issue in data_issues[:3]:
                print(f"   • {issue}")
            print("\n   🔧 RECOMMENDED ACTIONS:")
            print("   1. Check your data normalization - ensure k-space and targets have appropriate scales")
            print("   2. Verify target images are magnitude (not k-space)")
            print("   3. Ensure masks are binary (0 or 1) and have reasonable coverage")
            print("   4. Test with a simple dataset first (synthetic data)")
            
        # Priority 2: Training dynamics
        elif training_issues:
            print("\n🥈 PRIORITY 2: Fix Training Dynamics")
            
            losses = debug_results.get('training_simulation', {}).get('losses', [])
            grad_norms = debug_results.get('training_simulation', {}).get('grad_norms', [])
            
            if grad_norms and all(g < 1e-6 for g in grad_norms):
                print("   🚨 VANISHING GRADIENTS DETECTED!")
                print("   🔧 SOLUTIONS:")
                print("   1. Reduce learning rate by 10x")
                print("   2. Check weight initialization")
                print("   3. Use gradient clipping")
                print("   4. Simplify model architecture")
                
            elif grad_norms and any(g > 100 for g in grad_norms):
                print("   🚨 EXPLODING GRADIENTS DETECTED!")
                print("   🔧 SOLUTIONS:")
                print("   1. Add gradient clipping (clip at 1.0)")
                print("   2. Reduce learning rate by 100x")
                print("   3. Check for numerical instabilities in FFT operations")
                
            elif losses and len(losses) > 1 and not (losses[-1] < losses[0]):
                print("   ⚠️  LOSS NOT DECREASING!")
                print("   🔧 SOLUTIONS:")
                print("   1. Try simpler loss function (MSE only)")
                print("   2. Disable data consistency temporarily")
                print("   3. Check if targets match model outputs")
                
        # Priority 3: Component issues
        elif any('FAILED' in str(result) for result in component_issues.values()):
            print("\n🥉 PRIORITY 3: Fix Component Issues")
            
            if component_issues.get('fft_ops', '').startswith('FAILED'):
                print("   🚨 FFT OPERATIONS FAILING!")
                print("   🔧 SOLUTIONS:")
                print("   1. Check PyTorch version compatibility")
                print("   2. Verify input tensor formats")
                print("   3. Test FFT operations on CPU vs GPU")
                
            if component_issues.get('complex_ops', '').startswith('FAILED'):
                print("   🚨 COMPLEX OPERATIONS FAILING!")
                print("   🔧 SOLUTIONS:")
                print("   1. Check ComplexConv2d implementation")
                print("   2. Verify weight initialization")
                print("   3. Test with real-valued convolutions first")
        
        # If model outputs are zero/constant
        inference_issues = debug_results.get('trained_model_inference', {}).get('issues', [])
        if any('zero' in issue.lower() or 'variance' in issue.lower() for issue in inference_issues):
            print("\n🚨 CRITICAL: Model Outputs Are Zero/Constant!")
            print("   This usually means:")
            print("   1. 💀 Dead neurons due to poor initialization")
            print("   2. 💀 Learning rate too high (weights exploded)")
            print("   3. 💀 Data preprocessing destroyed information")
            print("   4. 💀 Loss function is inappropriate")
            print("\n   🔧 EMERGENCY FIXES:")
            print("   1. Restart training with lr=1e-5")
            print("   2. Use Xavier/He initialization")
            print("   3. Test with identity target (input=output)")
            print("   4. Switch to basic MSE loss only")
        
        # General recommendations
        print("\n💡 GENERAL DEBUGGING WORKFLOW:")
        print("   1. Test with synthetic/simple data first")
        print("   2. Start with smallest possible model")
        print("   3. Use basic MSE loss initially")
        print("   4. Disable data consistency during debugging")
        print("   5. Check that loss decreases on training set")
        print("   6. Gradually add complexity back")

def create_synthetic_test_data(batch_size=4, size=64, device='cuda'):
        """Create synthetic data for debugging"""
        print("🧪 Creating synthetic test data for debugging...")
        
        # Create simple k-space patterns
        k_space = torch.zeros(batch_size, 2, size, size).to(device)
        
        # Add some structure to k-space (Gaussian blob in center)
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        y, x = y.to(device), x.to(device)
        
        gaussian = torch.exp(-((x - center)**2 + (y - center)**2) / (size/4)**2)
        
        # Real part
        k_space[:, 0, :, :] = gaussian.unsqueeze(0).repeat(batch_size, 1, 1)
        # Imaginary part (smaller)
        k_space[:, 1, :, :] = 0.5 * gaussian.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Create corresponding image (IFFT of k-space)
        k_complex = torch.complex(k_space[:, 0], k_space[:, 1])
        img_complex = torch.fft.ifft2(k_complex, norm='ortho')
        img_magnitude = torch.abs(img_complex).unsqueeze(1)  # (B, 1, H, W)
        
        # Create reasonable masks (radial)
        masks = torch.zeros(batch_size, 1, size, size).to(device)
        distances = torch.sqrt((x - center)**2 + (y - center)**2)
        for i in range(batch_size):
            radius = size // 4 + i * 2  # Different radii for variety
            masks[i, 0] = (distances < radius).float()
        
        print(f"   K-space range: [{k_space.min():.4f}, {k_space.max():.4f}]")
        print(f"   Image range: [{img_magnitude.min():.4f}, {img_magnitude.max():.4f}]")
        print(f"   Mask coverage: {masks.mean().item()*100:.1f}%")
        
        # Create a simple dataset
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, k_space, img_magnitude, masks):
                self.k_space = k_space
                self.img_magnitude = img_magnitude
                self.masks = masks
                
            def __len__(self):
                return len(self.k_space)
                
            def __getitem__(self, idx):
                return self.k_space[idx], self.img_magnitude[idx], self.masks[idx]
        
        dataset = SyntheticDataset(k_space, img_magnitude, masks)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        return dataloader

def test_with_identity_mapping():
        """Test if model can learn identity mapping (input=output)"""
        print("🔍 Testing identity mapping capability...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        batch_size = 4
        size = 64
        
        # Create k-space
        k_space = torch.randn(batch_size, 2, size, size).to(device)
        
        # Target is k-space magnitude
        k_mag = torch.sqrt(k_space[:, 0]**2 + k_space[:, 1]**2).unsqueeze(1)
        
        # Full mask (no undersampling)
        mask = torch.ones(batch_size, 1, size, size).to(device)
        
        class IdentityDataset(torch.utils.data.Dataset):
            def __init__(self, k_space, k_mag, mask):
                self.k_space = k_space
                self.k_mag = k_mag
                self.mask = mask
                
            def __len__(self):
                return len(self.k_space)
                
            def __getitem__(self, idx):
                return self.k_space[idx], self.k_mag[idx], self.mask[idx]
        
        dataset = IdentityDataset(k_space, k_mag, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Run debug on identity task
        debugger = CUNetTrainingDebugger(None, device)
        debugger.test_training_simulation(dataloader)
        
        print("   If this fails, the model architecture has fundamental issues")
        print("   If this works, the problem is in your data or loss function")

def step_by_step_debug_guide():
        """Interactive step-by-step debugging guide"""
        print("\n🎯 STEP-BY-STEP DEBUGGING GUIDE")
        print("=" * 50)
        
        steps = [
            ("1. Run basic tests", "Check if model architecture works at all"),
            ("2. Test with synthetic data", "Eliminate data issues"),
            ("3. Test identity mapping", "Check if model can learn anything"),
            ("4. Check training dynamics", "Analyze gradients and loss"),
            ("5. Debug real data", "Find issues in actual dataset"),
            ("6. Fix training hyperparameters", "Optimize learning rate, loss, etc."),
            ("7. Test trained model", "Verify inference works correctly")
        ]
        
        print("Follow these steps in order:")
        for step, description in steps:
            print(f"   {step}: {description}")
        
        print(f"\nStart with step 1 and only proceed if the previous step passes!")