import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

class GradCAM_CNN:
    '''
    2D CNN version, specifically for the Sequential CNN architecture.
    Structured to handle (B,3,H,W).
    '''
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        # Placeholders for the forward pass activation & backward pass gradients
        self.activation = None
        self.gradients  = None
        
        # Attach the hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        # output = feature maps
        self.activation = output
    
    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] = gradients w.r.t. the feature maps
        self.gradients = grad_out[0]
    
    def generate(self, x):
        """
        x: input image of shape (B, 3, H, W)
        Returns a Grad-CAM heatmap of shape (H', W') for each image in the batch.
        Typically, B=1 if you want to generate a single heatmap.
        """
        # Forward
        self.model.zero_grad()
        output = self.model(x)  # shape: (B, 1)

        if output.shape[0] == 1:
            scalar = output[0, 0]
        else:
            scalar = output.mean()
        
        # Backprop from that scalar
        scalar.backward(retain_graph=True)

        cams = []
        for i in range(self.activation.size(0)):
            activation_i = self.activation[i]  # shape (num_channels, H', W')
            gradients_i  = self.gradients[i]   # shape (num_channels, H', W')

            # 1) Global-average gradient -> one weight per channel
            weights = torch.mean(gradients_i, dim=(1, 2))  # shape (num_channels,)
            
            # 2) Combine channels
            #    grad_cam_map = sum_i( weight_i * activation_i )
            grad_cam_map = torch.zeros_like(activation_i[0])
            for c, w in enumerate(weights):
                grad_cam_map += w * activation_i[c]
            
            # 3) Apply ReLU
            grad_cam_map = F.relu(grad_cam_map)
            
            # 4) Normalize between 0 and 1
            if grad_cam_map.max() > 0:
                grad_cam_map = grad_cam_map / grad_cam_map.max()
            
            grad_cam_map = grad_cam_map.detach().cpu().numpy()
            cams.append(grad_cam_map)
        
        return cams

    def remove_hooks(self):
        """Call this if you're done using GradCAM, to avoid dangling references."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def visualize_gradcam(model, 
                      val_loader, 
                      gradcam_class, 
                      target_layer, 
                      num_images=15, 
                      device=None, 
                      mean=[0.5, 0.5, 0.5], 
                      std=[0.5, 0.5, 0.5],
                      save_plots=False,
                      save_dir='gradcam_plots'):
  

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    gradcam = gradcam_class(model, target_layer) # choose conv layer
    current_image = 0

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (window_frames, window_bvp, window_eda) in enumerate(tqdm(val_loader, desc="Processing Batches")):
        
        B, window_size, C, H, W = window_frames.shape

        for b in range(B):
            for t in range(window_size):
                if current_image >= num_images:
                    break  # breaks if current_image reached

            
                frame_3d = window_frames[b, t] 
                frame_input = frame_3d.unsqueeze(0).to(device).requires_grad_(True)

                
                with torch.enable_grad():
                    cams = gradcam.generate(frame_input)

                if isinstance(cams, torch.Tensor):
                
                    cam = cams[0].detach().cpu().numpy()  # Shape: (H', W')
                elif isinstance(cams, (np.ndarray, list)):
                
                    cam = cams[0]
                else:
                    raise TypeError(f"Unsupported type for cams: {type(cams)}")

                
                cam_resized = Image.fromarray(cam).resize((W, H), resample=Image.BILINEAR)
                cam_resized = np.array(cam_resized)
                cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                cam_enhanced = np.uint8(255 * cam_normalized)
                cam_color = plt.cm.jet(cam_normalized)[:, :, :3]  

                frame_cpu = frame_input[0].detach().cpu().clone()
                for c in range(C):
                    frame_cpu[c] = frame_cpu[c] * std[c] + mean[c]
                frame_cpu = frame_cpu.clamp(0, 1)
                img_np = frame_cpu.permute(1, 2, 0).numpy()  
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                
                ax[0].imshow(img_np)
                ax[0].axis('off')  
                ax[0].set_title("Original Image")

                ax[1].imshow(img_np)
                ax[1].imshow(cam_color, cmap='jet', alpha=0.5)  
                ax[1].axis('off')  
                ax[1].set_title("Grad-CAM Heatmap")
                plt.tight_layout()

                if save_plots:
                    plot_path = os.path.join(save_dir, f"gradcam_image_{current_image + 1}.png")
                    plt.savefig(plot_path)
                    plt.close(fig)
                else:
                    
                    plt.show()

                current_image += 1

                if current_image >= num_images:
                    break  

        if current_image >= num_images:
            break 


    gradcam.remove_hooks()
