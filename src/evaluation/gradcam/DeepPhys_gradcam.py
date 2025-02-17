import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class GradCAM_DeepPhys:
    '''
    2D CNN with a 6-channel input, specifically for the DeepPhys architecture.
    '''
    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        # Store the activations and gradients captured by the hooks
        self._activations = None
        self._gradients = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        """
        Forward hook: saves the output feature maps of the target layer.
        out: shape (B, num_channels, H', W') 
        """
        self._activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        """
        Backward hook: saves the gradients w.r.t. the output feature maps.
        grad_out[0]: shape (B, num_channels, H', W')
        """
        self._gradients = grad_out[0]

    def generate(self, x):
        """
        Generates the Grad-CAM heatmap for a given input tensor.

        Args:
            x: Input tensor, shape (B, 6, H, W).

        Returns:
            Grad-CAM heatmap of shape (H', W'), normalized to [0,1].
        """
        self.model.zero_grad()

        output = self.model(x)  # shape (B, 1)
        scalar = output[0, 0]  

        scalar.backward(retain_graph=True) # compute gradients of the scalar output

        activations = self._activations[0]  # shape (num_channels, H', W')
        gradients = self._gradients[0]      # shape (num_channels, H', W')

        weights = gradients.mean(dim=(1, 2))  # shape (num_channels,)

        grad_cam_map = torch.zeros_like(activations[0])
        for c, w in enumerate(weights):
            grad_cam_map += w * activations[c]   # weighted sum of activations
        
        grad_cam_map = F.relu(grad_cam_map) # ReLU to retain only positive contributions

        if grad_cam_map.max() > 0:
            grad_cam_map /= grad_cam_map.max() # normalize to [0,1]

        grad_cam_map = grad_cam_map.detach().cpu().numpy()
        return grad_cam_map

    def remove_hooks(self):

        self._forward_hook.remove()
        self._backward_hook.remove()

def visualize_gradcam(model, val_loader, target_layer, device):
    """
     1) Takes the first batch from val_loader
     2) Applies Grad-CAM on a single sample
     3) Displays the resulting heatmap
    """
    gradcam = GradCAM(model, target_layer)

    model.eval()

    with torch.no_grad():
        for batch in val_loader:
        
            inputs = batch[0]  # Extract inputs from the batch
            inputs = inputs.to(device)

            single_input = inputs[0].unsqueeze(0)  # shape (1, 6, H, W)
            single_input.requires_grad = True

            with torch.enable_grad():
                cam_map = gradcam.generate(single_input)  # shape (H', W')

            img_tensor = single_input[0, 3:].cpu()  # Get appearance channels
            img = img_tensor.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Smaller figure size
            axes[0].imshow(img) # Original Image
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            heatmap = axes[1].imshow(cam_map, cmap='jet') # Heatmap
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
            gradcam.remove_hooks()

            break

def visualize_gradcam_batch(model, val_loader, target_layer, device, max_samples=16):
    """
    Visualize Grad-CAM for multiple samples in a batch. 
    """
    gradcam = GradCAM(model, target_layer)
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            batch_size = min(inputs.shape[0], max_samples)

            n_cols = 2  # Original and Heatmap
            n_rows = batch_size
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2 * n_rows))  

            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for idx in range(batch_size):
                single_input = inputs[idx:idx+1]
                single_input.requires_grad = True

                with torch.enable_grad():
                    cam_map = gradcam.generate(single_input)

                img_tensor = single_input[0, 3:].cpu()  # Get appearance channels
                img = img_tensor.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())

                axes[idx, 0].imshow(img)
                axes[idx, 0].set_title(f'Sample {idx+1} - Original')
                axes[idx, 0].axis('off')
                heatmap = axes[idx, 1].imshow(cam_map, cmap='jet')
                axes[idx, 1].set_title(f'Sample {idx+1} - Heatmap')
                axes[idx, 1].axis('off')


            plt.tight_layout()
            fig.subplots_adjust(right=0.9)
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(heatmap, cax=cax)
            plt.show()

            gradcam.remove_hooks()
            break

