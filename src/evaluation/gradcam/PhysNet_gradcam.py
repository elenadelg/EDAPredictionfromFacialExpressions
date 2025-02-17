import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class GradCAM3D_Physnet:
    """
    Grad-CAM for 3D convolutional models, specifically for the PhysNet architecture.
    3D CNN input (B,C,D,H,W), spatiotemporal data, plus multi-input handling.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        self.gradients   = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._forward_hook_fn)
        self._backward_hook = target_layer.register_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(self, module, input, output):
        self.activations = output  

    def _backward_hook_fn(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  

    def remove_hooks(self):
        """Call this when done to avoid duplicate hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def forward(self, sequences, bvp_inputs):
        """Simply run the forward pass on the model."""
        return self.model(sequences, bvp_inputs)

    def backward(self, output):
        """
        Run backward pass given a scalar output. 
        If output is (B,), pass a vector of ones so each sample
        in the batch gets a gradient.
        """
        self.model.zero_grad()

        if output.ndim > 0:
            grad_output = torch.ones_like(output)
            output.backward(grad_output, retain_graph=True)
        else:
            output.backward(retain_graph=True)

    def generate_cam(self, sequences, bvp_inputs):
        """High-level API: forward -> backward -> compute Grad-CAM"""
        # 1) Forward
        output = self.forward(sequences, bvp_inputs)

        # 2) Backward
        self.backward(output)

        # 3) Create CAM
        return self._compute_cam()

    def _compute_cam(self):
        """
        Grad-CAM calculation:
          - alpha_k = GlobalAveragePool(gradients) for each channel k
          - weighted sum: sum_{k}(alpha_k * activations_k)
          - ReLU
          - Then upsample to input size if desired
        """
        if self.activations is None or self.gradients is None:
            raise ValueError("Activations or gradients not found. "
                             "Did you run forward and backward pass?")

        # activations => (B, C, D, H, W)
        # gradients   => (B, C, D, H, W)

        # 1) Compute channel-wise mean of gradients => alpha_k
        alpha = self.gradients.mean(dim=[2, 3, 4], keepdim=True)  # shape (B, C, 1, 1, 1)

        # 2) Weighted sum of activations => (B, 1, D, H, W)
        weighted_activations = alpha * self.activations

        # 3) Sum across channels => (B, 1, D, H, W)
        cam = weighted_activations.sum(dim=1, keepdim=True)

        # 4) ReLU
        cam = F.relu(cam)

        # 5) Upsample to input resolution => (B, 1, 1, 128, 128)
        cam_upsampled = F.interpolate(
            cam, size=(1, 128, 128), mode='trilinear', align_corners=False
        )

        # 6) Remove dims => (B, 128, 128)
        cam_2d = cam_upsampled.squeeze(2).squeeze(1)

        # 7) Normalize each sample to [0,1] range
        B, H, W = cam_2d.shape
        cam_2d_flat = cam_2d.view(B, -1)  # (B, H*W)
        min_vals = cam_2d_flat.min(dim=1, keepdim=True)[0]
        max_vals = cam_2d_flat.max(dim=1, keepdim=True)[0]
        cam_2d_flat_norm = (cam_2d_flat - min_vals) / (max_vals - min_vals + 1e-8)
        cam_2d = cam_2d_flat_norm.view(B, H, W)

        return cam_2d
    

