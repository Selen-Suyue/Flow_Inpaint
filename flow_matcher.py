# flow_matcher.py
import torch
import torch.nn.functional as F
from dataset import generate_random_mask

class FlowMatcher:
    def __init__(self, net):
        self.net = net # The U-Net model

    def get_train_tuple(self, x_1, mask_type="random_rect"):
        """
        Prepares data for a single training step.
        x_1: clean images (B, C, H, W), range [-1, 1]
        """
        B, C, H, W = x_1.shape
        device = x_1.device

        # 1. Generate mask m (1 for known, 0 for unknown)
        if mask_type == "random_rect":
            # In training, mask should be different for each image in batch
            m = generate_random_mask(x_1.shape).to(device)
        else: # Add other mask generation types if needed
            raise NotImplementedError(f"Mask type {mask_type} not implemented.")

        x_known = x_1 * m

        # 2. Sample x_0_conditional: known parts + noise in masked parts
        z_masked = torch.randn_like(x_1) # Noise for masked regions
        x_0_conditional = x_known + (1 - m) * z_masked

        # 3. Sample t uniformly from [0, 1]
        t = torch.rand(B, device=device) * (1.0 - 1e-5) + 1e-5 # Shape (B,)

        # 4. Form x_t using linear interpolation
        t_broadcast = t.view(B, *((1,) * (x_1.dim() - 1)))

        x_t = (1 - t_broadcast) * x_0_conditional + t_broadcast * x_1

        # 5. Target vector field u_t = x_1 - x_0_conditional
        u_t_target = x_1 - x_0_conditional

        return x_t, t, m, x_known, u_t_target

    def loss_fn(self, predicted_vt, target_ut, mask_unknown):
        """
        Calculates MSE loss only on the unknown (masked) regions.
        predicted_vt: model output v_theta(x_t, t, m, x_known)
        target_ut: x_1 - x_0_conditional
        mask_unknown: (1-m), so it's 1 for unknown regions, 0 for known
        """
        loss = F.mse_loss(predicted_vt * mask_unknown, target_ut * mask_unknown, reduction='none')
        return loss.sum(dim=[1,2,3]).mean()


    @torch.no_grad()
    def sample_ode_inpainting(self, masked_img, mask, num_steps=100):
        """
        Perform inpainting using ODE sampling.
        masked_img: (B, C, H, W) image with masked areas (e.g., set to 0 or noise)
        mask: (B, 1, H, W) binary mask (1 for known, 0 for unknown)
        num_steps: Number of steps for Euler solver
        """
        self.net.eval()
        device = masked_img.device
        B, C, H, W = masked_img.shape

        x_known = masked_img * mask

        # Initialize x_0: known parts + noise in masked parts
        x_t = x_known + (1 - mask) * torch.randn_like(masked_img)

        dt = 1.0 / num_steps
        ts = torch.linspace(1e-5, 1.0, num_steps, device=device) # time from ~0 to 1

        for t_val_current in ts:
            t_batch = torch.ones(B, device=device) * t_val_current

            v_pred = self.net(x_t, t_batch, mask, x_known)

            x_t = x_t + v_pred * dt

            x_t = x_t * (1 - mask) + x_known

        return torch.clamp(x_t, -1.0, 1.0)