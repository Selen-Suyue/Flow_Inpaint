# train.py
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from unet import ConditionalUNet
from dataset import get_cifar10_dataloader,generate_random_mask
from flow_matcher import FlowMatcher

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_CHANNELS = 3
IMG_SIZE = 32
TIME_EMB_DIM = 64 
UNET_BASE_DIM = 64

LEARNING_RATE = 1e-4 
BATCH_SIZE = 256 
EPOCHS = 100 

MODEL_SAVE_PATH = "saved_models/inpainting_fm_cifar10.pth"
RESULTS_SAVE_PATH = "results/"
os.makedirs("saved_models", exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

def save_sample_images(epoch, flow_module, test_batch, mask, num_images=4, prefix="train"):
    """Saves a grid of original, masked, and reconstructed images."""
    x1_clean = test_batch[:num_images].to(DEVICE)
    test_mask = mask[:num_images].to(DEVICE)

    masked_input_for_sampling = x1_clean * test_mask # Known parts, 0 in unknown
    
    reconstructed_x1 = flow_module.sample_ode_inpainting(
        masked_input_for_sampling,
        test_mask,
        num_steps=50 # Fewer steps for quick eval during training
    )

    # Denormalize from [-1, 1] to [0, 1] for visualization
    x1_clean = (x1_clean + 1) / 2
    masked_display = (x1_clean * test_mask) + (0.5 * (1-test_mask)) # Show masked as gray
    reconstructed_x1 = (reconstructed_x1 + 1) / 2

    # Ensure they are on CPU for plotting
    x1_clean = x1_clean.cpu().permute(0, 2, 3, 1).numpy()
    masked_display = masked_display.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_x1 = reconstructed_x1.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(num_images, 3, figsize=(8, num_images * 2.5))
    for i in range(num_images):
        axes[i, 0].imshow(x1_clean[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masked_display[i])
        axes[i, 1].set_title("Masked")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(reconstructed_x1[i])
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_SAVE_PATH, f"{prefix}_epoch_{epoch+1}.png"))
    plt.close(fig)


def main():
    # --- Setup ---
    model = ConditionalUNet(
        img_channels=IMG_CHANNELS,
        time_emb_dim=TIME_EMB_DIM,
        base_dim=UNET_BASE_DIM
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loader, test_loader = get_cifar10_dataloader(batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    
    flow_module = FlowMatcher(net=model)

    # Get a fixed batch from test_loader for consistent visualization
    fixed_test_images, _ = next(iter(test_loader))
    fixed_test_images = fixed_test_images.to(DEVICE)
    # Generate a fixed mask for these test images for consistent visualization
    # This mask will be used to show how the model inpaints a specific masked version
    fixed_eval_mask = generate_random_mask(fixed_test_images.shape).to(DEVICE)


    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0

        for batch_idx, (x1_clean, _) in enumerate(progress_bar):
            x1_clean = x1_clean.to(DEVICE) # These are target images x_1
            optimizer.zero_grad()

            # Get data for flow matching step
            # x_t: current image state, t: time, m: mask (1=known),
            # x_known: known pixels, u_t_target: target velocity
            x_t, t, m, x_known, u_t_target = flow_module.get_train_tuple(x1_clean)
            
            # Model prediction for velocity v_theta(x_t, t, m, x_known)
            predicted_vt = model(x_t, t, m, x_known)

            # Loss calculation (only on unknown regions: 1-m)
            mask_unknown = 1 - m
            loss = flow_module.loss_fn(predicted_vt, u_t_target, mask_unknown)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

        # --- Evaluation and Saving ---
        if (epoch + 1) % 5 == 0: # Save model and samples every 5 epochs
            model.eval()
            save_sample_images(epoch, flow_module, fixed_test_images, fixed_eval_mask, num_images=4, prefix="eval")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save final model

if __name__ == "__main__":
    main()