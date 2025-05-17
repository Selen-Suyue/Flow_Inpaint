# inpaint_gui.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog

from unet import ConditionalUNet # Make sure this is the same as used in training
from flow_matcher import FlowMatcher # For sample_ode_inpainting

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/inpainting_fm_cifar10.pth" # Path to your trained model
IMG_SIZE = 32 # CIFAR-10 size
IMG_CHANNELS = 3
TIME_EMB_DIM = 64 # Must match training
UNET_BASE_DIM = 64 # Must match training
NUM_INPAINT_STEPS = 100 # Number of ODE steps for inpainting

# --- Global variables for GUI state ---
current_image_pil = None
current_image_tensor = None # Normalized tensor [-1, 1]
mask_np = None # NumPy array for mask (0 for masked, 1 for known)
fig_gui, ax_gui = None, None
drawing = False
brush_size = 2 # Radius of the brush

# --- Model Loading ---
def load_model():
    model = ConditionalUNet(
        img_channels=IMG_CHANNELS,
        time_emb_dim=TIME_EMB_DIM,
        base_dim=UNET_BASE_DIM
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please train the model first.")
        return None
    except Exception as e:
        print(f"ERROR: Could not load model. Error: {e}")
        return None
    return FlowMatcher(net=model)

flow_module_global = load_model()

# --- Image Transformations ---
# For converting PIL Image to tensor for model input
to_tensor_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), # Scales to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales to [-1, 1]
])

# For displaying tensor ([-1, 1]) as PIL Image ([0, 255])
def tensor_to_pil(tensor_img):
    tensor_img = (tensor_img + 1) / 2 # Denormalize to [0, 1]
    tensor_img = tensor_img.squeeze(0).cpu().clamp(0,1)
    print(tensor_img.shape)
    pil_img = transforms.ToPILImage()(tensor_img)
    return pil_img

# --- GUI Event Handlers ---
def on_motion(event):
    global drawing, mask_np, ax_gui, current_image_pil
    if not drawing or event.inaxes != ax_gui:
        return
    if current_image_pil is None:
        return

    x, y = int(event.xdata), int(event.ydata)
    
    # Apply brush
    y_min = max(0, y - brush_size)
    y_max = min(IMG_SIZE, y + brush_size + 1)
    x_min = max(0, x - brush_size)
    x_max = min(IMG_SIZE, x + brush_size + 1)
    
    if mask_np is not None:
        mask_np[y_min:y_max, x_min:x_max] = 0 # 0 for masked region

    update_display()

def on_press(event):
    global drawing
    if event.inaxes != ax_gui or current_image_pil is None:
        return
    drawing = True
    on_motion(event) # Draw a point on click

def on_release(event):
    global drawing
    drawing = False

def update_display():
    global current_image_pil, mask_np, ax_gui
    if current_image_pil is None or mask_np is None or ax_gui is None:
        return

    display_img_np = np.array(current_image_pil).astype(float) / 255.0
    # Overlay mask (make masked areas gray, for example)
    # mask_np is (H, W), needs to be (H, W, 1) for broadcasting
    mask_for_display = mask_np[:, :, np.newaxis]
    
    # Where mask is 0 (masked), make it gray. Where 1, keep original.
    # display_img_np = display_img_np * mask_for_display + 0.5 * (1 - mask_for_display)
    
    # Better: show original with semi-transparent red overlay for mask
    overlay = display_img_np
    overlay[mask_np == 0] = [1, 0, 0] # Red for masked areas
    
    ax_gui.clear()
    ax_gui.imshow(display_img_np)
    alpha_values = 0.3 * (1 - mask_np) # mask_np 是 (H,W), 1-mask_np 也是 (H,W)
    ax_gui.imshow(overlay, alpha=alpha_values)
    ax_gui.set_title("Draw mask (click & drag). Then press 'Inpaint'.")
    ax_gui.axis('off')
    plt.draw()


def load_image_action(event):
    global current_image_pil, current_image_tensor, mask_np, ax_gui
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
    )
    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB")
        current_image_pil = img.resize((IMG_SIZE, IMG_SIZE)) # Resize to model input size
        
        # Image tensor for model: (1, C, H, W) in range [-1, 1]
        current_image_tensor = to_tensor_transform(current_image_pil).unsqueeze(0).to(DEVICE)
        
        
        # Initialize mask: (H, W), all 1s (nothing masked yet)
        mask_np = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        update_display()
    except Exception as e:
        print(f"Error loading image: {e}")
        current_image_pil = None
        current_image_tensor = None
        mask_np = None


def inpaint_action(event):
    global current_image_tensor, mask_np, flow_module_global, ax_gui
    if current_image_tensor is None or mask_np is None or flow_module_global is None:
        print("Please load an image and define a mask first.")
        return
    if np.all(mask_np == 1): # Check if any part of the mask is drawn
        print("No mask drawn. Please select regions to inpaint.")
        return

    print("Inpainting...")
    # Prepare mask for model: (1, 1, H, W) tensor, 1 for known, 0 for unknown
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # Input to inpainting: image with masked areas potentially zeroed or noised
    # For simplicity, we'll use original values in known parts, and can put zeros or noise
    # in masked parts. The sample_ode_inpainting function will handle adding noise.
    masked_input_img_tensor = current_image_tensor * mask_tensor 
                                # This sets masked regions to 0, which is fine.
                                # sample_ode_inpainting will fill (1-mask) with noise.

    inpainted_tensor = flow_module_global.sample_ode_inpainting(
        masked_input_img_tensor,
        mask_tensor,
        num_steps=NUM_INPAINT_STEPS
    )
    
    inpainted_pil = tensor_to_pil(inpainted_tensor)

    # Update display with inpainted image
    ax_gui.clear()
    ax_gui.imshow(inpainted_pil)
    ax_gui.set_title("Inpainted Result")
    ax_gui.axis('off')
    plt.draw()
    print("Inpainting complete.")

    # Optionally, reset mask for further editing or save result
    # mask_np = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) # Reset mask


def reset_mask_action(event):
    global mask_np
    if current_image_pil is not None:
        mask_np = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        update_display()
    else:
        print("Load an image first to reset its mask.")


def setup_gui():
    global fig_gui, ax_gui
    fig_gui, ax_gui = plt.subplots(figsize=(6,7)) # Make space for buttons
    plt.subplots_adjust(bottom=0.25) # Adjust layout to make space for buttons

    ax_gui.set_title("Load an image, then draw mask")
    ax_gui.axis('off')

    # Connect mouse events
    fig_gui.canvas.mpl_connect('motion_notify_event', on_motion)
    fig_gui.canvas.mpl_connect('button_press_event', on_press)
    fig_gui.canvas.mpl_connect('button_release_event', on_release)

    # Add Buttons
    ax_load = plt.axes([0.1, 0.05, 0.25, 0.075]) # [left, bottom, width, height]
    btn_load = Button(ax_load, 'Load Image')
    btn_load.on_clicked(load_image_action)

    ax_inpaint = plt.axes([0.38, 0.05, 0.25, 0.075])
    btn_inpaint = Button(ax_inpaint, 'Inpaint')
    btn_inpaint.on_clicked(inpaint_action)

    ax_reset = plt.axes([0.66, 0.05, 0.25, 0.075])
    btn_reset = Button(ax_reset, 'Reset Mask')
    btn_reset.on_clicked(reset_mask_action)
    
    plt.show()

if __name__ == "__main__":
    if flow_module_global is None:
        print("Exiting: Model could not be loaded.")
    else:
        setup_gui()