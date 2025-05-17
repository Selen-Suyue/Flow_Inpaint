# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t): # t 的 shape 是 (B,)
        device = t.device 
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t_emb))
        time_emb = time_emb[(..., ) + (None, ) * 2] 
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

# ... (SinusoidalPosEmb and Block class remain the same) ...

class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=32, base_dim=64):
        super().__init__()
        input_channels = img_channels + 1 + img_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Downsampling
        self.inc = nn.Conv2d(input_channels, base_dim, 3, padding=1) # 32x32
        self.down1 = Block(base_dim, base_dim*2, time_emb_dim)      # 16x16
        self.down2 = Block(base_dim*2, base_dim*4, time_emb_dim)    # 8x8
        self.sa1 = nn.Identity()

        # Bottleneck
        self.bot1 = Block(base_dim*4, base_dim*8, time_emb_dim, up=False) # 4x4

        # Output of bot2.transform should be base_dim*4 channels to match x3.
        self.bot2 = Block(base_dim*4, base_dim*4, time_emb_dim, up=True)  # Output: base_dim*4 channels, spatially 8x8

        self.up1 = Block(base_dim*4, base_dim*2, time_emb_dim, up=True) # Output: base_dim*2 channels, spatially 16x16
        self.sa2 = nn.Identity()

        self.up2 = Block(base_dim*2, base_dim, time_emb_dim, up=True)   # Output: base_dim channels, spatially 32x32
        self.outc = nn.Conv2d(base_dim*2, img_channels, 1)

    def forward(self, x_t, t, mask, known_pixels):
        t_emb = self.time_mlp(t)
        x_cond = torch.cat([x_t, mask, known_pixels], dim=1)

        x1 = self.inc(x_cond)    # (B, base_dim, 32, 32)
        x2 = self.down1(x1, t_emb) # (B, base_dim*2, 16, 16)
        x2_sa = self.sa1(x2)
        x3 = self.down2(x2_sa, t_emb) # (B, base_dim*4, 8, 8)

        x_bot_down = self.bot1(x3, t_emb) # (B, base_dim*8, 4, 4) - Output of bot1.transform
        
        x_bot_up = self.bot2(x_bot_down, t_emb) # (B, base_dim*4, 8, 8)

        up1_out = self.up1(torch.cat([x_bot_up, x3], dim=1), t_emb) # (B, base_dim*2, 16, 16)
        up1_out_sa = self.sa2(up1_out)

        up2_out = self.up2(torch.cat([up1_out_sa, x2_sa], dim=1), t_emb) # (B, base_dim, 32, 32)

        output = self.outc(torch.cat([up2_out, x1], dim=1)) # (B, img_channels, 32, 32)
        return output