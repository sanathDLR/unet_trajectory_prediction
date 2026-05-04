import torch
import torch.nn as nn


# =================================
# BASIC BLOCK
# =================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# =================================
# CONVLSTM CELL
# =================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# =================================
# CONVLSTM ENCODER
# =================================
class ConvLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=48):  # 🔥 UPDATED
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h  # (B, hidden_dim, H, W)


# =================================
# CONVLSTM + UNET (1-step output)
# =================================
class ConvLSTM_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Temporal encoder ----
        self.temporal = ConvLSTM(input_dim=2, hidden_dim=48)  # 🔥 FIXED

        # ---- Static encoder ----
        self.static_enc = DoubleConv(2, 48)

        # ---- UNet encoder ----
        self.enc1 = DoubleConv(96, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(128, 256)

        # ---- Decoder ----
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ---- Output (ONLY 1 STEP) ----
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, dyn_seq, static):
        """
        dyn_seq: (B, T, 2, H, W)
        static:  (B, 2, H, W)
        """

        # Temporal features
        temporal_feat = self.temporal(dyn_seq)   # (B, 48, H, W)

        # Static features
        static_feat = self.static_enc(static)    # (B, 48, H, W)

        # Fusion
        x = torch.cat([temporal_feat, static_feat], dim=1)  # (B, 96, H, W)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)  # (B,1,H,W)

        return out