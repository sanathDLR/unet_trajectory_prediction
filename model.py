import torch
import torch.nn as nn


# ==========================================
# RESIDUAL BLOCK
# ==========================================

class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch)
        )

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


# ==========================================
# CONVLSTM CELL
# ==========================================

class ConvLSTMCell(nn.Module):

    def __init__(self, in_ch, hidden_ch):
        super().__init__()

        self.hidden_ch = hidden_ch

        self.conv = nn.Conv2d(
            in_ch + hidden_ch,
            4 * hidden_ch,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, h, c):

        gates = self.conv(
            torch.cat([x, h], dim=1)
        )

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


# ==========================================
# CONVLSTM
# ==========================================

class ConvLSTM(nn.Module):

    def __init__(self, in_ch=1, hidden_ch=64):
        super().__init__()

        self.hidden_ch = hidden_ch
        self.cell = ConvLSTMCell(in_ch, hidden_ch)

    def forward(self, x):

        B, T, C, H, W = x.shape

        h = torch.zeros(
            B,
            self.hidden_ch,
            H,
            W,
            device=x.device
        )

        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h


# ==========================================
# MODEL
# ==========================================

class ConvLSTM_UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------------------------------
        # Temporal encoder
        # ---------------------------------

        self.temporal = ConvLSTM(
            in_ch=1,
            hidden_ch=64
        )

        # ---------------------------------
        # Static encoder
        # ---------------------------------

        self.static_enc = ResBlock(
            2,
            64
        )

        # ---------------------------------
        # Fusion
        # ---------------------------------

        self.fuse = ResBlock(
            128,
            128
        )

        # ---------------------------------
        # Encoder
        # ---------------------------------

        self.enc1 = ResBlock(
            128,
            64
        )

        self.enc2 = ResBlock(
            64,
            128
        )

        self.enc3 = ResBlock(
            128,
            256
        )

        self.pool = nn.MaxPool2d(2)

        # ---------------------------------
        # Bottleneck
        # ---------------------------------

        self.bottleneck = ResBlock(
            256,
            256
        )

        self.dropout = nn.Dropout2d(0.1)

        # ---------------------------------
        # Decoder
        # ---------------------------------

        self.up3 = nn.ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ResBlock(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ResBlock(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ResBlock(
            128,
            64
        )

        # ---------------------------------
        # Final refinement
        # ---------------------------------

        self.refine = ResBlock(
            64,
            64
        )

        # ---------------------------------
        # Output
        # ---------------------------------

        self.head = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, dyn_seq, static):

        # -------------------------
        # Temporal
        # -------------------------

        temporal_feat = self.temporal(
            dyn_seq
        )

        # -------------------------
        # Static
        # -------------------------

        static_feat = self.static_enc(
            static
        )

        # -------------------------
        # Fusion
        # -------------------------

        x = self.fuse(
            torch.cat(
                [temporal_feat, static_feat],
                dim=1
            )
        )

        # -------------------------
        # Encoder
        # -------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(
            self.pool(e3)
        )

        b = self.dropout(b)

        # -------------------------
        # Decoder
        # -------------------------

        d3 = self.dec3(
            torch.cat(
                [self.up3(b), e3],
                dim=1
            )
        )

        d2 = self.dec2(
            torch.cat(
                [self.up2(d3), e2],
                dim=1
            )
        )

        d1 = self.dec1(
            torch.cat(
                [self.up1(d2), e1],
                dim=1
            )
        )

        # -------------------------
        # Refinement
        # -------------------------

        d1 = self.refine(d1)

        # -------------------------
        # Output
        # -------------------------

        return self.head(d1)