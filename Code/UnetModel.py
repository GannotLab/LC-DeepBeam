import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Attention mechanism for skip connections.
    
    Learns to gate the encoder features before concatenating them with the 
    decoder features, allowing the network to focus on relevant spatial/temporal 
    cues while suppressing background noise.
    """
    def __init__(self, d_in_channels, e_in__channels, out_channels, kernel_size=(1, 1), stride=(1, 1)):
        super().__init__()
        if out_channels == 0:
            out_channels = 1
            
        self.We = nn.Conv2d(
            in_channels=e_in__channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        
        self.Wd = nn.Conv2d(
            in_channels=d_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

        self.sigmoid = nn.Sigmoid()

        self.Watt = nn.Conv2d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, d, e):
        WeE = self.We(e)
        WdD = self.Wd(d)

        # B = sigmoid(We*e + Wd*d)
        B = self.sigmoid(WeE + WdD)    
        
        # A = sigmoid(Watt*B)
        A = self.sigmoid(self.Watt(B)) 

        # Apply attention mask to the encoder skip connection
        new_skip_e = torch.mul(A, e)

        return new_skip_e


class CausalConvBlock(nn.Module):
    """
    Encoder Block: Consists of a Conv2d layer followed by batch normalization, 
    dropout, and a LeakyReLU activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class CausalTransConvBlock(nn.Module):
    """
    Decoder Block: Consists of a ConvTranspose2d layer followed by batch normalization, 
    dropout, and a LeakyReLU activation function. Optionally includes an AttentionBlock 
    for the incoming skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

        self.att_block = AttentionBlock(in_channels, out_channels, out_channels // 2)

    def forward(self, x, skip, EnableSkipAtt):
        if EnableSkipAtt:
            skip = self.att_block(x, skip)
        
        return self.conv(torch.cat((x, skip), 1)), skip


class AttentionFusionBlock_Local(nn.Module):
    """
    Local temporal cross-attention between mixture and RTF.
    Acts as a learned subspace-tracking filter per frequency bin.

    Inputs: mix, rtf_estimate: [B, C, F, T]
    Returns: fused [B, stem_ch, F, T]
    """
    def __init__(self, rtf_in_ch, mix_in_ch, stem_ch=8, num_heads=1, attn_win=48, stride=32):
        super().__init__()
        self.attn_win = attn_win
        self.stride = stride

        # --- Light feature stems ---
        self.rtf_stem = nn.Sequential(
            nn.Conv2d(rtf_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.mix_stem = nn.Sequential(
            nn.Conv2d(mix_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

        # --- Local cross-attention ---
        self.attn = nn.MultiheadAttention(
            embed_dim=stem_ch,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(stem_ch)

        self.post = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, mix, rtf_estimate):
        B, _, F, T = mix.shape

        # --- Stems ---
        mix_s = self.mix_stem(mix)          # [B,S,F,T]
        rtf_s = self.rtf_stem(rtf_estimate) # [B,S,F,T]

        # Flatten frequency dimension
        mix_flat = mix_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)
        rtf_flat = rtf_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)

        fused = torch.zeros_like(mix_flat, device=mix.device)

        # --- Sliding local attention ---
        for t_start in range(0, T, self.stride):
            t_end = min(t_start + self.attn_win, T)
            q = mix_flat[:, t_start:t_end, :]
            k = rtf_flat[:, t_start:t_end, :]
            v = rtf_flat[:, t_start:t_end, :]
            attn_out, _ = self.attn(q, k, v, need_weights=False)
            fused[:, t_start:t_end, :] = self.norm(q + attn_out)

        # Reshape back
        fused = fused.reshape(B, F, T, -1).permute(0, 3, 1, 2)
        return self.post(fused)


class AttentionFusionBlock(nn.Module):
    """
    Cross-attention between mixture and a single RTF.
    Inputs: mix, rtf: [B, C, F, T]
    Returns: fused [B, stem_ch, F, T]
    """
    def __init__(self, rtf_in_ch, mix_in_ch, stem_ch=8, num_heads=1):
        super().__init__()

        # --- Light feature stems ---
        self.rtf_stem = nn.Sequential(
            nn.Conv2d(rtf_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.mix_stem = nn.Sequential(
            nn.Conv2d(mix_in_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

        # --- Full temporal cross-attention (per frequency) ---
        self.attn = nn.MultiheadAttention(
            embed_dim=stem_ch,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(stem_ch)

        self.post = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, mix, rtf):
        B, _, F, T = mix.shape

        # --- Stems ---
        mix_s = self.mix_stem(mix)   # [B, S, F, T]
        rtf_s = self.rtf_stem(rtf)   # [B, S, F, T]

        # Per-frequency sequences along time: [B*F, T, S]
        mix_flat = mix_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)
        rtf_flat = rtf_s.permute(0, 2, 3, 1).reshape(B * F, T, -1)

        # Q = mixture, K/V = RTF
        attn_out, _ = self.attn(mix_flat, rtf_flat, rtf_flat)
        fused = self.norm(mix_flat + attn_out)   # Residual

        # Reshape back to [B, S, F, T]
        fused = fused.reshape(B, F, T, -1).permute(0, 3, 1, 2)
        return self.post(fused)
    

# =========================================================================
# =================== 2-SPEAKER ARCHITECTURE ==============================
# =========================================================================

class UNETDualInput_Two_Speakers(nn.Module):
    """
    Utilizes local temporal cross-attention to track spatial features over time.
    Inputs: 
        mix:       [B, C_mix, F, T]
        rtf_first: [B, C_rtf, F, T] (Target)
        rtf_right: [B, C_rtf, F, T] (Interferer)
    """
    def __init__(
        self,
        rtf_in_ch: int,
        mix_in_ch: int,
        out_channels: int,
        activation: str = 'tanh',
        EnableSkipAttention: int = 0,
        stem_each=4,
        num_heads=1,
        attn_win=24,
        stride=24,
    ):
        super().__init__()
        self.EnableSkipAttention = EnableSkipAttention
        self.out_channels = out_channels
        self.stem_each = stem_each

        self.local_attn = AttentionFusionBlock_Local(
            rtf_in_ch=rtf_in_ch, mix_in_ch=mix_in_ch, stem_ch=stem_each,
            num_heads=num_heads, attn_win=attn_win, stride=stride,
        )

        in_channel = 2 * stem_each + mix_in_ch
        self.out_ch = in_channel

        self.conv_block_1 = CausalConvBlock(in_channel, 32,  (6, 3), (2, 2))
        self.conv_block_2 = CausalConvBlock(32,         32,  (7, 4), (2, 2))
        self.conv_block_3 = CausalConvBlock(32,         64,  (7, 5), (2, 2))
        self.conv_block_4 = CausalConvBlock(64,         64,  (6, 6), (2, 2))
        self.conv_block_5 = CausalConvBlock(64,         96,  (6, 6), (2, 2))
        self.conv_block_6 = CausalConvBlock(96,         96,  (6, 6), (2, 2))
        self.conv_block_7 = CausalConvBlock(96,         128, (2, 2), (2, 2))
        self.conv_block_8 = CausalConvBlock(128,        256, (2, 2), (1, 1))

        self.tran_conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.LeakyReLU(inplace=True),
        )
        self.tran_conv_block_2 = CausalTransConvBlock(256, 128, (2, 2), (2, 2))
        self.tran_conv_block_3 = CausalTransConvBlock(128, 96,  (6, 6), (2, 2))
        self.tran_conv_block_4 = CausalTransConvBlock(96,  96,  (6, 6), (2, 2))
        self.tran_conv_block_5 = CausalTransConvBlock(96,  64,  (6, 6), (2, 2))
        self.tran_conv_block_6 = CausalTransConvBlock(64,  64,  (7, 5), (2, 2))
        self.tran_conv_block_7 = CausalTransConvBlock(64,  32,  (7, 4), (2, 2))
        self.tran_conv_block_8 = CausalTransConvBlock(32,  32,  (6, 3), (2, 2))

        if EnableSkipAttention == 0:
            self.last_conv_block = nn.Sequential(
                nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels), nn.Dropout2d(0.5), nn.LeakyReLU(inplace=True),
            )
        else:
            self.last_conv_block = CausalTransConvBlock(32, out_channels, (1, 1), (1, 1))

        if activation == 'tanh':
            self.dense = nn.Sequential(nn.Linear(514, 514), nn.Tanh())
        else:
            self.dense = nn.Sequential(nn.Linear(514, 514), nn.Sigmoid())
        
        # Learnable gain parameter to adjust output target volume
        self.gain_scale = nn.Parameter(torch.ones(1))
    
    def get_applied_gain(self):
        return self.gain_scale.item() 
    
    def forward(self, mix, rtf_first, rtf_right, DUAL_MODEL):
        if DUAL_MODEL == 0:
            rtf_first = torch.zeros_like(rtf_first)
            rtf_right = torch.zeros_like(rtf_right)

        x1 = self.local_attn(mix, rtf_first)   
        x2 = self.local_attn(mix, rtf_right)   

        x = torch.cat([x1, x2, mix], dim=1)    

        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        EnableSkipAtt = 1 if self.EnableSkipAttention else 0

        d = self.tran_conv_block_1(e8)
        d, _ = self.tran_conv_block_2(d, e7, EnableSkipAtt)
        d, _ = self.tran_conv_block_3(d, e6, EnableSkipAtt)
        d, _ = self.tran_conv_block_4(d, e5, EnableSkipAtt)
        d, _ = self.tran_conv_block_5(d, e4, EnableSkipAtt)
        d, _ = self.tran_conv_block_6(d, e3, EnableSkipAtt)
        d, _ = self.tran_conv_block_7(d, e2, EnableSkipAtt)
        d, _ = self.tran_conv_block_8(d, e1, EnableSkipAtt)

        if EnableSkipAtt == 0:
            d = self.last_conv_block(d)
            last_skip = None
        else:
            d, last_skip = self.last_conv_block(d, mix, EnableSkipAtt)

        d = d.permute(0, 1, 3, 2)   
        d = self.dense(d)
        
        # --- TRUE COMPLEX NORMALIZATION ---
        # Ensures that the spatial weights act purely as phase/directional 
        # filters without distorting the underlying signal amplitude.
        B, C, T, F_514 = d.shape
        F_257 = F_514 // 2
        
        # Reshape to explicitly isolate real and imaginary components [B, C, T, 257, 2]
        d_reshaped = d.view(B, C, T, F_257, 2)
        
        # Calculate complex magnitude: sqrt(real^2 + imag^2)
        norm = torch.sqrt(torch.sum(d_reshaped ** 2, dim=(1, 4), keepdim=True)).clamp_min(1e-8)
        
        # Normalize to unit length and restore shape
        d_reshaped = d_reshaped / norm
        d = d_reshaped.view(B, C, T, F_514)

        # Apply the learned volume gain
        d_gained = d * self.gain_scale

        d = d.permute(0, 1, 3, 2)   
        d_gained = d_gained.permute(0, 1, 3, 2)
        
        return d, d_gained, last_skip


# =========================================================================
# =================== 3-SPEAKER ARCHITECTURE ==============================
# =========================================================================

class UNETDualInput_Three_Speakers(nn.Module):
    """
    Dual-input UNet designed for a Target + 2 Interferers scenario.
    Inputs: 
        mix:         [B, C_mix, F, T]
        rtf_target:  [B, C_rtf, F, T] (Target)
        rtf_interf1: [B, C_rtf, F, T] (Interferer 1)
        rtf_interf2: [B, C_rtf, F, T] (Interferer 2)
    """
    def __init__(
        self,
        rtf_in_ch: int,
        mix_in_ch: int,
        out_channels: int,
        activation: str = 'tanh',
        EnableSkipAttention: int = 0,
        stem_each=4,
        num_heads=1,
        attn_win=24,
        stride=24,
    ):
        super().__init__()
        self.EnableSkipAttention = EnableSkipAttention
        self.out_channels = out_channels
        self.stem_each = stem_each

        self.local_attn = AttentionFusionBlock_Local(
            rtf_in_ch=rtf_in_ch, mix_in_ch=mix_in_ch, stem_ch=stem_each,
            num_heads=num_heads, attn_win=attn_win, stride=stride,
        )

        in_channel = 3 * stem_each + mix_in_ch
        self.out_ch = in_channel

        self.conv_block_1 = CausalConvBlock(in_channel, 32,  (6, 3), (2, 2))
        self.conv_block_2 = CausalConvBlock(32,         32,  (7, 4), (2, 2))
        self.conv_block_3 = CausalConvBlock(32,         64,  (7, 5), (2, 2))
        self.conv_block_4 = CausalConvBlock(64,         64,  (6, 6), (2, 2))
        self.conv_block_5 = CausalConvBlock(64,         96,  (6, 6), (2, 2))
        self.conv_block_6 = CausalConvBlock(96,         96,  (6, 6), (2, 2))
        self.conv_block_7 = CausalConvBlock(96,         128, (2, 2), (2, 2))
        self.conv_block_8 = CausalConvBlock(128,        256, (2, 2), (1, 1))

        self.tran_conv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.LeakyReLU(inplace=True),
        )
        self.tran_conv_block_2 = CausalTransConvBlock(256, 128, (2, 2), (2, 2))
        self.tran_conv_block_3 = CausalTransConvBlock(128, 96,  (6, 6), (2, 2))
        self.tran_conv_block_4 = CausalTransConvBlock(96,  96,  (6, 6), (2, 2))
        self.tran_conv_block_5 = CausalTransConvBlock(96,  64,  (6, 6), (2, 2))
        self.tran_conv_block_6 = CausalTransConvBlock(64,  64,  (7, 5), (2, 2))
        self.tran_conv_block_7 = CausalTransConvBlock(64,  32,  (7, 4), (2, 2))
        self.tran_conv_block_8 = CausalTransConvBlock(32,  32,  (6, 3), (2, 2))

        if EnableSkipAttention == 0:
            self.last_conv_block = nn.Sequential(
                nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels), nn.Dropout2d(0.5), nn.LeakyReLU(inplace=True),
            )
        else:
            self.last_conv_block = CausalTransConvBlock(32, out_channels, (1, 1), (1, 1))

        self.dense = nn.Linear(514, 514)
        
        # Learnable gain parameter to adjust output target volume
        self.gain_scale = nn.Parameter(torch.ones(1))

    def get_applied_gain(self):
        return self.gain_scale.item()
    
    def forward(self, mix, rtf_target, rtf_interf1, rtf_interf2, DUAL_MODEL):
        if DUAL_MODEL == 0:
            rtf_target = torch.zeros_like(rtf_target)
            rtf_interf1 = torch.zeros_like(rtf_interf1)
            rtf_interf2 = torch.zeros_like(rtf_interf2)

        x1 = self.local_attn(mix, rtf_target)
        x2 = self.local_attn(mix, rtf_interf1)
        x3 = self.local_attn(mix, rtf_interf2)

        x = torch.cat([x1, x2, x3, mix], dim=1)

        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        EnableSkipAtt = 1 if self.EnableSkipAttention else 0

        d = self.tran_conv_block_1(e8)
        d, _ = self.tran_conv_block_2(d, e7, EnableSkipAtt)
        d, _ = self.tran_conv_block_3(d, e6, EnableSkipAtt)
        d, _ = self.tran_conv_block_4(d, e5, EnableSkipAtt)
        d, _ = self.tran_conv_block_5(d, e4, EnableSkipAtt)
        d, _ = self.tran_conv_block_6(d, e3, EnableSkipAtt)
        d, _ = self.tran_conv_block_7(d, e2, EnableSkipAtt)
        d, _ = self.tran_conv_block_8(d, e1, EnableSkipAtt)

        if EnableSkipAtt == 0:
            d = self.last_conv_block(d)
            last_skip = None
        else:
            d, last_skip = self.last_conv_block(d, mix, EnableSkipAtt)

        d = d.permute(0, 1, 3, 2)   
        d = self.dense(d)
        
        # --- TRUE COMPLEX NORMALIZATION ---
        B, C, T, F_514 = d.shape
        F_257 = F_514 // 2
        d_reshaped = d.view(B, C, T, F_257, 2)
        norm = torch.sqrt(torch.sum(d_reshaped ** 2, dim=(1, 4), keepdim=True)).clamp_min(1e-8)
        d_reshaped = d_reshaped / norm
        d = d_reshaped.view(B, C, T, F_514)

        d_gained = d * self.gain_scale

        d = d.permute(0, 1, 3, 2)   
        d_gained = d_gained.permute(0, 1, 3, 2)
        
        return d, d_gained, last_skip