import torch
import numpy as np

class CameraGuider(torch.nn.Module):
    def __init__(self, out_dim, concat_dim=4):
        super(CameraGuider, self).__init__()
        self.layer = torch.nn.Sequential(
            # Initial conv block (no downsampling)
            torch.nn.Conv2d(6, concat_dim * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            # Spatial downsampling x4 (total factor 16)
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(concat_dim * 4, concat_dim * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            # Final conv to project to out_dim and downsample one more time
            torch.nn.Conv2d(concat_dim * 4, out_dim, kernel_size=2, stride=2, padding=0)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layer:
            if isinstance(m, torch.nn.Conv2d):
                # Kaiming normal (He) initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                torch.nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,6,H,W)
        # out: (B,out_dim,H/16,W/16)
        return self.layer(x)