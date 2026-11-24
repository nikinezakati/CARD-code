import torch


class WhitenedH_deno:
    def __init__(self, H_orig, Sigma_inv_sqrt, channels, height, width,
                 patch_size=8, stride=8, device=None, dtype=torch.float32):
        assert stride == patch_size
        self.Hf = H_orig
        self.C, self.height, self.width = channels, height, width
        self.p = patch_size
        self.device = device
        self.dtype = dtype

        self.W_patch = Sigma_inv_sqrt.to(device=device, dtype=torch.float64)
        self.Wp_patch = torch.linalg.pinv(self.W_patch)

        self.Uw, self.sw, self.VwT = torch.linalg.svd(self.W_patch, full_matrices=False)
        self.Vw = self.VwT.transpose(-2, -1).contiguous()

        self.unfold = torch.nn.Unfold(kernel_size=self.p, stride=self.p)
        self.fold   = torch.nn.Fold(output_size=(self.height, self.width),
                                    kernel_size=self.p, stride=self.p)

        with torch.no_grad():
            ones = torch.ones(1, self.C, self.height, self.width,
                              device=device, dtype=self.dtype)
            self.weights = torch.clamp(self.fold(self.unfold(ones)), min=1e-8)

    def _img_like(self, x, B):
        return x.view(B, self.C, self.height, self.width)

    def _flatten(self, img):
        return img.view(img.shape[0], -1)

    @torch.no_grad()
    def _apply_patches_mat(self, img, M):
        B = img.shape[0]
        patches = self.unfold(img)
        BC, N = B * self.C, patches.shape[-1]
        patches = patches.view(BC, self.p * self.p, N)
        patches = torch.matmul(M.to(patches), patches)
        patches = patches.view(B, self.C * self.p * self.p, N)
        out = self.fold(patches) / self.weights
        return out.to(self.dtype)

    @torch.no_grad()
    def H(self, x):
        B = x.shape[0]
        x_img = self._img_like(x, B) if x.ndim == 2 else x
        y_img = self._apply_patches_mat(x_img, self.W_patch)
        return self._flatten(y_img)

    @torch.no_grad()
    def H_pinv(self, y_flat):
        B = y_flat.shape[0]
        y_img = self._img_like(y_flat, B)
        x_img = self._apply_patches_mat(y_img, self.Wp_patch)
        return self._flatten(x_img)

    @torch.no_grad()
    def Ut(self, y_flat):
        B = y_flat.shape[0]
        y_img = self._img_like(y_flat, B)
        patches = self.unfold(y_img)                          # (B, C*p*p, N)
        BC, N = B * self.C, patches.shape[-1]
        patches = patches.view(BC, self.p * self.p, N)        # (BC, p^2, N)
        patches = torch.matmul(self.Uw.transpose(-2, -1).to(patches), patches)
        patches = patches.view(B, self.C * self.p * self.p, N)
        out_img = self.fold(patches) / self.weights
        return self._flatten(out_img)

    @torch.no_grad()
    def Vt(self, x_flat):
        B = x_flat.shape[0]
        x_img = self._img_like(x_flat, B)
        patches = self.unfold(x_img)
        BC, N = B * self.C, patches.shape[-1]
        patches = patches.view(BC, self.p * self.p, N)
        patches = torch.matmul(self.Vw.transpose(-2, -1).to(patches), patches)
        patches = patches.view(B, self.C * self.p * self.p, N)
        out_img = self.fold(patches) / self.weights
        return self._flatten(out_img)

    @torch.no_grad()
    def V(self, x_flat):
        B = x_flat.shape[0]
        x_img = self._img_like(x_flat, B)
        patches = self.unfold(x_img)
        BC, N = B * self.C, patches.shape[-1]
        patches = patches.view(BC, self.p * self.p, N)
        patches = torch.matmul(self.Vw.to(patches), patches)
        patches = patches.view(B, self.C * self.p * self.p, N)
        out_img = self.fold(patches) / self.weights
        return self._flatten(out_img)

    @torch.no_grad()
    def singulars(self):
        tile = self.sw.to(self.dtype).view(self.p, self.p)              # (p,p)
        s_img_1c = tile.repeat(self.height // self.p, self.width // self.p)  # (H,W)
        s_img = s_img_1c.unsqueeze(0).repeat(self.C, 1, 1)              # (C,H,W)
        return s_img.reshape(-1)
