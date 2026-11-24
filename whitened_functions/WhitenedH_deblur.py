import torch


class WhitenedH_deblur:
    def __init__(self, orig_H, Sigma_inv_sqrt,
                 channels, height, width,
                 patch_size=8, stride=8,
                 device=None,
                 dtype=torch.float32,
                 matrix_dtype=torch.float32,
                 eps: float = 1e-6,
                 mc_samples: int = 64,
                 mc_batch: int = 16):
        assert stride == patch_size, "Use non-overlapping whitening patches for deblurring."
        self.Hf = orig_H                        
        self.C, self.H, self.W = channels, height, width
        self.p = patch_size
        self.device = device
        self.dtype = dtype
        self.matrix_dtype = matrix_dtype
        self.eps = eps

        self.unfold = torch.nn.Unfold(kernel_size=self.p, stride=self.p)
        self.fold   = torch.nn.Fold(output_size=(self.H, self.W),
                                    kernel_size=self.p, stride=self.p)
        with torch.no_grad():
            ones = torch.ones(1, self.C, self.H, self.W, device=self.device, dtype=self.dtype)
            self.weights = torch.clamp(self.fold(self.unfold(ones)), min=1e-8)

        p2 = self.p * self.p
        W_in = Sigma_inv_sqrt.to(device=self.device, dtype=self.matrix_dtype)
        assert W_in.shape == (p2, p2), f"Expected ({p2}x{p2}) whitener for deblurring, got {W_in.shape}."
        self.W_patch  = W_in
        self.Wp_patch = torch.linalg.pinv(self.W_patch)

        self.s_orig = self.Hf.singulars().to(self.dtype).to(self.device)   # shape: C*H*W
        self.gain = self._estimate_noise_gain(n_samples=mc_samples, batch=mc_batch)  # shape: C*H*W
        self.s_eff = self.s_orig / (self.gain + self.eps)

    def _flatten(self, img):             # (B,C,H,W) -> (B, C*H*W)
        return img.view(img.shape[0], -1)

    def _img_like(self, vec, B):         # (B, C*H*W) -> (B,C,H,W)
        return vec.view(B, self.C, self.H, self.W)

    @torch.no_grad()
    def _apply_patch_mat(self, img, M):  
        B = img.shape[0]
        patches = self.unfold(img)                           # (B, C*p2, Np)
        BC, Np = B * self.C, patches.shape[-1]
        p2 = patches.shape[1] // self.C
        patches = patches.reshape(BC, p2, Np)                # (BC, p2, Np)
        patches = torch.einsum('pq,bqn->bpn', M.to(patches), patches)
        patches = patches.reshape(B, self.C * p2, Np)
        out = self.fold(patches) / self.weights
        return out.to(self.dtype)

    @torch.no_grad()
    def whiten(self, y_img):             # (B,C,H,W) -> (B,C,H,W)
        return self._apply_patch_mat(y_img.to(self.dtype), self.W_patch)

    @torch.no_grad()
    def unwhiten(self, y_img):           # (B,C,H,W) -> (B,C,H,W)
        return self._apply_patch_mat(y_img.to(self.dtype), self.Wp_patch)

    @torch.no_grad()
    def _estimate_noise_gain(self, n_samples=64, batch=16):
        n_left = n_samples
        m2 = torch.zeros_like(self.s_orig, dtype=self.dtype, device=self.device) 
        while n_left > 0:
            b = min(batch, n_left)
            n_left -= b
            eps_w = torch.randn(b, self.C, self.H, self.W, device=self.device, dtype=self.dtype)
            eps = self.unwhiten(eps_w)
            u0  = self.Hf.Ut(self._flatten(eps))               # (b, C*H*W)
            m2 += (u0 * u0).sum(dim=0)
        var = m2 / max(1, n_samples)
        gain = torch.sqrt(var + self.eps)
        return gain

    @torch.no_grad()
    def H(self, inp):
        if inp.ndim == 2:
            x_img = self._img_like(inp, inp.shape[0])
        elif inp.ndim == 4:
            x_img = inp
        else:
            raise RuntimeError(f"H(): unexpected ndim {inp.ndim}")
        y_unw_flat = self.Hf.H(self._flatten(x_img))         
        y_unw = self._img_like(y_unw_flat, x_img.shape[0])
        y_w   = self.whiten(y_unw)
        return self._flatten(y_w)

    @torch.no_grad()
    def H_pinv(self, y_flat):
        y_img = self._img_like(y_flat, y_flat.shape[0])
        y_unw = self.unwhiten(y_img)
        return self.Hf.H_pinv(self._flatten(y_unw))

    @torch.no_grad()
    def Ut(self, y_flat):
        y_img = self._img_like(y_flat, y_flat.shape[0])
        y_unw = self.unwhiten(y_img)
        u0    = self.Hf.Ut(self._flatten(y_unw))             # (B, C*H*W)
        return u0 / (self.gain + self.eps)

    @torch.no_grad()
    def U(self, z_small):
        z_scaled = z_small * (self.gain + self.eps)
        y_unw_flat = self.Hf.U(z_scaled)
        y_unw = self._img_like(y_unw_flat, z_small.shape[0])
        y_w   = self.whiten(y_unw)
        return self._flatten(y_w)

    @torch.no_grad()
    def Vt(self, x_flat):
        return self.Hf.Vt(x_flat)

    @torch.no_grad()
    def V(self, z_full):
        return self.Hf.V(z_full)

    @torch.no_grad()
    def add_zeros(self, z_small):
        return self.Hf.add_zeros(z_small)

    @torch.no_grad()
    def singulars(self):
        return self.s_eff

