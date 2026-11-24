import torch


class WhitenedH_sr:
    def __init__(self, orig_H, Sigma_inv_sqrt,
                 channels, height, width,
                 patch_size=8, stride=8,
                 device=None,
                 dtype=torch.float32,
                 matrix_dtype=torch.float32,
                 eps: float = 1e-6):
        assert stride == patch_size, "Non-overlapping patches assumed"
        assert hasattr(orig_H, "ratio"), "SR-only: orig_H must have .ratio"
        self.Hf = orig_H
        self.C, self.height, self.width = channels, height, width
        self.p = patch_size
        self.device = device
        self.dtype = dtype
        self.matrix_dtype = matrix_dtype
        self.eps = eps
        
        self.r = int(self.Hf.ratio)
        assert self.height % self.r == 0 and self.width % self.r == 0
        assert self.p % self.r == 0
        self.hL, self.wL = self.height // self.r, self.width // self.r
        self.p_out = self.p // self.r

        self.unfold_x = torch.nn.Unfold(kernel_size=self.p, stride=self.p)
        self.fold_x   = torch.nn.Fold(output_size=(self.height, self.width),
                                      kernel_size=self.p, stride=self.p)
        with torch.no_grad():
            ones_x = torch.ones(1, self.C, self.height, self.width,
                                device=self.device, dtype=self.dtype)
            self.weights_x = torch.clamp(self.fold_x(self.unfold_x(ones_x)), min=1e-8)

        self.unfold_y = torch.nn.Unfold(kernel_size=self.p_out, stride=self.p_out)
        self.fold_y   = torch.nn.Fold(output_size=(self.hL, self.wL),
                                      kernel_size=self.p_out, stride=self.p_out)
        with torch.no_grad():
            ones_y = torch.ones(1, self.C, self.hL, self.wL,
                                device=self.device, dtype=self.dtype)
            self.weights_y = torch.clamp(self.fold_y(self.unfold_y(ones_y)), min=1e-8)

        self.H_patch  = self._build_H_patch()
        self.Hp_patch = torch.linalg.pinv(self.H_patch)

        p2 = self.p * self.p
        q2 = self.p_out * self.p_out
        Win = Sigma_inv_sqrt.to(device=self.device, dtype=self.matrix_dtype)
        if Win.shape[0] == q2:
            self.W_patch = Win  
        elif Win.shape[0] == p2:
            W_x   = Win
            W_x_p = torch.linalg.pinv(W_x)
            Sigma_x = W_x_p @ W_x_p.transpose(-2, -1)
            Sigma_x = 0.5*(Sigma_x + Sigma_x.transpose(-2, -1))
            Sigma_y = self.H_patch @ Sigma_x @ self.H_patch.transpose(-2, -1)
            Sigma_y = Sigma_y + self.eps * torch.eye(q2, device=self.device, dtype=self.matrix_dtype)
            ew, Uw = torch.linalg.eigh(0.5*(Sigma_y+Sigma_y.transpose(-2,-1)))
            ew = torch.clamp(ew, min=self.eps)
            self.W_patch = Uw @ torch.diag_embed(ew.rsqrt()) @ Uw.transpose(-2, -1)
        else:
            raise RuntimeError(f"Whitener has wrong size: {Win.shape}")

        self.Wp_patch = torch.linalg.pinv(self.W_patch)   # LR unwhitener (q^2 × q^2)


        self.WH_patch  = self.W_patch @ self.H_patch
        self.WHp_patch = torch.linalg.pinv(self.WH_patch)

        self.Uw, self.sw, self.VwT = torch.linalg.svd(self.WH_patch, full_matrices=False)
        self.Vw = self.VwT.transpose(-2, -1).contiguous()

    # ---------- helpers ----------
    def _img_like_x(self, x, B): return x.view(B, self.C, self.height, self.width)
    def _img_like_y(self, y, B): return y.view(B, self.C, self.hL, self.wL)
    def _flatten(self, img):     return img.view(img.shape[0], -1)

    @torch.no_grad()
    def _apply_patches_X_to_Y(self, img_x, M_rect):
        B = img_x.shape[0]
        patches = self.unfold_x(img_x)                               # (B, C*p^2, N_hr)
        BC, N_hr = B * self.C, patches.shape[-1]
        patches = patches.reshape(BC, self.p * self.p, N_hr)         # (BC, p^2, N_hr)
        patches = torch.einsum('qp,bpn->bqn', M_rect.to(patches), patches)  # (BC, q^2, N_hr)
        patches = patches.reshape(B, self.C * (self.p_out ** 2), N_hr)
        y_img = self.fold_y(patches) / self.weights_y                # (B, C, hL, wL)
        return y_img.to(self.dtype)

    @torch.no_grad()
    def _apply_patches_Y_to_Y(self, img_y, M):
        B = img_y.shape[0]
        patches = self.unfold_y(img_y)                               # (B, C*q^2, N_lr)
        BC, N_lr = B * self.C, patches.shape[-1]
        patches = patches.reshape(BC, self.p_out * self.p_out, N_lr) # (BC, q^2, N_lr)
        patches = torch.einsum('pq,bqn->bpn', M.to(patches), patches)# (BC, q^2, N_lr)
        patches = patches.reshape(B, self.C * (self.p_out ** 2), N_lr)
        y_img = self.fold_y(patches) / self.weights_y
        return y_img.to(self.dtype)

    @torch.no_grad()
    def _apply_patches_Y_to_X(self, img_y, M_rect_T):
        B = img_y.shape[0]
        patches = self.unfold_y(img_y)                               # (B, C*q^2, N_lr)
        BC, N_lr = B * self.C, patches.shape[-1]
        patches = patches.reshape(BC, self.p_out * self.p_out, N_lr) # (BC, q^2, N_lr)
        patches = torch.einsum('pq,bqn->bpn', M_rect_T.to(patches), patches)  # (BC, p^2, N_lr)
        patches = patches.reshape(B, self.C * (self.p ** 2), N_lr)
        x_img = self.fold_x(patches) / self.weights_x
        return x_img.to(self.dtype)


    @torch.no_grad()
    def _build_H_patch(self):
        p2 = self.p * self.p
        q2 = self.p_out * self.p_out
        B  = 1
        H_mat = torch.zeros(q2, p2, device=self.device, dtype=self.matrix_dtype)

        img = torch.zeros(B, self.C, self.height, self.width,
                        device=self.device, dtype=self.dtype)
        ph_idx = (self.height // self.p) // 2
        pw_idx = (self.width  // self.p) // 2
        h0, w0 = ph_idx * self.p, pw_idx * self.p

        img.zero_()
        img[0, 0, h0 + self.p // 2, w0 + self.p // 2] = 1.0
        y_flat = self.Hf.H(self._flatten(img))
        y_img  = y_flat.view(B, self.C, self.hL, self.wL).to(self.matrix_dtype)
        pchs   = self.unfold_y(y_img).view(self.C, q2, -1)  # (C,q2,N_lr)
        center_idx = int(torch.argmax((pchs[0] ** 2).sum(0)).item())

        for j in range(p2):
            rr, cc = divmod(j, self.p)
            img.zero_()
            img[0, 0, h0 + rr, w0 + cc] = 1.0
            y_flat = self.Hf.H(self._flatten(img))
            y_img  = y_flat.view(B, self.C, self.hL, self.wL).to(self.matrix_dtype)
            H_mat[:, j] = self.unfold_y(y_img).view(self.C, q2, -1)[0, :, center_idx]
        return H_mat


    # ---------- H_functions API ----------
    @torch.no_grad()
    def H(self, inp):
        B = inp.shape[0]

        if inp.ndim == 4:
            _, C, Hs, Ws = inp.shape
            if (C == self.C) and (Hs == self.height) and (Ws == self.width):
                # HR -> (H_patch) -> LR -> (W) -> LR
                y_lr = self._apply_patches_X_to_Y(inp, self.H_patch)
                y_lr = self._apply_patches_Y_to_Y(y_lr, self.W_patch)
                return self._flatten(y_lr)
            if (C == self.C) and (Hs == self.hL) and (Ws == self.wL):
                # LR -> (W) -> LR
                y_w = self._apply_patches_Y_to_Y(inp.to(self.dtype), self.W_patch)
                return y_w.view(B, -1)
            raise RuntimeError(f"H(): unexpected 4D shape {tuple(inp.shape)}")

        if inp.ndim == 2:
            n = inp.shape[1]
            x_len = self.C*self.height*self.width
            y_len = self.C*self.hL*self.wL
            if n == x_len:
                return self.H(inp.view(B, self.C, self.height, self.width))
            if n == y_len:
                return self.H(inp.view(B, self.C, self.hL, self.wL))
            raise RuntimeError(f"H(): unexpected vector length {n} (expected {x_len} or {y_len})")

        raise RuntimeError(f"H(): unexpected ndim {inp.ndim}")

    @torch.no_grad()
    def H_pinv(self, y_flat):
        y_img = self._img_like_y(y_flat, y_flat.shape[0])
        x_img = self._apply_patches_Y_to_X(y_img, self.WHp_patch)  # (WH)^+ applied LR->HR
        return self._flatten(x_img)

    @torch.no_grad()
    def Ut(self, y_flat):
        B = y_flat.shape[0]
        q2 = self.p_out * self.p_out
        k  = int(self.sw.shape[0])  # = q2
        N_lr = (self.hL // self.p_out) * (self.wL // self.p_out)

        y_img   = self._img_like_y(y_flat, B)
        patches = self.unfold_y(y_img).view(B, self.C, q2, N_lr)      # (B,C,q2,N_lr)
        U_t     = self.Uw.transpose(-2, -1).to(patches)               # (k,q2)
        small   = torch.einsum('kq,bcqn->bckn', U_t, patches)         # (B,C,k,N_lr)

        return small.permute(0, 3, 1, 2).contiguous().view(B, N_lr * self.C * k)

    @torch.no_grad()
    def U(self, z_small):
        B  = z_small.shape[0]
        q2 = self.p_out * self.p_out
        k  = int(self.sw.shape[0])  # = q2
        N_lr = (self.hL // self.p_out) * (self.wL // self.p_out)
        assert z_small.shape[1] == N_lr * self.C * k, \
            f"U(): expected {N_lr*self.C*k}, got {z_small.shape[1]}"

        small   = z_small.view(B, N_lr, self.C, k).permute(0, 2, 3, 1).contiguous()  # (B,C,k,N_lr)
        BC      = B * self.C
        small   = small.view(BC, k, N_lr)
        Umat    = self.Uw.to(small)                              # (q2,k)
        lr_pchs = torch.matmul(Umat, small).view(B, self.C * q2, N_lr)
        y_img   = self.fold_y(lr_pchs) / self.weights_y
        return self._flatten(y_img)

    @torch.no_grad()
    def Vt(self, x_flat):
        B = x_flat.shape[0]
        if x_flat.ndim == 4:
            x_flat = x_flat.reshape(B, -1)

        p2   = self.p * self.p
        k    = int(self.sw.shape[0])        # q^2
        N_hr = (self.height // self.p) * (self.width // self.p)
        n_hr = self.C * p2 * N_hr
        assert x_flat.shape[1] == n_hr, f"Vt(): expected {n_hr}, got {x_flat.shape[1]}"

        x_img   = self._img_like_x(x_flat, B)
        patches = self.unfold_x(x_img).reshape(B, self.C, p2, N_hr)    # (B,C,p2,N_hr)
        V_t     = self.Vw.transpose(-2, -1).to(patches)                # (k,p2)
        small   = torch.einsum('kp,bcpn->bckn', V_t, patches)          # (B,C,k,N_hr)

        head = small.permute(0, 3, 1, 2).contiguous().view(B, N_hr * self.C * k)
        tail = torch.zeros(B, self.C * (p2 - k) * N_hr,
                        device=x_flat.device, dtype=small.dtype)
        return torch.cat([head, tail], dim=1)

    @torch.no_grad()
    def V(self, z_full):
        B    = z_full.shape[0]
        p2   = self.p * self.p
        k    = int(self.sw.shape[0])        # q^2
        N_hr = (self.height // self.p) * (self.width // self.p)
        n_hr = self.C * p2 * N_hr
        assert z_full.shape[1] == n_hr, f"V(): expected {n_hr}, got {z_full.shape[1]}"

        head  = z_full[:, : self.C * k * N_hr]
        small = head.view(B, N_hr, self.C, k).permute(0, 2, 3, 1).contiguous()  # (B,C,k,N_hr)
        BC    = B * self.C
        small = small.view(BC, k, N_hr)

        Vmat  = self.Vw.to(small)                               # (p2,k)
        hr_p  = torch.matmul(Vmat, small).view(B, self.C * p2, N_hr)
        x_img = self.fold_x(hr_p) / self.weights_x
        return self._flatten(x_img)

    @torch.no_grad()
    def add_zeros(self, z_small):
        B    = z_small.shape[0]
        p2   = self.p * self.p
        k    = int(self.sw.shape[0])
        N_hr = (self.height // self.p) * (self.width // self.p)
        expect = N_hr * self.C * k
        assert z_small.shape[1] == expect, f"add_zeros(): expected {expect}, got {z_small.shape[1]}"

        tail = torch.zeros(B, self.C * (p2 - k) * N_hr,
                        device=z_small.device, dtype=z_small.dtype)
        return torch.cat([z_small, tail], dim=1)

    @torch.no_grad()
    def singulars(self):
        k   = int(self.sw.shape[0])      # q^2
        Nlr = (self.hL // self.p_out) * (self.wL // self.p_out)
        s = self.sw.view(1, 1, k).repeat(Nlr, self.C, 1)
        return s.reshape(-1).to(self.dtype)
