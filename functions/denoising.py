import torch
from tqdm import tqdm


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(
        0, t + 1
    ).view(-1, 1, 1, 1)


@torch.no_grad()
def efficient_generalized_steps(
    x, seq, model, b, H_funcs, y_0, sigma_0,
    etaB, etaA, etaC, cls_fn=None, classes=None, deg=""
):
    is_sr = deg.startswith("sr")
    device = x.device
    n = x.size(0)
    dim = x.shape[1] * x.shape[2] * x.shape[3]

    # Setup vectors
    singulars = H_funcs.singulars()
    Sigma = torch.zeros(dim, device=device)
    Sigma[:singulars.shape[0]] = singulars

    U_t_y = H_funcs.Ut(y_0)
    Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

    # Initialize x_T
    largest_alphas = compute_alpha(
        b, torch.full((n,), seq[-1], device=device, dtype=torch.long)
    )
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()

    large_singulars_index = torch.where(
        singulars * largest_sigmas[0, 0, 0, 0] > sigma_0
    )

    inv_singulars_and_zero = torch.zeros(dim, device=device)
    inv_singulars_and_zero[large_singulars_index] = (
        sigma_0 / singulars[large_singulars_index]
    )
    inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

    init_y = torch.zeros(n, dim, device=device)
    init_y[:, large_singulars_index[0]] = (
        U_t_y[:, large_singulars_index[0]]
        / singulars[large_singulars_index].view(1, -1)
    )
    init_y = init_y.reshape_as(x)

    remaining_s = (
        largest_sigmas.view(-1, 1) ** 2
        - inv_singulars_and_zero ** 2
    )
    remaining_s = remaining_s.reshape_as(x).clamp_min(0.0).sqrt()

    init_y = (init_y + remaining_s * x) / largest_sigmas

    # Setup iteration variables
    x = H_funcs.V(init_y.reshape(n, -1)).reshape_as(x)

    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # Iterate over diffusion timesteps
    for i, j in tqdm(
        zip(reversed(seq), reversed(seq_next)),
        total=len(seq)
    ):
        t = torch.full((n,), i, device=device, dtype=torch.float32)
        next_t = torch.full((n,), j, device=device, dtype=torch.float32)

        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())

        xt = xs[-1].to(device)

        # Predict noise
        if cls_fn is None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(
                x, t, classes
            )

        if et.size(1) == 6:
            et = et[:, :3]

        # Predict clean sample
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        V_t_x0 = H_funcs.Vt(x0_t)

        # Preserve null-space residual for super-resolution
        if is_sr:
            x0_head = H_funcs.V(V_t_x0).reshape_as(x0_t)
            r0 = x0_t - x0_head

        # Data conditioning in coefficient space
        sigma_next = (
            (1 - at_next).sqrt()[0, 0, 0, 0]
            / at_next.sqrt()[0, 0, 0, 0]
        )

        SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

        falses = torch.zeros(
            V_t_x0.shape[1] - singulars.shape[0],
            dtype=torch.bool,
            device=device
        )

        cond_before_lite = singulars * sigma_next > sigma_0
        cond_after_lite = singulars * sigma_next < sigma_0

        cond_before = torch.hstack((cond_before_lite, falses))
        cond_after = torch.hstack((cond_after_lite, falses))

        std_nextC = sigma_next * etaC
        sigma_tilde_nextC = torch.sqrt(
            sigma_next ** 2 - std_nextC ** 2
        )

        std_nextA = sigma_next * etaA
        sigma_tilde_nextA = torch.sqrt(
            sigma_next ** 2 - std_nextA ** 2
        )

        diff_sigma_t_nextB = torch.sqrt(
            sigma_next ** 2
            - sigma_0 ** 2
            / singulars[cond_before_lite] ** 2
            * etaB ** 2
        )

        # Case C: Missing components
        Vt_xt_mod_next = (
            V_t_x0
            + sigma_tilde_nextC * H_funcs.Vt(et)
            + std_nextC * torch.randn_like(V_t_x0)
        )

        # Case A: Less noisy than y
        Vt_xt_mod_next[:, cond_after] = (
            V_t_x0[:, cond_after]
            + sigma_tilde_nextA
            * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
            + std_nextA
            * torch.randn_like(V_t_x0[:, cond_after])
        )

        # Case B: Noisier than y
        Vt_xt_mod_next[:, cond_before] = (
            etaB * Sig_inv_U_t_y[:, cond_before_lite]
            + (1 - etaB) * V_t_x0[:, cond_before]
            + diff_sigma_t_nextB
            * torch.randn_like(U_t_y)[:, cond_before_lite]
        )

        # Reconstruct in image space
        xt_mod_next = H_funcs.V(Vt_xt_mod_next).reshape_as(x0_t)

        # Restore null-space residual for super-resolution
        if is_sr:
            xt_mod_next = xt_mod_next + r0

        # Scale back to the next diffusion timestep
        xt_next = (
            at_next.sqrt()[0, 0, 0, 0] * xt_mod_next
        ).reshape_as(xt)

        x0_preds.append(x0_t.cpu())
        xs.append(xt_next.cpu())

    return xs, x0_preds