import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_with_attnmap(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x= torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    # get attn map
    d_k = q.shape[-1]  # head_dim (D)
    attn_map = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, L, L]
    return x, attn_map

def attention_with_attnmap_injection(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attnmap_idxs, old_attnmaps) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # original attn
    # x= torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # x = rearrange(x, "B H L D -> B L (H D)")

    # get attn map
    d_k = q.shape[-1]  # head_dim (D)
    attn_map = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, L, L]
    attn_map = torch.softmax(attn_map, dim=-1)
    # inject attn map
    for idx,old_attnmap in zip(attnmap_idxs,old_attnmaps):
        attn_map[:,:,512:,idx] = old_attnmap
    x = attn_map @ v
    return x, attn_map

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
