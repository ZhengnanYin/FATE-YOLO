import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from ultralytics.nn.modules.head import Detect


__all__ = (
    "FTConv",
    "DAABlock",
    "GLDBlock",
    "FATEHead",
)

class FTConv(nn.Module):
    """
    Hybrid FT–Taylor convolution combining a lightweight shared-coefficient FT branch
    with a standard 3×3 convolution branch. Learnable mixing ensures stability and performance.

    Args:
      in_channels, out_channels: channel dims
      kernel_size: spatial kernel size for FT branch
      stride, padding: convolution parameters
      U, V, T: FT frequency and Taylor order
      separable: whether FT branch is depthwise separable
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        U: int = 1,
        V: int = 1,
        T: int = 2,
        separable: bool = True,
        sp: bool = True,
        g: int = 8,
        act: bool = True,
    ):
        super().__init__()
        # Basic
        self.in_channels, self.out_channels = in_channels, out_channels
        self.K = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.U, self.V, self.T = U, V, T
        self.separable = separable
        self.act = act

        # FT–Taylor branch (shared coefficients)
        self.G = nn.Parameter(torch.zeros(1, 1, 2*U+1, 2*V+1))
        self.mask_logits = nn.Parameter(torch.full_like(self.G, -5.0))
        with torch.no_grad():
            self.G.data[0,0,U,V] = 1.0
        if separable:
            self.ft_pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.ft_pw = None

        # Standard Conv branch (3×3 conv)
        if sp == True:
            self.std_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride=stride, padding=self.padding,
                groups=g,
                bias=False
            )
        else:
            self.std_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride=stride, padding=self.padding,
                groups=1,
                bias=False
            )
        # Mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Post-processing
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self._cached_w = None

        # Precompute grid
        coords = torch.arange(self.K, dtype=torch.float32) - (self.K - 1) / 2
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")

        self.register_buffer(
            "grid_x", gx.clone().contiguous(), persistent=False
        )
        self.register_buffer(
            "grid_y", gy.clone().contiguous(), persistent=False
        )

    def clear_cache(self):
        self._cached_w = None

    def reconstruct_ft_kernel(self):
        mask = torch.sigmoid(self.mask_logits)
        Gs = (self.G * mask).to(self.G.device)
        invK = 2 * math.pi / self.K
        # build spatial kernel g [K,K]
        g = torch.zeros(self.K, self.K, device=Gs.device)
        for u_idx in range(2*self.U+1):
            for v_idx in range(2*self.V+1):
                w_uv = Gs[0,0,u_idx,v_idx]
                theta = invK * ((u_idx-self.U)*self.grid_x + (v_idx-self.V)*self.grid_y)
                poly = torch.zeros_like(theta, dtype=torch.cfloat)
                for t in range(self.T+1):
                    poly += (1j**t)/math.factorial(t)*(theta**t)
                g = g + w_uv * poly.real
        # expand to weight w
        if self.separable:
            w = g.unsqueeze(0).unsqueeze(0).expand(self.in_channels, 1, self.K, self.K)
        else:
            w = g.unsqueeze(0).unsqueeze(0).expand(self.out_channels, self.in_channels, self.K, self.K)
        return w

    def forward(self, x):
        # Standard path
        std_out = self.std_conv(x)
        std_out = std_out.to(x.device, x.dtype)

        # FT–Taylor path
        if not self.training and self._cached_w is not None:
            w = self._cached_w
            if w.device != x.device or w.dtype != std_out.dtype:
                w = w.to(device=x.device, dtype=std_out.dtype)
                self._cached_w = w
        else:
            w = self.reconstruct_ft_kernel().to(x.device, std_out.dtype)
            if not self.training:
                self._cached_w = w.detach().clone()
        if self.separable:
            ft = F.conv2d(x, w, None, self.stride, self.padding, groups=self.in_channels)
            ft_out = self.ft_pw(ft)
        else:
            ft_out = F.conv2d(x, w, None, self.stride, self.padding)

        # Mix
        alpha = torch.sigmoid(self.alpha)
        out = alpha * ft_out + (1 - alpha) * std_out
        # print(out.shape)

        # Residual if same shape
        if out.shape == x.shape:
            out = out + x
        if self.act:
            return self.act(self.bn(out))
        else:
            return self.bn(out)


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if len(x) != 2:
        raise ValueError("window_size must be int or pair")
    return (int(x[0]), int(x[1]))


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]):
    """[B,C,H,W] -> [B*nwin,C,wh,ww], with right/bottom zero padding."""
    b, c, h, w = x.shape
    wh, ww = window_size
    ph, pw = (-h) % wh, (-w) % ww
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    nh, nw = hp // wh, wp // ww
    x = (x.reshape(b, c, nh, wh, nw, ww)
           .permute(0, 2, 4, 1, 3, 5)
           .reshape(b * nh * nw, c, wh, ww))
    return x, (b, c, h, w, hp, wp, nh, nw, wh, ww)


def window_reverse(xw: torch.Tensor, meta):
    """Inverse of window_partition, allowing channel count to change."""
    b, _, h, w, hp, wp, nh, nw, wh, ww = meta
    c = xw.shape[1]
    x = (xw.reshape(b, nh, nw, c, wh, ww)
           .permute(0, 3, 1, 4, 2, 5)
           .reshape(b, c, hp, wp))
    return x[:, :, :h, :w].contiguous()


def _reshape_heads(x: torch.Tensor, heads: int):
    """[B,C,H,W] -> [B,heads,H*W,C/heads]"""
    b, c, h, w = x.shape
    return (x.flatten(2).transpose(1, 2)
             .reshape(b, h * w, heads, c // heads)
             .transpose(1, 2))


def _channel_ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # LayerNorm operates over the channel dimension, not H or W.
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class GLAA(nn.Module):
    """Global--Local Affinity Aggregation.

    Local branch: Q/K/V from windows; attention computed within each window.
    Global branch: Q from each pixel; K/V from adaptive-pooled feature map.
    Returns (y_local, y_global, entropy_local, entropy_global), all BCHW;
    the two entropy maps have exactly one channel, values in [0,1].

    max_windows_per_chunk and query_chunk_size limit peak attention memory.
    """
    def __init__(
        self,
        c: int,
        window_size: Union[int, Tuple[int, int]] = 7,
        global_size: Union[int, Tuple[int, int]] = 7,
        heads: int = 4,
        attn_dim: Optional[int] = None,
        attn_tau: float = 0.7,
        attn_drop: float = 0.0,
        max_windows_per_chunk: int = 256,
        query_chunk_size: int = 1024,
    ):
        super().__init__()
        if c <= 0 or heads <= 0 or attn_tau <= 0:
            raise ValueError("c, heads, and attn_tau must all be positive")
        if not (0.0 <= attn_drop < 1.0):
            raise ValueError("attn_drop must be in [0,1)")
        self.c = int(c)
        self.window_size = _pair(window_size)
        self.global_size = _pair(global_size)
        if min(*self.window_size, *self.global_size) <= 0:
            raise ValueError("window_size and global_size must be positive")
        self.heads = heads
        attn_dim = max(8, c // 4) if attn_dim is None else int(attn_dim)
        self.inner = max(heads, math.ceil(attn_dim / heads) * heads)
        self.head_dim = self.inner // heads
        self.scale = self.head_dim ** -0.5 / float(attn_tau)
        self.attn_drop = attn_drop
        self.max_windows_per_chunk = int(max_windows_per_chunk)
        self.query_chunk_size = int(query_chunk_size)
        if min(self.max_windows_per_chunk, self.query_chunk_size) <= 0:
            raise ValueError("chunk sizes must be positive")

        self.q_local = nn.Conv2d(c, self.inner, 1, bias=False)
        self.k_local = nn.Conv2d(c, self.inner, 1, bias=False)
        self.v_local = nn.Conv2d(c, self.inner, 1, bias=False)
        self.local_proj = nn.Conv2d(self.inner, c, 1, bias=False)

        self.q_global = nn.Conv2d(c, self.inner, 1, bias=False)
        self.k_global = nn.Conv2d(c, self.inner, 1, bias=False)
        self.v_global = nn.Conv2d(c, self.inner, 1, bias=False)
        self.global_proj = nn.Conv2d(self.inner, c, 1, bias=False)

    def _local(self, x: torch.Tensor):
        b, _, h, w = x.shape
        q_w, meta = window_partition(self.q_local(x), self.window_size)
        k_w, _ = window_partition(self.k_local(x), self.window_size)
        v_w, _ = window_partition(self.v_local(x), self.window_size)
        valid_w, _ = window_partition(x.new_ones((b, 1, h, w)), self.window_size)

        q, k, v = (_reshape_heads(t, self.heads) for t in (q_w, k_w, v_w))
        valid_keys = valid_w.flatten(2).squeeze(1).bool()  # [B*nwin,N]
        valid_count = valid_keys.sum(-1).clamp_min(2).float()
        n = q.size(-2)
        wh, ww = self.window_size

        outputs, entropies = [], []
        for start in range(0, q.size(0), self.max_windows_per_chunk):
            end = start + self.max_windows_per_chunk
            qi, ki, vi = q[start:end], k[start:end], v[start:end]
            logits = torch.matmul(qi.float(), ki.float().transpose(-1, -2)) * self.scale
            logits = logits.masked_fill(
                ~valid_keys[start:end, None, None, :],
                torch.finfo(logits.dtype).min,
            )
            prob = torch.softmax(logits, dim=-1)
            # Entropy is computed BEFORE dropout, from normalized affinities.
            ent = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            ent = ent / valid_count[start:end, None, None].log()
            ent = ent.mean(dim=1).clamp(0.0, 1.0)  # [chunk,N]
            prob_used = F.dropout(prob, p=self.attn_drop, training=self.training)
            yi = torch.matmul(prob_used, vi.float())
            yi = (yi.transpose(1, 2).contiguous()
                    .reshape(yi.size(0), n, self.inner)
                    .transpose(1, 2).reshape(-1, self.inner, wh, ww))
            outputs.append(yi.to(dtype=x.dtype))
            entropies.append(ent[:, None, :].reshape(-1, 1, wh, ww))

        yl = window_reverse(torch.cat(outputs, dim=0), meta)
        el = window_reverse(torch.cat(entropies, dim=0), meta)
        return self.local_proj(yl), el

    def _global(self, x: torch.Tensor):
        b, _, h, w = x.shape
        gh = min(self.global_size[0], h)
        gw = min(self.global_size[1], w)
        pooled = F.adaptive_avg_pool2d(x, (gh, gw))
        q = _reshape_heads(self.q_global(x), self.heads)  # [B,heads,HW,d]
        k = _reshape_heads(self.k_global(pooled), self.heads)  # [B,heads,S,d]
        v = _reshape_heads(self.v_global(pooled), self.heads)
        support = gh * gw

        outputs, entropies = [], []
        for start in range(0, h * w, self.query_chunk_size):
            end = start + self.query_chunk_size
            qi = q[:, :, start:end, :]
            logits = torch.matmul(qi.float(), k.float().transpose(-1, -2)) * self.scale
            prob = torch.softmax(logits, dim=-1)
            if support > 1:
                ent = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1) / math.log(support)
            else:
                ent = torch.zeros_like(prob[..., 0])
            entropies.append(ent.mean(dim=1).clamp(0.0, 1.0))
            prob_used = F.dropout(prob, p=self.attn_drop, training=self.training)
            outputs.append(torch.matmul(prob_used, v.float()))

        yg = torch.cat(outputs, dim=2)
        yg = (yg.transpose(1, 2).contiguous()
              .reshape(b, h * w, self.inner)
              .transpose(1, 2).reshape(b, self.inner, h, w))
        eg = torch.cat(entropies, dim=1).reshape(b, 1, h, w)
        return self.global_proj(yg.to(dtype=x.dtype)), eg

    def forward(self, x: torch.Tensor):
        yl, ent_l = self._local(x)
        yg, ent_g = self._global(x)
        return yl, yg, ent_l, ent_g


class EGAA(nn.Module):
    """Entropy-Gated Affinity Aggregation (paper Eqs. 5--7).

    alpha_ent(h) = 1 - normalized_attention_entropy(h)
    alpha(h) = gamma * alpha_edge(h) + (1-gamma) * alpha_ent(h)
    y(h) = alpha(h)*y_local(h) + (1-alpha(h))*y_global(h)

    ``entropy_source`` is a paper-underspecified choice: 'local' (default),
    'global', or 'mean'. ``gamma`` is made learnable via sigmoid(logit).
    """
    def __init__(
        self,
        use_entropy_gate: bool = True,
        use_edge_gate: bool = True,
        gate_blend: float = 0.6,
        entropy_source: str = "local",
        edge_pool: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        if entropy_source not in ("local", "global", "mean"):
            raise ValueError("entropy_source must be local, global or mean")
        if not (0.0 <= gate_blend <= 1.0):
            raise ValueError("gate_blend must be in [0,1]")
        if edge_pool < 1 or edge_pool % 2 != 1:
            raise ValueError("edge_pool must be a positive odd integer")
        self.use_entropy_gate = bool(use_entropy_gate)
        self.use_edge_gate = bool(use_edge_gate)
        self.entropy_source = entropy_source
        self.edge_pool = edge_pool
        self.eps = eps

        clipped = min(1 - 1e-5, max(1e-5, gate_blend))
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(clipped / (1 - clipped))))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None],
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)[None, None],
            persistent=False,
        )

    def _edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gray = x.float().mean(dim=1, keepdim=True)
        gray = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype))
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype))
        edge = torch.sqrt(gx.square() + gy.square() + self.eps)
        if self.edge_pool > 1:
            edge = F.avg_pool2d(
                edge, kernel_size=self.edge_pool, stride=1,
                padding=self.edge_pool // 2, count_include_pad=False,
            )
        # Per-sample min-max normalization preserves pixel-wise edge modulation.
        low = edge.flatten(2).amin(-1).reshape(b, 1, 1, 1)
        high = edge.flatten(2).amax(-1).reshape(b, 1, 1, 1)
        return ((edge - low) / (high - low + self.eps)).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        y_local: torch.Tensor,
        y_global: torch.Tensor,
        entropy_local: torch.Tensor,
        entropy_global: torch.Tensor,
        return_aux: bool = False,
    ):
        if self.entropy_source == "local":
            entropy = entropy_local
        elif self.entropy_source == "global":
            entropy = entropy_global
        else:
            entropy = 0.5 * (entropy_local + entropy_global)

        alpha_ent = (1.0 - entropy).clamp(0.0, 1.0)
        if self.use_edge_gate:
            alpha_edge = self._edge_strength(x)
        else:
            alpha_edge = torch.zeros_like(alpha_ent)

        if self.use_entropy_gate and self.use_edge_gate:
            gamma = torch.sigmoid(self.gamma_logit)
            alpha = gamma * alpha_edge + (1.0 - gamma) * alpha_ent
        elif self.use_entropy_gate:
            alpha = alpha_ent
        elif self.use_edge_gate:
            alpha = alpha_edge
        else:
            alpha = torch.full_like(alpha_ent, 0.5)

        alpha = alpha.clamp(0.0, 1.0)
        y = alpha.to(y_local.dtype) * y_local + (1.0 - alpha).to(y_global.dtype) * y_global
        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "alpha": alpha.detach(),
                "alpha_entropy": alpha_ent.detach(),
                "alpha_edge": alpha_edge.detach(),
                "entropy_local": entropy_local.detach(),
                "entropy_global": entropy_global.detach(),
                "gamma": torch.sigmoid(self.gamma_logit).detach(),
            }
            return y, aux
        return y


class GLA(nn.Module):
    """Paper-style GLAA + EGAA without a residual connection.

    This keeps the historical GLA(C1,C2,...) interface. It is a replacement
    implementation, not the historical ProtoDynGL-based GLA from the question.
    """
    def __init__(
        self,
        C1: int,
        C2: int,
        G: int = 8,            # Legacy interface; not used as conv groups
        K: int = 3,            # Legacy interface; paper uses window affinities
        Sg: int = 7,
        Sl: int = 7,
        Sv: int = 5,           # Legacy interface; not used
        window_size: Optional[Union[int, Tuple[int, int]]] = None,
        attn_tau: float = 0.7,
        attn_drop: float = 0.0,
        use_entropy_gate: bool = True,
        use_edge_gate: bool = True,
        gate_blend: float = 0.6,
        k_mode: str = "hybrid",  # Legacy interface; pooled/global keys explicit
        v_mode: str = "hybrid",  # Legacy interface; pooled/global values explicit
        edge_pool: int = 3,
        heads: int = 4,
        attn_dim: Optional[int] = None,
        entropy_source: str = "local",
        max_windows_per_chunk: int = 256,
        query_chunk_size: int = 1024,
    ):
        super().__init__()
        del G, K, Sv, k_mode, v_mode  # Retain call compatibility, no silent fake use
        self.glaa = GLAA(
            c=C1,
            window_size=window_size if window_size is not None else Sl,
            global_size=Sg,
            heads=heads,
            attn_dim=attn_dim,
            attn_tau=attn_tau,
            attn_drop=attn_drop,
            max_windows_per_chunk=max_windows_per_chunk,
            query_chunk_size=query_chunk_size,
        )
        self.egaa = EGAA(
            use_entropy_gate=use_entropy_gate,
            use_edge_gate=use_edge_gate,
            gate_blend=gate_blend,
            entropy_source=entropy_source,
            edge_pool=edge_pool,
        )
        self.out_proj = nn.Conv2d(C1, C2, 1, bias=False)

    def forward(self, x: torch.Tensor, return_aux: bool = False,
                edge_source: Optional[torch.Tensor] = None):
        y_local, y_global, ent_local, ent_global = self.glaa(x)
        # Use pre-normalization features for Sobel when called from a Pre-LN block.
        # Channel LayerNorm makes channel-mean ~= 0, destroying edge signals.
        result = self.egaa(
            x if edge_source is None else edge_source,
            y_local, y_global, ent_local, ent_global,
            return_aux=return_aux,
        )
        if return_aux:
            fused, aux = result
            return self.out_proj(fused), aux
        return self.out_proj(result)


class DAABlock(nn.Module):
    """Pre-LN GLAA+EGAA, projected residual, optional channel change c->c2.

    Compatible with DAABlock(c, c2, r=16, window_size=7) calls.
    The old `r` argument is accepted for compatibility but is not used by the
    paper's affinity formulation.
    """
    def __init__(self, c: int, c2: int, r: int = 16,
                 window_size: Union[int, Tuple[int, int]] = 7, **kwargs):
        super().__init__()
        del r
        self.norm1 = nn.LayerNorm(c)
        self.gla = GLA(c, c2, window_size=window_size, **kwargs)
        self.shortcut = nn.Identity() if c == c2 else nn.Conv2d(c, c2, 1, bias=False)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        xn = _channel_ln(self.norm1, x)
        if return_aux:
            y, aux = self.gla(xn, return_aux=True, edge_source=x)
            return self.shortcut(x) + y, aux
        return self.shortcut(x) + self.gla(xn, edge_source=x)


class GLDBlock(nn.Module):
    """Drop-in GLDBlock(C1,C2,...) replacement with pre-LN, GLA and residual.

    Important: original user-provided GLDBlock employed a different
    ProtoDynGL_C2MainTiny operator. Replacing it changes the model.
    """
    def __init__(self, C1: int, C2: int,
                 norm_layer=nn.LayerNorm, mlp_ratio: float = 1.2, **kwargs):
        super().__init__()
        del mlp_ratio  # Previous GLDBlock also did not instantiate the MLP
        self.norm1 = norm_layer(C1)
        self.gla = GLA(C1, C2, **kwargs)
        self.project_out = nn.Conv2d(C1, C2, 1, bias=False)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        xn = _channel_ln(self.norm1, x)
        if return_aux:
            y, aux = self.gla(xn, return_aux=True, edge_source=x)
            return self.project_out(x) + y, aux
        return self.project_out(x) + self.gla(xn, edge_source=x)


class FATEHead(Detect):
    # Configurable prior to constructing YOLO model (e.g. FATEHead.head_width = 64)
    head_width = 64

    def __init__(self, nc: int = 80, reg_max: int = 16,
                 end2end: bool = False, ch: tuple[int, ...] = ()):
        if end2end:
            raise ValueError("FATEHead currently supports one-to-many Detect only (end2end=False).")
        if not ch:
            raise ValueError("FATEHead requires feature channels from the Ultralytics YAML parser.")
        # Initialize official decoding/stride/loss attributes, then replace
        # the expensive two-tower feature predictors.
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        hidden = int(self.head_width)
        if hidden <= 0:
            raise ValueError("head_width must be a positive integer")

        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ) for c in ch
        )

        # Shared convolution weights; per-scale normalization avoids mixing
        # the statistics of P3/P4/P5 (different spatial resolutions).
        self.shared_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.shared_pw = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.dw_norm = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in ch])
        self.pw_norm = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in ch])

        # Parameter-light local high-frequency / boundary response.
        self.boundary = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        nn.init.zeros_(self.boundary.weight)
        nn.init.zeros_(self.boundary.bias)

        # Per-scale, per-task strength; zero-init => identity when training begins.
        self.reg_strength = nn.Parameter(torch.zeros(self.nl))
        self.cls_strength = nn.Parameter(torch.zeros(self.nl))

        # Preserve standard Detect names and bias_init() expectations:
        # each head[i][-1] must be an nn.Conv2d with a bias.
        self.cv2 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, 4 * self.reg_max, 1, bias=True))
            for _ in ch
        ])
        self.cv3 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, nc, 1, bias=True))
            for _ in ch
        ])

    def forward_head(self, x: list[torch.Tensor], box_head=None, cls_head=None):
        # The official Detect.forward calls this method with
        # box_head=self.cv2 and cls_head=self.cv3 (via one2many).
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        b_preds, c_preds = [], []
        for i, xi in enumerate(x):
            zi = self.adapters[i](xi)
            zi = F.silu(self.dw_norm[i](self.shared_dw(zi)))
            zi = F.silu(self.pw_norm[i](self.shared_pw(zi)))

            # Spatial residual is a cheap high-frequency magnitude estimator.
            high = (zi - F.avg_pool2d(zi, kernel_size=3, stride=1, padding=1)).abs()
            response = high.mean(1, keepdim=True)
            boundary = torch.sigmoid(self.boundary(response))

            # Bound the strengths in [-1, 1] for numerical stability.
            sr = torch.tanh(self.reg_strength[i])
            sc = torch.tanh(self.cls_strength[i])
            reg_feat = zi * (1.0 + sr * boundary)
            cls_feat = zi * (1.0 + sc * (1.0 - boundary))

            b_preds.append(box_head[i](reg_feat).reshape(bs, 4 * self.reg_max, -1))
            c_preds.append(cls_head[i](cls_feat).reshape(bs, self.nc, -1))

        return {
            "boxes": torch.cat(b_preds, dim=-1),
            "scores": torch.cat(c_preds, dim=-1),
            "feats": x,  # required by official loss and stride initialization
        }