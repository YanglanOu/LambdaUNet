import torch
from torch import nn, einsum
from einops import rearrange

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads
        self.dim_v = dim_v

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = True
        self.temporal = cfg.temporal
        self.cfg = cfg

        if self.local_contexts:
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
            
        if self.temporal:
            self.tem_conv = nn.Conv2d(dim_u, dim_k, (1, cfg.tr), padding = (0, cfg.tr // 2))

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        v_p = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
        λp = self.pos_conv(v_p)
        Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))


        if self.temporal:
            v_t = rearrange(v, '(g t) u v p -> (g p) u v t', t = self.cfg.var_t)
            λt = self.tem_conv(v_t)
            # λt = rearrange(λt, '(g p) u v t -> (g t) u v p', g = int(self.cfg.batchsize/len(self.cfg.gpus)), t = self.cfg.seq_len, v = self.dim_v)
            λt = rearrange(λt, '(g p) k v t -> (g t) k v p', p = hh*ww)
            Yt = einsum('b h k n, b k v n -> b h v n', q, λt)
            Y = Yc + Yp + Yt
        else:
            Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out