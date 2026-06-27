"""Model B: EchoCoDirScore — inject co-occurrence & directional priors as SCORES.

Role-matched injection (per the symmetric/antisymmetric split):
  - CO-OCCURRENCE C (symmetric, PPMI) -> additive PAIRWISE BIAS on the
    self-attention scores (attention is order-agnostic & pairwise, the natural
    home of a symmetric relation).  scores = QK^T/sqrt(d) + mask + alpha_co * C[ids,ids]
  - DIRECTIONAL D (antisymmetric, precedence) -> per-step INPUT-GAIN / SALIENCE
    modulation of the Mamba FORWARD (causal) scan ONLY (applying it to the flipped
    backward scan would be sign-anti-aligned since D is antisymmetric).
    xf_fwd = xf * (1 + dir_gate * D[item_{i-1}, item_i]); backward stays unbiased.

Why input-modulation and not a true Δ/decay edit: the Mamba branch uses the
mamba_ssm CUDA kernel (Mamba-1), which does not expose per-step Δ injection.
Modulating the per-step input magnitude is the kernel-compatible approximation of
"how much this transition drives the state". A true decay injection would require
switching to Mamba-2/SSD with a custom 1-semiseparable mask (TiM4Rec-style) — a
larger change, left out here to keep the backbone identical to EchoMambaSA.

C is kept FULL-RANK (dense V×V buffer) so the attention bias is the exact pairwise
co-occurrence (this is the point of the "score" approach vs the low-rank embedding
of Model A). Memory: V×V float32 (~0.6GB on Beauty, ~50MB on ML-1M/LastFM).
alpha_co / dir_gate init at 0 => the model starts identical to EchoMambaSA and
must LEARN to use the priors. Full-vocab CE.

Structure flag --echo_attn_raw: by default BOTH branches consume the FFT-filtered
xf (= FilterLayer(x)); with the flag the attention branch instead reads the RAW
pre-FFT x, i.e. (Freq -> Mamba) in series PARALLEL attention(x) — keeping the
frequency inductive bias off the sharp content-specific attention path (closer to
BSARec, where attention also sees raw x). The Mamba branch always reads xf.
"""
import math
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model.echomamba4rec import FilterLayer, GLU, _MAMBA_AVAILABLE
from model.cooc_dir import build_codir

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


class BiasedMultiHeadAttention(nn.Module):
    """Self-attention identical to model._modules.MultiHeadAttention but accepting
    an additive pairwise bias (B,1,L,L) added to the pre-softmax scores."""
    def __init__(self, args):
        super().__init__()
        assert args.hidden_size % args.num_attention_heads == 0
        self.h = args.num_attention_heads
        self.dh = args.hidden_size // self.h
        self.scale = math.sqrt(self.dh)
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).permute(0, 2, 1, 3)

    def forward(self, x, attention_mask, pair_bias=None):
        q, k, v = self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        scores = scores + attention_mask
        if pair_bias is not None:
            scores = scores + pair_bias            # (B,1,L,L) broadcast over heads
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        B, L, _ = x.shape
        ctx = ctx.view(B, L, self.h * self.dh)
        return self.LayerNorm(self.out_dropout(self.dense(ctx)) + x)


class CoDirScoreLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.hidden_size
        drop = args.hidden_dropout_prob
        self.inner = args.num_hidden_layers
        self.filter_layer = FilterLayer(args.max_seq_length, d, drop)
        self.norms_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.inner)])
        self.norms_b = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.inner)])
        self.mamba_f = nn.ModuleList([Mamba(d_model=d, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand) for _ in range(self.inner)])
        self.mamba_b = nn.ModuleList([Mamba(d_model=d, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand) for _ in range(self.inner)])
        self.dropout = nn.Dropout(drop)
        self.attention = BiasedMultiHeadAttention(args)
        # echo_attn_raw: attention sees RAW pre-FFT x (so freq is in series with Mamba
        # ONLY, parallel raw-x attention); default False = attention shares xf.
        self.attn_raw = bool(getattr(args, "echo_attn_raw", False))
        self.gate_logit = nn.Parameter(torch.zeros(1))     # mamba/attn fusion (sigmoid->0.5)
        self.alpha_co = nn.Parameter(torch.zeros(1))       # co-occurrence attn-bias scale (init off)
        self.dir_gate = nn.Parameter(torch.zeros(1))       # directional mamba input-modulation (init off)
        self.glu = GLU(d_model=d, dropout=drop)

    def forward(self, x, attention_mask, co_bias, dir_mod):
        xf = self.filter_layer(x)
        m = xf
        for i in range(self.inner):
            # directional prior modulates ONLY the forward (causal) scan: dir_mod[i]
            # = D[item_{i-1}, item_i] is a forward-precedence signal. Applying it to
            # the flipped backward scan would be sign-anti-aligned (D antisymmetric),
            # so the backward pass is left as unbiased bidirectional context.
            m_f = m * (1.0 + self.dir_gate * dir_mod)       # (B,L,1) broadcast
            fwd = self.norms_f[i](self.dropout(self.mamba_f[i](m_f)) + m)
            rev = torch.flip(m, [1])
            bwd = torch.flip(self.mamba_b[i](rev), [1])
            bwd = self.norms_b[i](self.dropout(bwd) + m)
            m = fwd + bwd
        # --- co-occurrence pairwise bias (attention branch)
        # attn_raw -> attention reads the raw pre-FFT x (freq in series w/ Mamba only);
        # else reads xf (shared filtered input, current structure).
        attn_in = x if self.attn_raw else xf
        a = self.attention(attn_in, attention_mask, self.alpha_co * co_bias)
        g = torch.sigmoid(self.gate_logit)
        return self.glu(g * m + (1.0 - g) * a)


class EchoCoDirScoreModel(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        if not _MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm unavailable")
        self.args = args
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        window = int(getattr(args, "codir_window", 3))
        cap = int(getattr(args, "codir_cap", args.max_seq_length))
        norm = str(getattr(args, "codir_norm", "l2"))
        dir_norm = str(getattr(args, "codir_dir_norm", "ratio_l2"))
        C, D = build_codir(args.data_file, args.item_size, window, cap, norm=norm, dir_norm=dir_norm)
        self.register_buffer("C", torch.tensor(C.toarray(), dtype=torch.float32))   # (V,V) symmetric PPMI
        self.register_buffer("D", torch.tensor(D.toarray(), dtype=torch.float32))   # (V,V) antisymmetric

        self.layers = nn.ModuleList([CoDirScoreLayer(args)
                                     for _ in range(args.num_hidden_layers)])
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        mask = self.get_attention_mask(input_ids)                 # causal
        x = self.add_position_embedding(input_ids)
        # pairwise co-occurrence bias: co_bias[b,i,j] = C[ids[b,i], ids[b,j]]
        co_bias = self.C[input_ids.unsqueeze(2), input_ids.unsqueeze(1)].unsqueeze(1)  # (B,1,L,L)
        # directional consecutive-pair modulation: dir_mod[b,i] = D[ids[b,i-1], ids[b,i]]
        prev, cur = input_ids[:, :-1], input_ids[:, 1:]
        dvals = self.D[prev, cur]                                  # (B,L-1)
        zeros = torch.zeros(input_ids.size(0), 1, dtype=dvals.dtype, device=dvals.device)
        dir_mod = torch.cat([zeros, dvals], dim=1).unsqueeze(-1)   # (B,L,1)
        for layer in self.layers:
            x = layer(x, mask, co_bias, dir_mod)
        return [x] if all_sequence_output else x

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
