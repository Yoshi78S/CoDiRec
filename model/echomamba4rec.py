"""EchoMamba4Rec, ported from the submitted RecBole impl
(EchoMamba4Rec/EchoMamba4Rec_submitted/model.py) into the BSARec SequentialRecModel
framework as a baseline.

Faithful port: embed -> dropout -> LN -> [BiMambaLayer] x num_layers -> last (real) pos.
BiMambaLayer = FFT complex filter (FilterLayer) -> bidirectional Mamba (forward + flipped
backward, each residual+LN, summed) over an inner num_layers loop -> GLU. Item embedding
only (no position embedding). CE loss. init applies to all (clobbers Mamba) — as in the
original and the other BSARec Mamba baselines.

Note: the original instantiates a MultiQueryTransformerBlock in BiMambaLayer.__init__ but
NEVER calls it in forward (dead code); it is omitted here — no effect on the output."""
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except Exception as _e:
    Mamba = None
    _MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERR = _e


class FilterLayer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, dropout_prob):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x):
        seq_len = x.size(1)
        f = torch.fft.rfft(x, dim=1, norm='ortho')
        f = f * torch.view_as_complex(self.complex_weight)
        out = torch.fft.irfft(f, n=seq_len, dim=1, norm='ortho')
        return self.LayerNorm(self.out_dropout(out) + x)


class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        value, gate = self.fc1(x).chunk(2, dim=-1)
        gated = self.fc2(value * torch.sigmoid(gate))
        return self.LayerNorm(self.dropout(gated + x))


class BiMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, max_seq_length):
        super().__init__()
        self.num_layers = num_layers
        self.filter_layer = FilterLayer(max_seq_length, d_model, dropout)
        self.norms_forward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_backward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.mamba_forwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)])
        self.mamba_backwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model=d_model, dropout=dropout)

    def forward(self, x):
        x = self.filter_layer(x)
        for i in range(self.num_layers):
            fwd = self.norms_forward[i](self.dropout(self.mamba_forwards[i](x)) + x)
            rev = torch.flip(x, [1])
            bwd = torch.flip(self.mamba_backwards[i](rev), [1])
            bwd = self.norms_backward[i](self.dropout(bwd) + x)
            x = fwd + bwd
        return self.glu(x)


class EchoMamba4RecModel(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        if not _MAMBA_AVAILABLE:
            raise ImportError(f"mamba-ssm unavailable: {_MAMBA_IMPORT_ERR}")
        self.args = args
        self.num_layers = args.num_hidden_layers
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.mamba_layers = nn.ModuleList([
            BiMambaLayer(args.hidden_size, args.d_state, args.d_conv, args.expand,
                         args.hidden_dropout_prob, self.num_layers, args.max_seq_length)
            for _ in range(self.num_layers)
        ])
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        emb = self.LayerNorm(self.dropout(self.item_embeddings(input_ids)))   # item only, no pos
        for layer in self.mamba_layers:
            emb = layer(emb)
        return [emb] if all_sequence_output else emb

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(seq_output, self.item_embeddings.weight.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)
