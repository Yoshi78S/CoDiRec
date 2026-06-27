"""Build global item-item CO-OCCURRENCE (symmetric, PPMI) and DIRECTIONAL
(antisymmetric, precedence-asymmetry) matrices from the TRAIN portion only.

Leak discipline: matrices use each user's train pool s[:-2] (test transition
s[-2]->s[-1] and valid target s[-2] never enter), capped to the last `cap`
items (= what the model can see). Padding index 0 stays all-zero (items >= 1).

  - C (co-occurrence): items within a window of `window` consecutive positions
    are co-occurring (symmetric). Value = PPMI = max(0, log( c_ab * Z / (c_a*c_b) )).
  - D (directional): for EVERY ordered position pair i<j in the capped sequence
    (whole-sequence precedence; adjacent and non-adjacent pairs both count), item
    tr[i] precedes tr[j]. With forward count F, D = (F - F^T) / (F + F^T) in
    [-1,1], antisymmetric (D = -D^T). (Note: C is LOCAL/windowed, D is GLOBAL.)

Returned as scipy.sparse CSR of shape (item_size, item_size); cached to disk.
"""
import os
import numpy as np
import scipy.sparse as sp
from collections import Counter


def _load_seqs(data_file):
    seqs = []
    for line in open(data_file):
        toks = line.strip().split(' ')
        seqs.append([int(x) for x in toks[1:]])
    return seqs


def build_codir(data_file, item_size, window=3, cap=50, norm='l2', dir_norm='ratio_l2',
                cache_dir='output/codir_cache'):
    """norm controls the co-occurrence C normalization:
       'l2'   = raw window counts, each ROW L2-normalized (popularity NOT modeled;
                magnitude removed, cosine-style profile). [default]
       'ppmi' = max(0, log(c_ab*Z/(c_a*c_b))) — popularity-corrected.
       'raw'  = raw counts (no normalization).
    dir_norm controls the directional D normalization:
       'ratio'    = (F-F^T)/(F+F^T) in [-1,1], antisymmetric (per-pair co-occ removed).
       'ratio_l2' = the ratio, then each ROW L2-normalized (also removes per-item
                    activity scale; breaks exact antisymmetry but keeps sign). [default]"""
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(data_file))[0]
    cache = os.path.join(cache_dir, f"codir_{name}_w{window}_cap{cap}_V{item_size}_{norm}_{dir_norm}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        shp = tuple(z['shp'])
        C = sp.csr_matrix((z['Cd'], z['Ci'], z['Cp']), shape=shp)
        D = sp.csr_matrix((z['Dd'], z['Di'], z['Dp']), shape=shp)
        return C, D

    cooc = Counter()      # symmetric window co-occurrence counts
    fwd = Counter()       # directional precedence counts (a before b)
    comarg = Counter()    # co-occurrence marginal per item (row sum of cooc)
    Z = 0                 # total co-occurrence mass
    for s in _load_seqs(data_file):
        if len(s) < 3:
            continue
        tr = s[:-2]
        if cap and cap > 0:
            tr = tr[-cap:]
        n = len(tr)
        # symmetric co-occurrence within a sliding window
        for i in range(n):
            for j in range(i + 1, min(i + window, n)):
                a, b = tr[i], tr[j]
                if a == b:
                    continue
                cooc[(a, b)] += 1
                cooc[(b, a)] += 1
                comarg[a] += 1
                comarg[b] += 1
                Z += 2
        # directional precedence over ALL ordered position pairs (non-adjacent)
        for i in range(n):
            ai = tr[i]
            for j in range(i + 1, n):
                b = tr[j]
                if ai == b:
                    continue
                fwd[(ai, b)] += 1

    # ---- raw symmetric co-occurrence, then normalize per `norm`
    crows = np.fromiter((a for (a, b) in cooc), dtype=np.int64)
    ccols = np.fromiter((b for (a, b) in cooc), dtype=np.int64)
    cvals = np.fromiter((cooc[(a, b)] for (a, b) in cooc), dtype=np.float32)
    C_raw = sp.csr_matrix((cvals, (crows, ccols)), shape=(item_size, item_size), dtype=np.float32)
    if norm == 'ppmi':
        comarg_arr = np.zeros(item_size, dtype=np.float64)
        for a, c in comarg.items():
            comarg_arr[a] = c
        logZ = np.log(Z) if Z > 0 else 0.0
        coo = C_raw.tocoo()
        pm = np.log(coo.data) + logZ - np.log(comarg_arr[coo.row]) - np.log(comarg_arr[coo.col])
        pm = np.clip(pm, 0, None).astype(np.float32)
        C = sp.csr_matrix((pm, (coo.row, coo.col)), shape=C_raw.shape, dtype=np.float32)
        C.eliminate_zeros()
    elif norm == 'l2':
        rownorm = np.sqrt(np.asarray(C_raw.multiply(C_raw).sum(axis=1)).ravel())
        rownorm[rownorm == 0] = 1.0
        C = (sp.diags((1.0 / rownorm).astype(np.float32)) @ C_raw).tocsr()
    else:  # 'raw'
        C = C_raw

    # ---- antisymmetric asymmetry ratio for D
    rows, cols, vals = [], [], []
    done = set()
    for (a, b) in list(fwd.keys()):
        if (a, b) in done or (b, a) in done:
            continue
        f_ab = fwd.get((a, b), 0)
        f_ba = fwd.get((b, a), 0)
        tot = f_ab + f_ba
        if tot == 0:
            continue
        r = (f_ab - f_ba) / tot          # in [-1, 1]
        if r != 0:
            rows.append(a); cols.append(b); vals.append(np.float32(r))
            rows.append(b); cols.append(a); vals.append(np.float32(-r))
        done.add((a, b))
    D = sp.csr_matrix((vals, (rows, cols)), shape=(item_size, item_size), dtype=np.float32)
    if dir_norm == 'ratio_l2':
        # treat each item's directional ratio row as its vector and L2-normalize it
        # (removes per-item activity/popularity scale; note: breaks exact antisymmetry,
        #  but the SIGN per entry — the direction — is preserved).
        dn = np.sqrt(np.asarray(D.multiply(D).sum(axis=1)).ravel())
        dn[dn == 0] = 1.0
        D = (sp.diags((1.0 / dn).astype(np.float32)) @ D).tocsr()

    np.savez(cache,
             Cd=C.data, Ci=C.indices, Cp=C.indptr,
             Dd=D.data, Di=D.indices, Dp=D.indptr,
             shp=np.array(C.shape))
    return C, D


def svd_features(M, rank=64, antisym=False):
    """Truncated-SVD item features from a (sparse) item-item matrix M = U S V^T.

    - antisym=False (symmetric M, e.g. PPMI): feat[i] = (U sqrt(S))[i] -> (V, rank).
      For a symmetric matrix this is the Eckart-Young-optimal rank-`rank`
      factorization (U == V up to sign), so a single per-item vector suffices.
    - antisym=True (antisymmetric M, e.g. directional D): a single U sqrt(S) loses
      the SIGN/direction (its Gram is symmetric). We keep BOTH factors:
      feat[i] = concat( (U sqrt(S))[i], (V sqrt(S))[i] ) -> (V, 2*rank), so that
      D[i,j] = (U sqrt S)[i] . (V sqrt S)[j] is recoverable and the per-item
      feature carries both the "leads" (U) and "follows" (V) profile.

    Deterministic (fixed v0) so features are identical across training seeds."""
    from scipy.sparse.linalg import svds
    out_dim = rank * (2 if antisym else 1)
    k = min(rank, min(M.shape) - 1)
    if M.nnz == 0 or k < 1:
        return np.zeros((M.shape[0], out_dim), dtype=np.float32)
    v0 = np.ones(min(M.shape), dtype=np.float64) / np.sqrt(min(M.shape))
    try:
        u, sgl, vt = svds(M.asfptype(), k=k, v0=v0)
    except Exception:
        return np.zeros((M.shape[0], out_dim), dtype=np.float32)
    rs = np.sqrt(np.clip(sgl, 0, None))[None, :]
    su = (u * rs).astype(np.float32)                        # (V, k)
    if antisym:
        sv = (vt.T * rs).astype(np.float32)                # (V, k)
        feat = np.concatenate([su, sv], axis=1)            # (V, 2k)
    else:
        feat = su                                          # (V, k)
    if feat.shape[1] < out_dim:                             # pad if rank capped by k
        feat = np.concatenate(
            [feat, np.zeros((feat.shape[0], out_dim - feat.shape[1]), np.float32)], 1)
    return feat.astype(np.float32)


def build_augment_associates(data_file, item_size, topk=50, cap=50,
                             cache_dir='output/codir_cache'):
    """For sequence augmentation (proposed method): per item x, the top-`topk`
    associate items j ranked by  PPMI[x,j] * relu(D[x,j])  -- j both co-occurs
    strongly with x (all-pairs PPMI) AND tends to follow x (forward direction
    D>0). Built leak-safely from the TRAIN pool s[:-2] (capped to last `cap`).
    Returns an int array (item_size, topk), -1 padding for empty slots. Cached."""
    import math
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(data_file))[0]
    cache = os.path.join(cache_dir, f"augassoc_{name}_top{topk}_cap{cap}_V{item_size}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    co = Counter()       # symmetric all-pairs co-occurrence
    fwd = Counter()      # forward (a before b)
    for s in _load_seqs(data_file):
        if len(s) < 3:
            continue
        tr = s[:-2]
        if cap and cap > 0:
            tr = tr[-cap:]
        n = len(tr)
        for i in range(n):
            a = tr[i]
            for j in range(i + 1, n):
                b = tr[j]
                if a == b:
                    continue
                co[(a, b)] += 1; co[(b, a)] += 1
                fwd[(a, b)] += 1
    rowsum = np.zeros(item_size, dtype=np.float64)
    for (a, b), c in co.items():
        rowsum[a] += c
    Z = rowsum.sum()
    logZ = math.log(Z) if Z > 0 else 0.0

    cand = [[] for _ in range(item_size)]            # per-item (score, j)
    for (a, b), c_ab in co.items():
        if rowsum[a] <= 0 or rowsum[b] <= 0:
            continue
        ppmi = math.log(c_ab) + logZ - math.log(rowsum[a]) - math.log(rowsum[b])
        if ppmi <= 0:
            continue
        f_ab = fwd.get((a, b), 0); f_ba = fwd.get((b, a), 0); tot = f_ab + f_ba
        if tot == 0:
            continue
        d = (f_ab - f_ba) / tot                       # in [-1,1]
        if d <= 0:                                     # keep only forward (x precedes j)
            continue
        cand[a].append((ppmi * d, b))

    assoc = np.full((item_size, topk), -1, dtype=np.int64)
    for a in range(item_size):
        if not cand[a]:
            continue
        top = sorted(cand[a], key=lambda t: -t[0])[:topk]
        assoc[a, :len(top)] = [b for _, b in top]
    np.save(cache, assoc)
    return assoc
