import argparse, glob, json, math, os, re, sys, time, unicodedata
_ap = argparse.ArgumentParser()
_ap.add_argument("--meco", required=True); _ap.add_argument("--base", default=".")
_ap.add_argument("--inspect", action="store_true")
_ap.add_argument("--min_readers", type=int, default=5)
ARGS = _ap.parse_args()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

from mtp_common import *
LANG = "hindi"; PROFILE = LANG_PROFILES[LANG]; SPEC = SPECS["devanagari"]
DEV_WORD = SPEC.word_re
OUT = os.path.join(os.path.abspath(ARGS.base), "meco"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ============================================================================
# 1. Load MECO word-level data
# ============================================================================
COLS = dict(
    subject=["uniform_id", "subid", "subject", "participant", "id"],
    sample=["lang", "language", "site", "sample", "lang_site"],
    text=["trialid", "itemid", "text_id", "trial", "item"],
    word_idx=["ianum", "ia_id", "wordnum", "word_id", "ia.num"],
    word=["ia", "word", "ia_label", "ia.label"],
    first_fix=["firstfix.dur", "firstfix_dur", "ffd"],
    gaze=["firstrun.dur", "firstrun_dur", "gaze.dur", "gaze_dur", "gd"],
    total=["dur", "total.dur", "total_dur", "trt", "dwell"],
    skip=["skip", "firstrun.skip"],
    sentence=["sentnum", "sentence", "sent", "sent_id"],
)

def _read_any(path):
    if path.endswith((".rda", ".rdata", ".RData", ".rds")):
        import pyreadr
        return list(pyreadr.read_r(path).items())
    if path.endswith(".csv"):
        return [(os.path.basename(path), pd.read_csv(path, low_memory=False))]
    return []

def _pick(df, key):
    for c in COLS[key]:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in COLS[key]:
        if c.lower() in low: return low[c.lower()]
    return None

def _is_hindi(v):
    s = str(v).strip().lower()
    return s in ("hi", "hindi") or s.startswith("hi_") or s.startswith("hindi")

def load_meco(folder, verbose=True):
    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                   if f.lower().endswith((".rda", ".rdata", ".rds", ".csv")))
    if not files: raise FileNotFoundError(f"no .rda/.csv files under {folder}")
    found = []
    for f in files:
        try:
            for name, df in _read_any(f):
                if not isinstance(df, pd.DataFrame) or df.empty: continue
                m = {k: _pick(df, k) for k in COLS}
                if verbose:
                    print(f"\n{f} :: object {name!r}  rows={len(df):,}  cols={len(df.columns)}")
                    print("   columns:", list(df.columns)[:40])
                    print("   matched:", {k: v for k, v in m.items()})
                if m["text"] and m["word_idx"] and m["word"] and (m["gaze"] or m["total"]):
                    found.append((f, name, df, m))
        except Exception as e:
            print(f"   could not read {f}: {type(e).__name__}: {e}")
    if not found:
        raise RuntimeError("no word-level eye-tracking table found (need text, word index, word and a duration column)")
    rows = []
    for f, name, df, m in found:
        hi = None
        for c in [m["sample"], m["subject"]] + [c for c in df.columns if df[c].dtype == object]:
            if c is None: continue
            mask = df[c].map(_is_hindi) if c != m["subject"] else df[c].astype(str).str.lower().str.startswith("hi_")
            if mask.any(): hi = mask; hcol = c; break
        if hi is None: continue
        d = df[hi].copy(); d["__source"] = f"{os.path.basename(f)}::{name}"; d["__hcol"] = hcol
        d = d.rename(columns={v: k for k, v in m.items() if v})
        rows.append(d)
        if verbose: print(f"   -> {hi.sum():,} Hindi rows (detected in column {hcol!r})")
    if not rows: raise RuntimeError("found eye-tracking tables but no Hindi rows; check the 'matched' lines above")
    d = pd.concat(rows, ignore_index=True)
    if "sample" not in d: d["sample"] = d["__hcol"]
    if "subject" not in d: raise RuntimeError("no participant column found")
    d["subject"] = d["subject"].astype(str) + "@" + d["sample"].astype(str)
    d["word"] = d["word"].astype(str).map(lambda s: unicodedata.normalize("NFC", s.strip()))
    for k in ["first_fix", "gaze", "total", "skip", "word_idx", "sentence"]:
        if k in d: d[k] = pd.to_numeric(d[k], errors="coerce")
    d["text"] = d["text"].astype(str)
    return d

def build_passages(d):
    """One passage per text: words in interest-area order (mode over readers), joined by spaces."""
    P = {}
    for t, g in d.groupby("text"):
        ia = (g.groupby("word_idx").word.agg(lambda x: x.value_counts().index[0]).sort_index())
        words = [w for w in ia.values if w and w != "nan"]
        idx = [i for i, w in zip(ia.index, ia.values) if w and w != "nan"]
        text, spans = "", []
        for w in words:
            if text: text += " "
            spans.append((len(text), len(text) + len(w))); text += w
        P[t] = dict(text=text, word_idx=idx, words=words, spans=spans)
    return P

d = load_meco(ARGS.meco, verbose=True)
P = build_passages(d)
print(f"\nHindi MECO: {d.subject.nunique()} readers, {len(P)} passages, "
      f"{sum(len(p['words']) for p in P.values())} words; samples: {sorted(d['sample'].astype(str).unique())[:6]}")
for t in list(P)[:2]:
    print(f"   passage {t}: {P[t]['text'][:160]}…")
if ARGS.inspect:
    print("\nINSPECT_DONE"); sys.exit(0)

# ============================================================================
# 2. Human reading measures per word (mean over readers; durations over fixated readers)
# ============================================================================
def _agg(g):
    out = dict(n_readers=g.subject.nunique())
    if "skip" in g: out["skip_rate"] = g.skip.mean()
    for k in ["first_fix", "gaze", "total"]:
        if k in g:
            v = g[k][(g[k] > 0) & g[k].notna()]
            out[f"{k}_ms"] = v.mean() if len(v) else np.nan; out[f"n_{k}"] = len(v)
    return pd.Series(out)
H = d.groupby(["text", "word_idx"]).apply(_agg).reset_index()
# v8.4: the same measures within each Hindi sample, for an independent replication
SAMPLES = sorted(d["sample"].astype(str).unique())
for smp in SAMPLES:
    hs = d[d["sample"].astype(str) == smp].groupby(["text", "word_idx"]).apply(_agg).reset_index()
    keepc = [c for c in hs.columns if c.endswith("_ms") or c.startswith("n_")]
    H = H.merge(hs[["text", "word_idx"] + keepc].rename(columns={c: f"{c}@{smp}" for c in keepc}),
                on=["text", "word_idx"], how="left")
MEASURES = [m for m in ["gaze_ms", "total_ms", "first_fix_ms"] if m in H]
print("human measures:", MEASURES)

# split-half reliability between the two Hindi samples (validity check of the human data)
rel = {}
samples = sorted(d["sample"].astype(str).unique())
if len(samples) >= 2:
    for m in [x[:-3] for x in MEASURES]:
        a = d[d["sample"].astype(str) == samples[0]].query(f"{m} > 0").groupby(["text", "word_idx"])[m].mean()
        b = d[d["sample"].astype(str) == samples[1]].query(f"{m} > 0").groupby(["text", "word_idx"])[m].mean()
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        rel[m] = dict(r=float(j.corr().iloc[0, 1]), n=len(j), samples=samples[:2])
print("between-sample reliability:", {k: round(v["r"], 3) for k, v in rel.items()})

# ============================================================================
# 3. Model surprisal per word
# ============================================================================
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_pipeline.py"), encoding="utf-8").read()
exec(_src[_src.index("# ===================== scorer"):_src.index("# ===================== per-text measurements")])

def char_surprisal(surr, offs, text):
    """Same spreading rule as measure_text: trimmed token spans, whitespace tokens carry forward."""
    cs, carry = np.zeros(len(text)), 0.0
    for s, (a, b) in zip(surr, offs):
        if b <= a: continue
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b - 1].isspace(): b -= 1
        if b == a: carry += s; continue
        cs[a:b] += (s + carry) / (b - a); carry = 0.0
    return cs

FREQ = None
try:
    from datasets import load_dataset
    from collections import Counter
    _tr = xlsum_split("hindi", "train")                    
    FREQ = Counter()
    for a in _tr["text"]: FREQ.update(DEV_WORD.findall(unicodedata.normalize("NFC", a)))
    print(f"frequency table: {len(FREQ):,} types (XL-Sum Hindi train)")
except Exception as e:
    print("frequency table unavailable:", e)

rows = []
for t, p in P.items():
    text = p["text"]
    for (a, b), w, wi in zip(p["spans"], p["words"], p["word_idx"]):
        dev = DEV_WORD.findall(w); core = max(dev, key=len) if dev else ""
        rows.append(dict(text=t, word_idx=wi, ia=w, word=core, n_char=sum(map(len, dev)),
                         n_aks=sum(len(SPEC.spans(x)) for x in dev), has_dev=bool(dev),
                         has_mark=bool(SPEC.mark_re.search(w)), has_punct=w != "".join(dev),
                         logfreq=np.log1p(FREQ.get(core, 0)) if FREQ else np.nan, a=a, b=b))
W = pd.DataFrame(rows)
W["is_func"] = W.word.isin(set(PROFILE["func"] or ()))

if "sentence" in d:                                      # MECO's own sentence numbers
    sid = d.groupby(["text", "word_idx"]).sentence.agg(lambda x: x.value_counts().index[0]).rename("sent_id")
    W = W.merge(sid.reset_index(), on=["text", "word_idx"], how="left")
else:                                                    # fallback: sentence ends at । . ? !
    end = W.ia.astype(str).str.contains(r'[।.?!]["\')\]]*$')
    W["sent_id"] = end.groupby(W.text).transform(lambda x: x.shift(1, fill_value=False).cumsum())
W = W.sort_values(["text", "word_idx"]).reset_index(drop=True)
W["pos_in_sent"] = W.groupby(["text", "sent_id"]).cumcount() + 1
W["sent_final"] = (W.groupby(["text", "sent_id"]).word_idx.transform("max") == W.word_idx).astype(float)
W["log_pos_sent"] = np.log(W.pos_in_sent); W["log_pos_text"] = np.log(W.groupby("text").cumcount() + 1)
W["func"] = W.is_func.astype(float); W["punct"] = W.has_punct.astype(float)
WORD_FEATS = ["logfreq", "prev_logfreq", "func", "punct", "sent_final", "log_pos_sent", "log_pos_text"]


FONT_URL = ("https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansDevanagari/"
            "hinted/ttf/NotoSansDevanagari-Regular.ttf")
def word_widths(strings):
    try:
        import uharfbuzz as hb, urllib.request
        fp = os.path.join(OUT, "NotoSansDevanagari-Regular.ttf")
        if not os.path.exists(fp): urllib.request.urlretrieve(FONT_URL, fp)
        face = hb.Face(hb.Blob.from_file_path(fp)); font = hb.Font(face)
        def _w(x):
            buf = hb.Buffer(); buf.add_str(x); buf.guess_segment_properties(); hb.shape(font, buf, {})
            return sum(g.x_advance for g in buf.glyph_positions) / face.upem
        return [_w(x) for x in strings]
    except Exception as e:
        print(f"visual width unavailable ({type(e).__name__}: {e}); width controls skipped"); return None
_wd = word_widths(W.ia.astype(str).tolist())
HAS_WIDTH = _wd is not None
if HAS_WIDTH:
    W["width"] = _wd
    print(f"visual width: mean {W.width.mean():.2f} em; correlation with aksharas "
          f"{np.corrcoef(W.width, W.n_aks)[0, 1]:.2f}, with characters {np.corrcoef(W.width, W.n_char)[0, 1]:.2f}")

MODELS = tuple(PROFILE["models"]) + (tuple(PROFILE.get("extra", ())) if os.environ.get("MTP_EXTRA") == "1" else ())
for mid in MODELS:                                    # v9: extra scorers (XGLM-564M, Sarvam-1) when enabled
    tag = mid.split("/")[-1]
    try:
        sc = ReferenceScorer(mid, device=DEVICE, revision=PROFILE["revisions"].get(mid))
    except Exception as e:
        print(f"scorer {mid} unavailable: {e}"); continue
    S_c, S_r, NT, CR, TR = [], [], [], [], []
    for t, p in P.items():
        text = p["text"]
        sur, offs, bow, isb = sc.token_surprisals(text, return_bow=True)
        n_text_tok = len(sc.tokenizer(text, add_special_tokens=False).input_ids)
        cs_c = char_surprisal(bow_adjust(sur, bow, isb), offs, text)
        cs_r = char_surprisal(sur, offs, text)
        last = offs[-1][1] if offs else 0
        edges = {e for x, y in offs for e in (x, y)}
        for (a, b), w in zip(p["spans"], p["words"]):
            ok = b <= last
            S_c.append(cs_c[a:b].sum() if ok else np.nan); S_r.append(cs_r[a:b].sum() if ok else np.nan)
            m = max(DEV_WORD.finditer(w), key=lambda m: m.end() - m.start(), default=None)
            if m:
                wa, wb = a + m.start(), a + m.end()
                NT.append(sum(1 for x, y in offs if min(y, wb) - max(x, wa) > 0))
                CR.append(any(wa + x < e < wa + y for x, y in SPEC.spans(m.group()) for e in edges))
            else:
                NT.append(np.nan); CR.append(False)
            TR.append(len(sur) < n_text_tok)
    W[f"S_{tag}"] = S_c; W[f"Sraw_{tag}"] = S_r; W[f"ntok_{tag}"] = NT; W[f"cross_{tag}"] = CR
    print(f"{tag}: scored {np.isfinite(W[f'S_{tag}']).sum()} words"
          + ("  (WARNING: some passages were truncated at max length)" if any(TR) else ""))
    del sc; torch.cuda.empty_cache() if torch.cuda.is_available() else None
TAGS = [c[2:] for c in W.columns if c.startswith("S_")]

W = W.merge(H, on=["text", "word_idx"], how="left")
W = W.sort_values(["text", "word_idx"]).reset_index(drop=True)
for c in ["logfreq", "n_aks", "n_char"] + (["width"] if HAS_WIDTH else []) + [f"S_{t}" for t in TAGS] + [f"Sraw_{t}" for t in TAGS] + [f"ntok_{t}" for t in TAGS]:
    W[f"prev_{c}"] = W.groupby("text")[c].shift(1)
first = W.groupby("text").cumcount() == 0
keep = (~first & W.has_dev & (W.n_aks > 0) & W.logfreq.notna()
        & W[[f"S_{t}" for t in TAGS] + [f"prev_S_{t}" for t in TAGS]].notna().all(axis=1))
W["analysed"] = keep
W.to_csv(os.path.join(OUT, "meco_words.csv"), index=False, encoding="utf-8-sig")
print(f"analysed words: {keep.sum()} of {len(W)}")

# ============================================================================
# 4. Statistics helpers: leave-one-passage-out Gaussian log-likelihood per word
# ============================================================================
def cv_ll(df, y, X):
    """Held-out log-likelihood of each word (nats), fitting on the other passages."""
    out = np.full(len(df), np.nan)
    Xa = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in X]); ya = df[y].to_numpy(float)
    texts = df.text.to_numpy()
    for t in np.unique(texts):
        tr, te = texts != t, texts == t
        b, *_ = np.linalg.lstsq(Xa[tr], ya[tr], rcond=None)
        s2 = np.mean((ya[tr] - Xa[tr] @ b) ** 2)
        out[te] = -0.5 * np.log(2 * np.pi * s2) - (ya[te] - Xa[te] @ b) ** 2 / (2 * s2)
    return out

def cluster_boot(v, texts, n_boot=2000):
    """v9: bootstrap of a per-word mean that resamples PASSAGES (words within a passage are not independent)."""
    g = pd.Series(np.asarray(v, float)).groupby(np.asarray(texts)).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    k = rng.integers(0, len(g), (n_boot, len(g)))
    return s[k].sum(1) / c[k].sum(1)

def delta(df, y, X0, X1, n_boot=2000):
    """Mean per-word gain in held-out log-lik of X1 over X0 (x1000), bootstrap CI over passages (v9; the
    word-level CI is kept as ci_lo_word/ci_hi_word), and the number of passages where the gain is positive."""
    dll = cv_ll(df, y, X1) - cv_ll(df, y, X0)
    bw = np.array([dll[rng.integers(0, len(dll), len(dll))].mean() for _ in range(n_boot)])
    bs = cluster_boot(dll, df.text.to_numpy(), n_boot)
    per_text = pd.Series(dll).groupby(df.text.to_numpy()).mean()
    return dict(dll_x1000=1000 * dll.mean(), ci_lo=1000 * np.percentile(bs, 2.5), ci_hi=1000 * np.percentile(bs, 97.5),
                ci_lo_word=1000 * np.percentile(bw, 2.5), ci_hi_word=1000 * np.percentile(bw, 97.5),
                p_boot=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                texts_positive=int((per_text > 0).sum()), n_texts=len(per_text), n_words=len(dll))

def fit_pred(train, test, y, X):
    Xa = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in X])
    b, *_ = np.linalg.lstsq(Xa, train[y].to_numpy(float), rcond=None)
    Xt = np.column_stack([np.ones(len(test))] + [test[c].to_numpy(float) for c in X])
    return Xt @ b

A = W[W.analysed].copy()
for t in TAGS:
    A[f"len_tok_{t}"] = A[f"ntok_{t}"].fillna(0); A[f"prev_len_tok_{t}"] = A[f"prev_ntok_{t}"].fillna(0)
def _lxf(u):                                    # u, prev_u, u*freq, prev_u*prev_freq
    A[f"{u}_xf"] = A[u] * A.logfreq; A[f"prev_{u}_xf"] = A[f"prev_{u}"] * A.prev_logfreq
    return [u, f"prev_{u}", f"{u}_xf", f"prev_{u}_xf"]
LEN = {"character": _lxf("n_char"), "akshara": _lxf("n_aks")}
if HAS_WIDTH: LEN["visual width"] = _lxf("width")
LEN.update({f"token ({t})": _lxf(f"len_tok_{t}") for t in TAGS})
res = {}

rows = []
for y in MEASURES:
    D = A[A[f"n_{y[:-3]}"] >= ARGS.min_readers] if f"n_{y[:-3]}" in A else A
    base = ["logfreq", "prev_logfreq"]
    for u, cols in LEN.items():
        rows.append(dict(measure=y, formula="length x frequency", comparison=f"{u} length over frequency",
                         **delta(D, y, base, base + cols)))
    C_, K_ = LEN["character"], LEN["akshara"]
    rows.append(dict(measure=y, formula="length x frequency", comparison="akshara over character (both in model)",
                     **delta(D, y, base + C_, base + C_ + K_)))
    rows.append(dict(measure=y, formula="length x frequency", comparison="character over akshara (both in model)",
                     **delta(D, y, base + K_, base + K_ + C_)))
    # sensitivity: the same two comparisons without the interaction terms
    rows.append(dict(measure=y, formula="linear", comparison="akshara over character (both in model)",
                     **delta(D, y, base + C_[:2], base + C_[:2] + K_[:2])))
    rows.append(dict(measure=y, formula="linear", comparison="character over akshara (both in model)",
                     **delta(D, y, base + K_[:2], base + K_[:2] + C_[:2])))
T_ruler = pd.DataFrame(rows); T_ruler.to_csv(os.path.join(OUT, "meco_ruler_ll.csv"), index=False)
print("\nA1. Native ruler: held-out log-likelihood gain per word (x1000):")
print(T_ruler[["measure", "formula", "comparison", "dll_x1000", "ci_lo", "ci_hi", "texts_positive", "n_texts"]].round(2).to_string(index=False))

rows = []
for y in MEASURES:
    D = A[A[f"n_{y[:-3]}"] >= ARGS.min_readers] if f"n_{y[:-3]}" in A else A
    rows.append(dict(measure=y, comparison="all word-level features over intercept", **delta(D, y, [], WORD_FEATS)))
    for f in WORD_FEATS:                                   # drop-one: unique contribution of each feature
        rest = [c for c in WORD_FEATS if c != f]
        rows.append(dict(measure=y, comparison=f"word feature: {f} (drop-one)",
                         **delta(D, y, rest + LEN["akshara"], WORD_FEATS + LEN["akshara"])))
    for u, cols in LEN.items():
        rows.append(dict(measure=y, comparison=f"{u} length over word-level features",
                         **delta(D, y, WORD_FEATS, WORD_FEATS + cols)))
    C_, K_ = LEN["character"], LEN["akshara"]
    if HAS_WIDTH:
        V_ = LEN["visual width"]
        rows.append(dict(measure=y, comparison="akshara over word-level features + visual width",
                         **delta(D, y, WORD_FEATS + V_, WORD_FEATS + V_ + K_)))
        rows.append(dict(measure=y, comparison="visual width over word-level features + akshara",
                         **delta(D, y, WORD_FEATS + K_, WORD_FEATS + K_ + V_)))
        rows.append(dict(measure=y, comparison="akshara over word-level features + visual width + character",
                         **delta(D, y, WORD_FEATS + V_ + C_, WORD_FEATS + V_ + C_ + K_)))
    rows.append(dict(measure=y, comparison="akshara over word-level features + character",
                     **delta(D, y, WORD_FEATS + C_, WORD_FEATS + C_ + K_)))
    rows.append(dict(measure=y, comparison="character over word-level features + akshara",
                     **delta(D, y, WORD_FEATS + K_, WORD_FEATS + K_ + C_)))
    for t in TAGS:                                         # word surprisal vs surprisal per akshara (density)
        D2 = D.assign(**{f"dens_{t}": D[f"S_{t}"] / D.n_aks,
                         f"prev_dens_{t}": D[f"prev_S_{t}"] / D.prev_n_aks.where(D.prev_n_aks > 0)}).dropna(subset=[f"prev_dens_{t}"])
        rows.append(dict(measure=y, comparison=f"word surprisal ({t}) over word features + akshara length",
                         **delta(D2, y, WORD_FEATS + K_, WORD_FEATS + K_ + [f"S_{t}", f"prev_S_{t}"])))
        rows.append(dict(measure=y, comparison=f"surprisal per akshara ({t}) over word features + akshara length",
                         **delta(D2, y, WORD_FEATS + K_, WORD_FEATS + K_ + [f"dens_{t}", f"prev_dens_{t}"])))
T_word = pd.DataFrame(rows); T_word.to_csv(os.path.join(OUT, "meco_word_features_ll.csv"), index=False)

rows = []
for smp in (SAMPLES if len(SAMPLES) > 1 else []):
    for y in MEASURES:
        ys, ns = f"{y}@{smp}", f"n_{y[:-3]}@{smp}"
        if ys not in A: continue
        D = A[A[ns] >= ARGS.min_readers].dropna(subset=[ys])
        C_, K_ = LEN["character"], LEN["akshara"]
        tests = [("akshara over word-level features + character", WORD_FEATS + C_, WORD_FEATS + C_ + K_),
                 ("character over word-level features + akshara", WORD_FEATS + K_, WORD_FEATS + K_ + C_)]
        if HAS_WIDTH:
            V_ = LEN["visual width"]
            tests.append(("akshara over word-level features + visual width", WORD_FEATS + V_, WORD_FEATS + V_ + K_))
        tests += [(f"word surprisal ({t}) over word features + akshara length", WORD_FEATS + K_,
                   WORD_FEATS + K_ + [f"S_{t}", f"prev_S_{t}"]) for t in TAGS]
        for name, x0, x1 in tests:
            rows.append(dict(sample=smp, n_readers=int(d[d["sample"].astype(str) == smp].subject.nunique()),
                             measure=y, comparison=name, **delta(D, ys, x0, x1)))
T_samp = pd.DataFrame(rows)
if len(T_samp):
    T_samp.to_csv(os.path.join(OUT, "meco_by_sample.csv"), index=False)
    print("\nA1c. Replication within each sample (x1000):")
    print(T_samp[["sample", "measure", "comparison", "dll_x1000", "ci_lo", "ci_hi", "texts_positive"]].round(1).to_string(index=False))
print("\nA1b. Word-level features (held-out log-lik gain per word x1000):")
print(T_word[["measure", "comparison", "dll_x1000", "ci_lo", "ci_hi", "texts_positive"]].round(1).to_string(index=False))


def _cv(x): x = np.asarray(x, float); return x.std() / x.mean() if len(x) > 2 and x.mean() > 0 else np.nan
ok_words = set(zip(A.text, A.word_idx))
E = []

tokcols = [f"ntok_{t}" for t in TAGS]
dd = d[[k in ok_words for k in zip(d.text, d.word_idx)]].merge(
    W[["text", "word_idx", "n_aks", "n_char"] + tokcols], on=["text", "word_idx"])
for (s, t), g in dd.groupby(["subject", "text"]):
    m = "gaze" if "gaze" in g else "total"
    g = g[g[m] > 0]
    if len(g) < 10: continue
    e = dict(subject=s, text=t, n=len(g), cv_word=_cv(g[m]), cv_char=_cv(g[m] / g.n_char), cv_aks=_cv(g[m] / g.n_aks))
    for tt in TAGS: e[f"cv_tok_{tt}"] = _cv(g[m] / g[f"ntok_{tt}"].clip(lower=1))
    E.append(e)
E = pd.DataFrame(E)
LR = {"lr_word_vs_aks": "cv_word", "lr_char_vs_aks": "cv_char"}
LR.update({f"lr_tok_{tt}_vs_aks": f"cv_tok_{tt}" for tt in TAGS})
for k, c in LR.items(): E[k] = np.log(E[c] / E.cv_aks)
subj = E.groupby("subject")[list(LR)].mean()
even = {}
from scipy import stats
for c in LR:
    v = subj[c].dropna().to_numpy(); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    even[c] = dict(mean=float(v.mean()), pct=float((np.exp(v.mean()) - 1) * 100), ci_lo=float(np.percentile(bs, 2.5)),
                   ci_hi=float(np.percentile(bs, 97.5)), p_wilcoxon=float(stats.wilcoxon(v).pvalue),
                   frac_readers_positive=float((v > 0).mean()), n_readers=len(v))
E.to_csv(os.path.join(OUT, "meco_evenness_by_reader_text.csv"), index=False)
print("\nA2. Reading-time evenness (log CV ratio of each unit vs akshara, > 0 = per-akshara more even), per reader:")
for k, v in even.items(): print(f"   {k}: {v['pct']:+.1f}%  [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  p={v['p_wilcoxon']:.2g}  "
                                f"readers positive {v['frac_readers_positive']:.0%} (n={v['n_readers']})")


rows = []
for y in MEASURES:
    D = A[A[f"n_{y[:-3]}"] >= ARGS.min_readers] if f"n_{y[:-3]}" in A else A
    base = ["logfreq", "prev_logfreq"] + LEN["akshara"]
    for t in TAGS:
        for kind, col in [("corrected", "S"), ("raw", "Sraw")]:
            rows.append(dict(measure=y, model=t, surprisal=kind,
                             **delta(D, y, base, base + [f"{col}_{t}", f"prev_{col}_{t}"])))
        rows.append(dict(measure=y, model=t, surprisal="corrected minus raw",
                         **delta(D, y, base + [f"Sraw_{t}", f"prev_Sraw_{t}"], base + [f"S_{t}", f"prev_S_{t}"])))
T_pp = pd.DataFrame(rows); T_pp.to_csv(os.path.join(OUT, "meco_surprisal_ll.csv"), index=False)
print("\nB. Surprisal predictive power: held-out log-lik gain over length + frequency (x1000):")
print(T_pp[["measure", "model", "surprisal", "dll_x1000", "ci_lo", "ci_hi", "texts_positive"]].round(2).to_string(index=False))

rows = []
for y in MEASURES:
    D = A[A[f"n_{y[:-3]}"] >= ARGS.min_readers] if f"n_{y[:-3]}" in A else A
    for t in TAGS:
        X = ["logfreq", "prev_logfreq"] + LEN["akshara"] + [f"S_{t}", f"prev_S_{t}"]
        split = (D[f"ntok_{t}"] > 1).to_numpy(); cross = D[f"cross_{t}"].astype(bool).to_numpy()
        cats = {"unsplit": ~split, "split, akshara-aligned": split & ~cross, "split, crosses akshara": split & cross}
        for fit in ["UNSPLIT", "FULL"]:
            pred = np.full(len(D), np.nan)
            for tx in D.text.unique():                         # leave-one-passage-out
                tr = (D.text != tx).to_numpy() & ((~split) if fit == "UNSPLIT" else True); te = (D.text == tx).to_numpy()
                if tr.sum() > len(X) + 5: pred[te] = fit_pred(D[tr], D[te], y, X)
            resid = D[y].to_numpy(float) - pred
            for cat, m in cats.items():
                v = resid[m & np.isfinite(resid)]
                if len(v) < 10: continue
                bs = cluster_boot(v, D.text.to_numpy()[m & np.isfinite(resid)])      # v9: over passages
                rows.append(dict(measure=y, model=t, fit=fit, category=cat, n=len(v), share=m.mean(),
                                 mean_resid_ms=v.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
T_split = pd.DataFrame(rows); T_split.to_csv(os.path.join(OUT, "meco_split_residuals.csv"), index=False)
print("\nC. Residual reading time (observed - predicted, ms) by split status:")
print(T_split[T_split.measure == MEASURES[0]].round(2).to_string(index=False))

# ============================================================================
# 5. Figures and findings
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
r = T_ruler[(T_ruler.measure == MEASURES[0]) & (T_ruler.formula == "length x frequency")
            & T_ruler.comparison.str.endswith("over frequency")]
axes[0].barh(r.comparison.str.replace(" length over frequency", ""), r.dll_x1000,
             xerr=[r.dll_x1000 - r.ci_lo, r.ci_hi - r.dll_x1000], color="C0")
axes[0].set_title(f"A. Which length unit predicts\n{MEASURES[0].replace('_ms', '')} duration?", fontsize=10)
axes[0].set_xlabel("held-out log-lik gain per word (x1000)")
b = T_pp[(T_pp.measure == MEASURES[0]) & T_pp.surprisal.isin(["corrected", "raw"])]
for i, kind in enumerate(["raw", "corrected"]):
    s = b[b.surprisal == kind]
    axes[1].bar(np.arange(len(s)) + (i - 0.5) * 0.38, s.dll_x1000, 0.38, yerr=[s.dll_x1000 - s.ci_lo, s.ci_hi - s.dll_x1000],
                label=kind, color=["C7", "C1"][i])
axes[1].set_xticks(range(len(TAGS))); axes[1].set_xticklabels(TAGS); axes[1].legend(fontsize=8)
axes[1].set_title("B. Surprisal beyond length + frequency", fontsize=10); axes[1].set_ylabel("log-lik gain per word (x1000)")
c = T_split[(T_split.measure == MEASURES[0]) & (T_split.fit == "UNSPLIT")]
cats = ["unsplit", "split, akshara-aligned", "split, crosses akshara"]
for j, cat in enumerate(cats):
    s = c[c.category == cat].set_index("model").reindex(TAGS)
    axes[2].errorbar(np.arange(len(TAGS)) + (j - 1) * 0.2, s.mean_resid_ms, yerr=[s.mean_resid_ms - s.ci_lo, s.ci_hi - s.mean_resid_ms],
                     fmt="o", capsize=3, label=cat)
axes[2].axhline(0, color="grey", lw=0.8); axes[2].set_xticks(range(len(TAGS))); axes[2].set_xticklabels(TAGS)
axes[2].set_title("C. Residual reading time (UNSPLIT fit)", fontsize=10); axes[2].set_ylabel("ms"); axes[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_meco.png"), dpi=200); plt.close(fig)

y0 = MEASURES[0]
def _g(u):
    r_ = T_ruler[(T_ruler.measure == y0) & (T_ruler.formula == "length x frequency")
                 & (T_ruler.comparison == f"{u} length over frequency")]
    if not len(r_): return "—"
    r_ = r_.iloc[0]; return f"{r_.dll_x1000:+.1f} [{r_.ci_lo:+.1f}, {r_.ci_hi:+.1f}] ({r_.texts_positive}/{r_.n_texts})"
def _e(u):
    if u == "akshara": return "reference"
    k = "lr_char_vs_aks" if u == "character" else f"lr_tok_{u[7:-1]}_vs_aks"
    v = even.get(k); return f"{v['pct']:+.1f}% (p = {v['p_wilcoxon']:.2g})" if v else "—"
def _row(tab, **kw):
    m = tab.copy()
    for k, v in kw.items(): m = m[m[k] == v]
    return m.iloc[0] if len(m) else None
ra = _row(T_ruler, measure=y0, formula="length x frequency", comparison="akshara over character (both in model)")
rc = _row(T_ruler, measure=y0, formula="length x frequency", comparison="character over akshara (both in model)")
la = _row(T_ruler, measure=y0, formula="linear", comparison="akshara over character (both in model)")
lc = _row(T_ruler, measure=y0, formula="linear", comparison="character over akshara (both in model)")
md = [f"# MECO Hindi: reading-time analysis", "",
      f"Readers: {d.subject.nunique()} ({', '.join(samples)}), passages: {len(P)}, analysed words: {len(A)}. "
      f"Main measure: {y0.replace('_ms', '')} duration. Between-sample reliability: "
      + ", ".join(f"{k} r = {v['r']:.2f}" for k, v in rel.items()) + ".", "",
      "## A. Native ruler check", "",
      f"* Unique gain of akshara length over character length: {ra.dll_x1000:+.2f} [{ra.ci_lo:+.2f}, {ra.ci_hi:+.2f}] "
      f"(positive in {ra.texts_positive}/{ra.n_texts} passages); character over akshara: {rc.dll_x1000:+.2f} "
      f"[{rc.ci_lo:+.2f}, {rc.ci_hi:+.2f}] (positive in {rc.texts_positive}/{rc.n_texts}). Regression: Nguyen & Arehalli's "
      f"length x frequency formula. Without the interaction terms: akshara {la.dll_x1000:+.2f} vs character {lc.dll_x1000:+.2f}.",
      "", "### All units side by side (word = whole word, no length information)", "",
      "| unit | gain over a word-only model (frequency), x1000 | reading time per unit: CV vs akshara (> 0 = akshara more even) |",
      "|---|---|---|",
      f"| word | 0 (reference) | {even['lr_word_vs_aks']['pct']:+.1f}% (p = {even['lr_word_vs_aks']['p_wilcoxon']:.2g}) |"] + [
      f"| {u} | {_g(u)} | {_e(u)} |" for u in ["character", "akshara"] + [f"token ({t})" for t in TAGS]] + ["",
      "### Word-level features (frequency, previous-word frequency, function word, punctuation, sentence-final, "
      "position in sentence, position in passage)", "",
      "| comparison | gain x1000 [95% CI] | passages positive |", "|---|---|---|"] + [
      f"| {r_.comparison} | {r_.dll_x1000:+.1f} [{r_.ci_lo:+.1f}, {r_.ci_hi:+.1f}] | {r_.texts_positive}/{r_.n_texts} |"
      for r_ in T_word[T_word.measure == y0].itertuples()] + (["",
      "### Replication within each Hindi sample (independent readers)", "",
      "| sample | readers | comparison | gain x1000 [95% CI] | passages positive |", "|---|---|---|---|---|"] + [
      f"| {r_.sample} | {r_.n_readers} | {r_.comparison} | {r_.dll_x1000:+.1f} [{r_.ci_lo:+.1f}, {r_.ci_hi:+.1f}] | "
      f"{r_.texts_positive}/{r_.n_texts} |" for r_ in T_samp[T_samp.measure == y0].itertuples()]
      if len(T_samp) else []) + ["",
      "## B. Surprisal predictive power (held-out log-lik gain per word x1000, over length + frequency)", "",
      "| model | tokenizer | raw | corrected | corrected − raw |", "|---|---|---|---|---|"]
for t in TAGS:
    g = {k: _row(T_pp, measure=y0, model=t, surprisal=k) for k in ["raw", "corrected", "corrected minus raw"]}
    md.append(f"| {t} | {'rule (cuts marks)' if t in RULE_SCORERS else 'learned subwords'} | " + " | ".join(f"{g[k].dll_x1000:+.2f} [{g[k].ci_lo:+.2f}, {g[k].ci_hi:+.2f}]" for k in g) + " |")
md += ["", "## C. Split words (residual ms = observed − predicted)", "",
       "| model | fit | unsplit | split, aligned | split, crosses akshara | note |", "|---|---|---|---|---|---|"]
for t in TAGS:
    for fit in ["UNSPLIT", "FULL"]:
        cells = []
        for cat in cats:
            rr = _row(T_split, measure=y0, model=t, fit=fit, category=cat)
            cells.append(f"{rr.mean_resid_ms:+.1f} [{rr.ci_lo:+.1f}, {rr.ci_hi:+.1f}] (n={rr.n})" if rr is not None else "—")
        u = _row(T_split, measure=y0, model=t, fit=fit, category="unsplit")
        note = ("UNSPLIT fit is trained on only " + f"{u.share:.0%}" + " of words: it extrapolates, so read the FULL fit"
                if fit == "UNSPLIT" and u is not None and u.share < 0.3 else "")
        md.append(f"| {t} | {fit} | " + " | ".join(cells) + f" | {note} |")
md += ["", "Log-likelihood gains are leave-one-passage-out; CIs bootstrap over passages (v9; word-level CIs are "
       "in the CSVs as ci_lo_word/ci_hi_word). "
       "With 12 passages, 'positive in k/12 passages' shows how consistent an effect is across texts."]
open(os.path.join(OUT, "MECO_FINDINGS.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump(dict(reliability=rel, evenness=even, n_readers=int(d.subject.nunique()), n_passages=len(P),
               n_words=int(len(A)), measures=MEASURES, models=TAGS,
               versions=dict(torch=torch.__version__, transformers=transformers.__version__)),
          open(os.path.join(OUT, "meco_summary.json"), "w"), indent=1, default=str)
print("\nWrote", OUT); print("MECO_DONE")
