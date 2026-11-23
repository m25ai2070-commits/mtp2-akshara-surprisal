import argparse, os, sys, time
_ap = argparse.ArgumentParser()
_ap.add_argument("lang"); _ap.add_argument("--n", type=int, default=1000); _ap.add_argument("--base", default=".")
_ap.add_argument("--extra", type=int, default=int(os.environ.get("MTP_EXTRA", "0")))   # v9: extra scorers
ARGS = _ap.parse_args()
import matplotlib; matplotlib.use("Agg")          # figures are saved, not shown

# ===================== imports =====================
import os, json, math, shutil, hashlib, platform, datetime, warnings, re, unicodedata
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

import torch
import transformers, datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoConfig)
from datasets import load_dataset

warnings.filterwarnings("once")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
      f"datasets {datasets.__version__} | device {DEVICE}")

def tqdm(it, **kw):                                  # plain-text progress (works in a subprocess)
    it = list(it); n = len(it); step = max(1, n // 20); t0 = time.time()
    for i, x in enumerate(it, 1):
        yield x
        if i % step == 0 or i == n:
            print(f"    progress {i}/{n} ({i / n:.0%}), {time.time() - t0:,.0f}s", flush=True)

# HF_TOKEN (if needed) is read from the environment by the Hugging Face libraries.
from mtp_common import *
assert validate_all(verbose=False), "akshara segmenter failed its gold tests"
LANG = ARGS.lang
PROFILE = LANG_PROFILES[LANG]

# ===================== config =====================
from dataclasses import dataclass, field

@dataclass
class Config:
    # data
    DATASET:      str = "csebuetnlp/xlsum"
    DATASET_CFG:  str = LANG
    SPLIT:        str = "test"
    NUM_SAMPLES:  int = 1000
    SEED:         int = 42

    RUN_GENERATION: bool = False

    RQ2_MIN_WORDS:  int = 5
    N_PERM_GRAIN:   int = 100          

    SCORING_MODELS: tuple = PROFILE["models"]
   
    MODEL_REVISIONS: dict = field(default_factory=lambda: dict(PROFILE["revisions"]))

    USE_ARTICLE_SOURCE: bool = True
    ARTICLE_MIN_WORDS:  int = 25
    ARTICLE_MAX_WORDS:  int = 60

    
    SCORE_GOLD_WITH_CONTEXT: bool = False
    CONTEXT_MAX_WORDS:       int = 300

    
    FREQ_SPLIT: str = "train"

    GENERATOR_MODEL: str = "csebuetnlp/mT5_multilingual_XLSum"
    NUM_BEAMS:       int = 4
    MAX_NEW_TOKENS:  int = 84
    MIN_NEW_TOKENS:  int = 12
    NO_REPEAT_NGRAM: int = 2

    N_BOOTSTRAP: int = 5000            
    BOW_CORRECT:     bool = True
    KEEP_RAW_VARIANT: bool = True
    N_SHUFFLE_NULL:  int = 100         
    CHECKPOINT_EVERY: int = 100        
    CODE_VERSION:    str = "mtp-9"
    REF_SCORER:      str = "bloom-560m"    # cross-scorer comparisons are made against this model
    FUNC_TOPK:       int = 100             # no curated list -> the 100 most frequent train-split words
    RUN_AUDIT:       bool = True           # Part A (tokenizer audit, CPU only); set False after once

cfg = Config()
cfg.NUM_SAMPLES = ARGS.n          # set from the final notebook
if ARGS.extra:                    
    cfg.SCORING_MODELS = tuple(PROFILE["models"]) + tuple(PROFILE.get("extra", ()))   # of those stays the same
CORE_TAGS = [m.split("/")[-1] for m in PROFILE["models"]]
cfg.RUN_AUDIT = False             
BASE_DIR = os.path.abspath(ARGS.base)
RUN_DIR = os.path.join(BASE_DIR, f"run_{cfg.DATASET_CFG}")
os.makedirs(os.path.join(RUN_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "plots"), exist_ok=True)
os.chdir(RUN_DIR); print("working folder:", RUN_DIR)
import random as _random
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); _random.seed(cfg.SEED)
print(f"Config ready | task=summarization({cfg.DATASET_CFG}) | n={cfg.NUM_SAMPLES} "
      f"| generation={'on' if cfg.RUN_GENERATION else 'off (gold only)'}")

# ===================== script binding =====================
SCRIPT = LANG_SCRIPT[cfg.DATASET_CFG]; SPEC = SPECS[SCRIPT]
akshara_spans = SPEC.spans
def aksharas(word): return [word[a:b] for a, b in akshara_spans(word)]
DEV_WORD = SPEC.word_re          
_MARK = SPEC.mark_re             # any combining mark of this script
GOLD_AKSHARA = GOLD_BY_SCRIPT[SCRIPT]
print(f"\nlanguage={cfg.DATASET_CFG} script={SCRIPT} | consonants={len(SPEC.cons)} vowels={len(SPEC.vowel)} "
      f"signs={len(SPEC.dep)} modifiers={len(SPEC.mod)} virama={SPEC.virama!r} nukta={SPEC.nukta!r}")

# ===================== scorer =====================
_DEV_CHAR = SPEC.char_re          

def _bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
_BYTE_DEC = _bytes_to_unicode()

def _starts_word_bytes(piece):
    """Fallback for byte-fragment tokens (decode gives U+FFFD): look at raw bytes."""
    if not piece or any(ch not in _BYTE_DEC for ch in piece): return False
    raw = bytes(_BYTE_DEC[ch] for ch in piece)
    b0 = raw[0]
    if b0 < 0x80:  return True                    # ASCII: space, punctuation, digit, Latin
    if b0 < 0xC0:  return False                   # continuation byte: inside a character
    try:                                          # full first character available?
        n = 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
        return not _DEV_CHAR.match(raw[:n].decode("utf-8"))
    except UnicodeDecodeError:                    # truncated lead sequence
        return not (b0 == 0xE0 and (len(raw) < 2 or raw[1] in SPEC.byte2))    # script lead byte -> inside word

_SP_SPACE = "\u2581"                               # SentencePiece word-start marker
_SP_BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")    # SentencePiece byte-fallback token

def is_metaspace(tokenizer, sample=20000):
    """v9: True for SentencePiece-style vocabularies (XGLM, Sarvam-1, Llama), where a word-initial
    token carries a leading U+2581 that decoding a single token silently drops."""
    pieces = tokenizer.convert_ids_to_tokens(list(range(min(len(tokenizer), sample))))
    return np.mean([bool(p) and p.startswith(_SP_SPACE) for p in pieces]) > 0.05

def _sp_byte_starts_word(b0):
    if b0 < 0x80: return True                      
    if b0 < 0xC0: return False                     
    return b0 != 0xE0                              

def build_bow_flags(tokenizer, vocab_n, eos_ids=(), metaspace=False):
    """Boolean array over the model vocabulary: True = token begins a new word."""
    n = min(len(tokenizer), vocab_n)
    strs = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
    pieces = tokenizer.convert_ids_to_tokens(list(range(n)))
    flags = np.zeros(vocab_n, bool)
    for i, (s, p) in enumerate(zip(strs, pieces)):
        if metaspace and p and _SP_BYTE.match(p):               
            flags[i] = _sp_byte_starts_word(int(_SP_BYTE.match(p).group(1), 16))
        elif metaspace and p and p.startswith(_SP_SPACE):       
            flags[i] = True
        elif not s:               flags[i] = False
        elif s[0] == "\ufffd":    flags[i] = _starts_word_bytes(p)
        else:                     flags[i] = not _DEV_CHAR.match(s[0])
    for t in tokenizer.all_special_ids:          
        if 0 <= t < vocab_n: flags[t] = False
    for t in eos_ids:                             
        if t is not None and 0 <= t < vocab_n: flags[t] = True
    return flags

def bow_adjust(sur, bow_lp, tok_is_bow):
    """Apply the word-boundary correction per token (sum over a word's tokens gives S(w)).
    bow_lp[t] = log2 P(text token t starts a word | prefix), len = n_tokens + 1.
    A word-initial token gives back -log2 P(previous word ended); a token followed by a
    word-initial token (or the end of the text) pays -log2 P(this word ends)."""
    s = np.asarray(sur, float).copy(); n = len(s)
    if n == 0: return s
    b = np.asarray(bow_lp, float); f = np.asarray(tok_is_bow, bool)
    s[f] += b[:n][f]
    nxt = np.r_[f[1:], True]
    s[nxt] -= b[1:n + 1][nxt]
    return s


class ReferenceScorer:
    def __init__(self, model_id: str, device: str = None, revision: str = None, dtype=None):
        self.model_id = model_id
        self.device = device or DEVICE
        print(f"  loading reference scorer: {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        assert self.tokenizer.is_fast, f"{model_id}: fast tokenizer required (offset mapping)"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype or torch.float32, revision=revision).to(self.device).eval()
        mpe = getattr(self.model.config, "max_position_embeddings", None)
        npos = getattr(self.model.config, "n_positions", None)
        self.max_len = int(min([x for x in (mpe, npos, 1024) if x]))
        self.vocab_n = self.model.get_input_embeddings().num_embeddings

        self.metaspace = is_metaspace(self.tokenizer)          # v9
        special = set(self.tokenizer.all_special_ids)
        cands = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                 self.model.config.bos_token_id, self.model.config.eos_token_id]
        bos = next((c for c in cands if c is not None and c in special and 0 <= c < self.vocab_n), None)
        own = []
        if self.metaspace:   # v9: SentencePiece models get the prefix their own tokenizer adds (XGLM: </s>)
            w_, wo_ = (self.tokenizer("x", add_special_tokens=s_).input_ids for s_ in (True, False))
            own = w_[:len(w_) - len(wo_)] if w_[len(w_) - len(wo_):] == wo_ else []
        if own and max(own) < self.vocab_n:
            self.prefix_ids = list(own)
        elif bos is not None:
            self.prefix_ids = [bos]
        else:
            self.prefix_ids = self.tokenizer("\n", add_special_tokens=False).input_ids
        assert self.prefix_ids and max(self.prefix_ids) < self.vocab_n, f"{model_id}: bad prefix"
        self.revision = getattr(self.model.config, "_commit_hash", None)

        
        self.is_bow = build_bow_flags(self.tokenizer, self.vocab_n,
                                      eos_ids=(self.tokenizer.eos_token_id, self.model.config.eos_token_id),
                                      metaspace=self.metaspace)
        self.bow_ids = torch.tensor(np.flatnonzero(self.is_bow), device=self.device)
        
        ps, pw = ((PROFILE["probe_start"].strip(), PROFILE["probe_word"].strip()) if self.metaspace
                  else (PROFILE["probe_start"], PROFILE["probe_word"]))
        a = self.tokenizer(ps, add_special_tokens=False).input_ids
        b = self.tokenizer(pw, add_special_tokens=False).input_ids
        assert self.is_bow[a[0]], f"{model_id}: {ps!r} should start a word"
        assert not any(self.is_bow[i] for i in b[1:]), f"{model_id}: word-internal token flagged as word start"
        if PROFILE["danda"]:
            d = self.tokenizer("।", add_special_tokens=False).input_ids
            if not (len(d) == 1 and self.is_bow[d[0]]):
                print(f"  WARNING {model_id}: danda is not one word-start token; correction is approximate before '।'")
        print(f"  ready on {self.device} (max_len={self.max_len}, vocab={self.vocab_n}, revision={self.revision}, "
              f"sentencepiece={self.metaspace}, "
              f"prefix={self.tokenizer.convert_ids_to_tokens(self.prefix_ids)}, "
              f"word-start tokens={self.is_bow.mean():.1%})")

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    def show_tokens(self, text):
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        toks = self.tokenizer.convert_ids_to_tokens(enc.input_ids)
        return [(t, (a, b), text[a:b]) for t, (a, b) in zip(toks, enc.offset_mapping)]

    def token_surprisals(self, text, condition=None, return_bow=False):
        """Raw token surprisals (bits) + offsets. With return_bow=True also returns
        bow_lp (log2 P(next token starts a word), len n+1) and the word-start flag of
        every actual token, for bow_adjust()."""
        if not text or not text.strip():
            return ([], [], [], []) if return_bow else ([], [])
        enc = self.tokenizer(text, add_special_tokens=False,
                             return_offsets_mapping=True, return_tensors="pt")
        txt_ids, offsets = enc["input_ids"], enc["offset_mapping"][0].tolist()
        assert int(txt_ids.max()) < self.vocab_n, \
            f"{self.model_id}: tokenizer produced an id outside the model vocab"
        prefix = torch.tensor([self.prefix_ids])
        if condition and condition.strip():
            cond = self._ids(condition)
            budget = self.max_len - txt_ids.shape[1] - prefix.shape[1]
            cond = cond[:, -budget:] if budget > 0 else cond[:, :0]
            prefix = torch.cat([prefix, cond], dim=1)
        ids = torch.cat([prefix, txt_ids], dim=1)[:, : self.max_len].to(self.device)
        start = prefix.shape[1]
        with torch.no_grad():
            logp = torch.log_softmax(self.model(ids).logits[0].float(), dim=-1)
        tgt = ids[0, start:]
        sur = (-logp[start - 1 : ids.shape[1] - 1].gather(1, tgt[:, None]).squeeze(1)
               / math.log(2)).tolist()
        if not return_bow:
            return sur, offsets[: len(sur)]
        bow_lp = (torch.logsumexp(logp[start - 1 :, self.bow_ids], dim=-1) / math.log(2)).tolist()
        tok_is_bow = self.is_bow[tgt.cpu().numpy()].tolist()
        return sur, offsets[: len(sur)], bow_lp, tok_is_bow

# ===================== per-text measurements =====================
import bisect
FUNC_WORDS = set(PROFILE["func"] or ())

def _cv(a):
    a = np.asarray(a, float)
    if len(a) < 2: return float("nan")
    m = a.mean()
    return float(a.std() / m) if m > 1e-9 else float("nan")

def _lr(x, y):
    return math.log(x / y) if (x == x and y == y and x > 0 and y > 0) else float("nan")

def measure_text(surr, offs, text, drop_first=True, n_perm_grain=200, min_words=5, tok_is_bow=None,
                 n_shuffle_null=None):
    """Per-text measurements. Returns (summary_dict, per_word_df) or (None, None).
    tok_is_bow (CHANGE 12, optional): word-start flag per token; used only to flag
    words whose boundaries line up with token boundaries (bow_ok)."""
    text = unicodedata.normalize("NFC", text)
    surr = np.asarray(surr, float)
    words = [(m.group(), m.start(), m.end()) for m in DEV_WORD.finditer(text)]
    if drop_first: words = words[1:]
    if len(words) < min_words or len(surr) < 2: return None, None
    cut = words[0][1]

    char_s = np.zeros(len(text)); covered = 0.0; carry = 0.0
    for s, (a, b) in zip(surr, offs):
        if b <= a: continue
        covered += s
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b - 1].isspace(): b -= 1
        if b == a: carry += s; continue
        char_s[a:b] += (s + carry) / (b - a); carry = 0.0
    coverage = covered / surr.sum() if surr.sum() > 0 else float("nan")


    tspan = []
    for (a, b) in offs:
        if b <= a: tspan.append(None); continue
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b - 1].isspace(): b -= 1
        tspan.append((a, b))
    def _spread(sv):
        cs, cr = np.zeros(len(text)), 0.0
        for s_, sp in zip(sv, tspan):
            if sp is None: continue
            x, y = sp
            if y == x: cr += s_; continue
            cs[x:y] += (s_ + cr) / (y - x); cr = 0.0
        return cs

    
    in_word = np.zeros(len(text), bool)
    for _, a, b in words: in_word[a:b] = True
    post = surr[[i for i, (a, b) in enumerate(offs) if b > cut]] if len(offs) else surr[:0]
    mass_on_words = float(char_s[in_word].sum() / post.sum()) if post.sum() > 0 else float("nan")

    
    tok_all  = np.array([s for s, (a, b) in zip(surr, offs) if b > cut])
    tok_series = np.array([s for s, (a, b) in zip(surr, offs)
                           if b > cut and b > a and in_word[a:b].any()])

    # 2) word-level totals + three length normalisers
    rows, aks_series, word_chunks = [], [], []
    word_pos, shuf_groups = [], []      
    tok_E = sorted({e for (x, y) in offs for e in (x, y)})
    def _cuts_inside(lo, hi):
        k = bisect.bisect_right(tok_E, lo)
        return k < len(tok_E) and tok_E[k] < hi
    for wi, (w, a, b) in enumerate(words):
        ak = akshara_spans(w)
        if not ak: continue
        tix = [i for i, (x, y) in enumerate(offs) if min(y, b) - max(x, a) > 0]
        n_tok = len(tix)
        bow_ok = float("nan")
        if tok_is_bow is not None and tix:
            j, k = tix[0], tix[-1]
            xj = offs[j][0]
            while xj < offs[j][1] and text[xj].isspace(): xj += 1
            start_ok = bool(tok_is_bow[j]) and xj == a
            end_ok = offs[k][1] == b and (k + 1 >= len(tok_is_bow) or bool(tok_is_bow[k + 1]))
            bow_ok = start_ok and end_ok
        rows.append(dict(widx=wi,                                  # stable id for cross-scorer pairing
                         word=w, S=char_s[a:b].sum(), n_char=b - a,
                         n_aks=len(ak), n_tok=max(n_tok, 1), is_func=w in FUNC_WORDS,
                         has_mark=bool(_MARK.search(w)),      # CHANGE 13
                         bow_ok=bow_ok,
                         crosses_aks=any(_cuts_inside(a + x, a + y) for x, y in ak)))
        aks_series.extend(char_s[a + x : a + y].sum() for x, y in ak)
        pos = np.concatenate([np.arange(a + x, a + y) for x, y in ak])
        
        u = char_s[pos].sum() / len(pos)
        word_pos.append((pos, np.array([y - x for x, y in ak])))
        shuf_groups.append(np.array([i for i in range(len(offs)) if tspan[i] is not None
                                     and a <= tspan[i][0] < tspan[i][1] <= b], int))
        word_chunks.append((np.cumsum(np.r_[0.0, char_s[pos]]),
                            np.array([y - x for x, y in ak]),
                            np.cumsum(np.r_[0.0, np.full(len(pos), u)])))
    wdf = pd.DataFrame(rows)
    if len(wdf) == 0: return None, None
    wdf = wdf[wdf.S > 0]
    if len(wdf) < min_words: return None, None
    aks_series = np.array(aks_series)
    cv_aks_grain = _cv(aks_series)
    aks_null = np.concatenate([np.diff(cu[np.r_[0, np.cumsum(sz)]]) for _, sz, cu in word_chunks])
    cv_aks_null = _cv(aks_null)

    
    cv_perm = cv_rcut = cv_rcut_null = float("nan")
    cv_aks_sh = cv_perm_sh = cv_rcut_sh = float("nan")
    informative = np.mean([1 < len(sz) < sz.sum() for _, sz, _ in word_chunks])
    if n_perm_grain and n_perm_grain > 0:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)          
        rng2 = np.random.default_rng(seed + 1)     
        cv_p, cv_r, cv_n = [], [], []
        for _ in range(n_perm_grain):
            sp, sr, sn = [], [], []
            for csum, sizes, csum_u in word_chunks:
                n, k = int(sizes.sum()), len(sizes)
                e = np.r_[0, np.cumsum(rng.permutation(sizes))]
                sp.append(np.diff(csum[e]))
                if k > 1:
                    cuts = np.sort(rng2.choice(np.arange(1, n), k - 1, replace=False))
                    e = np.r_[0, cuts, n]
                else:
                    e = np.array([0, n])
                sr.append(np.diff(csum[e])); sn.append(np.diff(csum_u[e]))   # same cuts, null series
            cv_p.append(_cv(np.concatenate(sp))); cv_r.append(_cv(np.concatenate(sr)))
            cv_n.append(_cv(np.concatenate(sn)))
        cv_perm, cv_rcut = float(np.nanmean(cv_p)), float(np.nanmean(cv_r))
        cv_rcut_null = float(np.nanmean(cv_n))

      
        n_sh = n_perm_grain if n_shuffle_null is None else n_shuffle_null
        rng3 = np.random.default_rng(seed + 2)
        sv0 = np.asarray(surr, float)
        cv_sa, cv_sp, cv_sr = [], [], []
        for _ in range(n_sh):
            sv = sv0.copy()
            for gi in shuf_groups:
                if len(gi) > 1: sv[gi] = rng3.permutation(sv[gi])
            ch = _spread(sv); sa, sp_, sr_ = [], [], []
            for pos, sizes in word_pos:
                cs_ = np.cumsum(np.r_[0.0, ch[pos]]); n, k = int(sizes.sum()), len(sizes)
                sa.append(np.diff(cs_[np.r_[0, np.cumsum(sizes)]]))
                sp_.append(np.diff(cs_[np.r_[0, np.cumsum(rng3.permutation(sizes))]]))
                e = (np.r_[0, np.sort(rng3.choice(np.arange(1, n), k - 1, replace=False)), n]
                     if k > 1 else np.array([0, n]))
                sr_.append(np.diff(cs_[e]))
            cv_sa.append(_cv(np.concatenate(sa))); cv_sp.append(_cv(np.concatenate(sp_)))
            cv_sr.append(_cv(np.concatenate(sr_)))
        if n_sh:
            cv_aks_sh, cv_perm_sh, cv_rcut_sh = (float(np.nanmean(v)) for v in (cv_sa, cv_sp, cv_sr))

    # 4) tokenizer/akshara alignment
    tok_edges = {b for (_, b) in offs} | {a for (a, _) in offs}
    aks_bounds = [(a + x, a + y) for w, a, b in words for x, y in akshara_spans(w)]
    inside = sum(any(x < e < y for e in tok_edges) for (x, y) in aks_bounds)
    complex_b = [(x, y) for (x, y) in aks_bounds if y - x > 1]
    inside_c = sum(any(x < e < y for e in tok_edges) for (x, y) in complex_b)

    cw = wdf[~wdf.is_func]
    ok_c = len(cw) >= min_words
    out = dict(
        n_words=len(wdf), n_content_words=len(cw),
        n_tokens=len(tok_series), n_tokens_all=len(tok_all), n_aksharas=len(aks_series),
        surprisal_coverage=coverage, mass_on_words=mass_on_words,
        frac_aksharas_split_by_tokens=inside / max(1, len(aks_bounds)),
        frac_complex_aksharas_split=inside_c / max(1, len(complex_b)),
        tokens_per_akshara=len(tok_series) / max(1, len(aks_series)),
        frac_words_randcut_informative=float(informative),
        # grain (different units along the sequence)
        var_token=float(np.var(tok_series)), cv_token=_cv(tok_series), cv_token_all=_cv(tok_all),
        var_akshara=float(np.var(aks_series)), cv_akshara_grain=cv_aks_grain,
        cv_random_chunk_grain=cv_perm, cv_randcut_grain=cv_rcut,
        cv_randcut_null=cv_rcut_null, cv_akshara_null=cv_aks_null,
        cv_akshara_shufnull=cv_aks_sh, cv_random_chunk_shufnull=cv_perm_sh, cv_randcut_shufnull=cv_rcut_sh,
        frac_words_bow_ok=float(np.nanmean(wdf.bow_ok.astype(float))) if tok_is_bow is not None else float("nan"),
        # same (word) grain, different rate units: all words
        cv_word_total=_cv(wdf.S),
        cv_per_char=_cv(wdf.S / wdf.n_char),
        cv_per_token=_cv(wdf.S / wdf.n_tok),
        cv_per_akshara=_cv(wdf.S / wdf.n_aks),
        # ... content words only
        cv_per_char_content=_cv(cw.S / cw.n_char) if ok_c else float("nan"),
        cv_per_token_content=_cv(cw.S / cw.n_tok) if ok_c else float("nan"),
        cv_per_akshara_content=_cv(cw.S / cw.n_aks) if ok_c else float("nan"),
    )
    out["logratio_tok_vs_aks"]   = _lr(out["cv_per_token"], out["cv_per_akshara"])
    out["logratio_char_vs_aks"]  = _lr(out["cv_per_char"],  out["cv_per_akshara"])
    out["logratio_tok_vs_aks_content"]  = _lr(out["cv_per_token_content"], out["cv_per_akshara_content"])
    out["logratio_char_vs_aks_content"] = _lr(out["cv_per_char_content"],  out["cv_per_akshara_content"])
    out["logratio_grain_tok_vs_aks"]     = _lr(out["cv_token"], cv_aks_grain)
    out["logratio_grain_random_vs_aks"]  = _lr(cv_perm, cv_aks_grain)
    out["logratio_grain_randcut_vs_aks"] = _lr(cv_rcut, cv_aks_grain)
    # CHANGE 14: random-cut ratio expected from segment lengths alone, and the excess
    out["logratio_grain_randcut_null"]   = _lr(cv_rcut_null, cv_aks_null)
    out["logratio_grain_randcut_excess"] = out["logratio_grain_randcut_vs_aks"] - out["logratio_grain_randcut_null"]
    # CHANGE 15: strict test = excess over the within-word token-shuffle null
    out["logratio_grain_random_shufnull"]   = _lr(cv_perm_sh, cv_aks_sh)
    out["logratio_grain_random_excess"]     = out["logratio_grain_random_vs_aks"] - out["logratio_grain_random_shufnull"]
    out["logratio_grain_randcut_shufnull"]  = _lr(cv_rcut_sh, cv_aks_sh)
    out["logratio_grain_randcut_shufexcess"] = out["logratio_grain_randcut_vs_aks"] - out["logratio_grain_randcut_shufnull"]
    return out, wdf

# ===================== data =====================
_WS8 = re.compile(r"\s+")

_blk = re.escape(chr(SPEC.base)) + "-" + re.escape(chr(SPEC.base + 0x7F))   # v7: current script block
_SENT = re.compile(r"(?<=[।!?])\s+|(?<=\.)\s+|(?<=\.)(?=[" + _blk + "])")

def split_sents(article):
    return [s.strip() for s in _SENT.split(_WS8.sub(" ", article).strip()) if s.strip()]

def article_passage(article, min_w=25, max_w=60):
    sents = split_sents(article)
    if len(sents) < 2: return None
    out = []
    for s in sents[1:]:
        out.append(s)
        if sum(len(x.split()) for x in out) >= min_w: break
    words = " ".join(out).split()
    if len(words) < min_w: return None
    return " ".join(words[:max_w])

def article_context(article, max_w=300):
    """Article minus its first sentence (which often paraphrases the summary)."""
    sents = split_sents(article)
    return " ".join(" ".join(sents[1:]).split()[:max_w]) if len(sents) > 1 else None

def load_xlsum(n):

    ds = xlsum_split(cfg.DATASET_CFG, cfg.SPLIT)          # v9.1: local copy if available
    df = pd.DataFrame(ds)[["id", "text", "summary"]].rename(
            columns={"text": "article", "summary": "gold_summary"})
    df = df[df["article"].str.strip().ne("") & df["gold_summary"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    print(f"XL-Sum: {cfg.DATASET_CFG} [{cfg.SPLIT}]: {len(df)} usable rows")

    df_eval = df.sample(n=min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    df_eval["article_passage"] = df_eval["article"].map(
        lambda a: article_passage(a, cfg.ARTICLE_MIN_WORDS, cfg.ARTICLE_MAX_WORDS))
    print(f"Eval sample: {len(df_eval)} rows | article passages: "
          f"{df_eval.article_passage.notna().sum()}")
    return df_eval

df = load_xlsum(cfg.NUM_SAMPLES)

# v7: word frequencies are needed before scoring (function-word list for non-Hindi)
from collections import Counter
_tr = xlsum_split(cfg.DATASET_CFG, cfg.FREQ_SPLIT)       # v9.1: local copy if available
FREQ = Counter()
for a in _tr["text"]:
    FREQ.update(DEV_WORD.findall(unicodedata.normalize("NFC", a)))
print(f"frequency table: {len(FREQ):,} types from {cfg.FREQ_SPLIT} ({len(_tr):,} articles)")
if not FUNC_WORDS:
    FUNC_WORDS = {w for w, _ in FREQ.most_common(cfg.FUNC_TOPK)}
    print(f"function words = top {cfg.FUNC_TOPK} train words, e.g.:", " ".join(list(FUNC_WORDS)[:25]))
print("\nExample passage:", df.article_passage.dropna().iloc[0][:200])

# ===================== text cleaning (generation off) =====================
import re as _re
_WS = _re.compile(r"\s+")
def _clean(t): return _WS.sub(" ", t).strip()

if cfg.RUN_GENERATION:
    print(f"Loading generator: {cfg.GENERATOR_MODEL} …")
    gen_tok = AutoTokenizer.from_pretrained(cfg.GENERATOR_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.GENERATOR_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
    print(f"Generator ready, revision={getattr(gen_model.config, '_commit_hash', None)}")

    def summarize(article: str) -> str:
        inp = gen_tok([_clean(article)], return_tensors="pt",
                      truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = gen_model.generate(**inp, num_beams=cfg.NUM_BEAMS,
                                     max_new_tokens=cfg.MAX_NEW_TOKENS,
                                     min_new_tokens=cfg.MIN_NEW_TOKENS,
                                     no_repeat_ngram_size=cfg.NO_REPEAT_NGRAM)
        return _clean(gen_tok.decode(out[0], skip_special_tokens=True))

    print("Sanity:", summarize(df.iloc[0]["article"])[:160])
else:
    print("Generation OFF: RQ2 runs on gold (human) summaries only.")
# ===================== load scorers =====================
scorers = {}
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
for mid in cfg.SCORING_MODELS:

    extra = mid not in PROFILE["models"]
    dev = "cuda:1" if (extra and N_GPU > 1) else DEVICE
    dt = torch.float16 if (extra and N_GPU == 1) else torch.float32
    try:
        scorers[mid] = ReferenceScorer(mid, device=dev, revision=cfg.MODEL_REVISIONS.get(mid), dtype=dt)
        if extra: print(f"  {mid}: device={dev}, dtype={dt}")
    except Exception as e:
        print(f"  scorer {mid} unavailable ({type(e).__name__}: {e}); skipping")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
assert scorers, "Need at least one reference scorer."
assert any(m.split("/")[-1] == cfg.REF_SCORER for m in scorers), \
    f"reference scorer {cfg.REF_SCORER} failed to load; cross-scorer analyses need it"
print("\nRecord these in LANG_PROFILES[...]['revisions']:",
      {m: s.revision for m, s in scorers.items()})

for mid, sc in scorers.items():
    print(f"\n{mid}")
    for w in PROFILE["probe_words"]:
        print("  ", w, "->", sc.show_tokens(unicodedata.normalize("NFC", w)))
    try:   # what the pre-tokenizer does before BPE (v6 cell 17, now generic)
        pt = sc.tokenizer.backend_tokenizer.pre_tokenizer
        if pt is not None:
            print("   pre-tokens:", [p for p, _ in pt.pre_tokenize_str(PROFILE["probe_words"][0])])
    except Exception as e:
        print("   pre-tokens: n/a", e)

for mid, sc in scorers.items():
    tot = split = 0
    for w in GOLD_AKSHARA:
        w = unicodedata.normalize("NFC", w)
        enc = sc.tokenizer(w, add_special_tokens=False, return_offsets_mapping=True)
        edges = {e for a, b in enc.offset_mapping for e in (a, b)}
        for x, y in akshara_spans(w):
            if y - x > 1:
                tot += 1; split += any(x < e < y for e in edges)
    print(f"{mid:28s} complex aksharas split: {split}/{tot} = {split/tot:.0%}")

# ===================== calibration of grain controls =====================
CAL_TEXTS, CAL_PERM = 150, 60
cal_rng = np.random.default_rng(cfg.SEED + 7)
cal_txt = [unicodedata.normalize("NFC", _clean(str(x))) for x in df.gold_summary.iloc[:CAL_TEXTS]]
CAL_COLS = ["logratio_grain_random_vs_aks", "logratio_grain_randcut_vs_aks", "logratio_grain_randcut_excess",
            "logratio_grain_random_excess", "logratio_grain_randcut_shufexcess"]
rows = []
for mid, sc in scorers.items():
    res = []
    for x in cal_txt:
        enc = sc.tokenizer(x, add_special_tokens=False, return_offsets_mapping=True)
        offs = [list(o) for o in enc.offset_mapping]
        s_, _ = measure_text(cal_rng.gamma(2.0, 2.0, len(offs)), offs, x,
                             n_perm_grain=CAL_PERM, min_words=cfg.RQ2_MIN_WORDS)
        if s_: res.append(s_)
    d = pd.DataFrame(res)
    for c in CAL_COLS:
        v = d[c].dropna().to_numpy()
        rows.append(dict(scorer=mid.split("/")[-1], metric=c, n=len(v), mean=v.mean(),
                         se=v.std(ddof=1) / np.sqrt(len(v)), p_wilcoxon=stats.wilcoxon(v).pvalue))
tab_cal = pd.DataFrame(rows); tab_cal.to_csv("results/rq2_grain_calibration.csv", index=False)
print("Grain controls under random, exchangeable surprisal (calibrated test -> mean ~0, p large):")
print(tab_cal.round(4).to_string(index=False))

# ===================== scoring loop =====================
def _save(summ_rows, word_rows, suffix=""):
    pd.DataFrame(summ_rows).to_csv(f"results/rq2_per_text{suffix}.csv", encoding="utf-8-sig", index=False)
    pd.concat(word_rows, ignore_index=True).to_csv(f"results/rq2_per_word{suffix}.csv",
                                                   encoding="utf-8-sig", index=False)

summ_rows, word_rows = [], []
for idx in tqdm(range(len(df))):
    r = df.iloc[idx]
    # each entry: source -> (text, conditioning context or None)
    texts = {"gold": (_clean(str(r["gold_summary"])), None)}
    if cfg.USE_ARTICLE_SOURCE and isinstance(r["article_passage"], str):
        texts["article"] = (r["article_passage"], None)
    if cfg.SCORE_GOLD_WITH_CONTEXT:
        ctx = article_context(r["article"], cfg.CONTEXT_MAX_WORDS)
        if ctx: texts["gold_ctx"] = (texts["gold"][0], ctx)
    if cfg.RUN_GENERATION:
        texts["gen"] = (summarize(r["article"]), None)
    for mid, sc in scorers.items():
        tag = mid.split("/")[-1]
        for src, (t, ctx) in texts.items():
            t = unicodedata.normalize("NFC", t)
            surr, offs, bow_lp, isbow = sc.token_surprisals(t, condition=ctx, return_bow=True)
            surr_c = bow_adjust(surr, bow_lp, isbow) if cfg.BOW_CORRECT else np.asarray(surr, float)
            variants = [("main", surr_c, True, cfg.N_PERM_GRAIN), ("with_first", surr_c, False, 0)]
            if cfg.BOW_CORRECT and cfg.KEEP_RAW_VARIANT:
                variants.append(("raw", np.asarray(surr, float), True, 0))
            for variant, s_use, drop, nperm in variants:
                s, w = measure_text(s_use, offs, t, drop_first=drop, n_perm_grain=nperm,
                                    min_words=cfg.RQ2_MIN_WORDS, tok_is_bow=isbow,
                                    n_shuffle_null=cfg.N_SHUFFLE_NULL if nperm else 0)
                if s is None: continue
                s["bow_total_shift"] = float(np.sum(s_use) - np.sum(surr))   # bits moved by the correction
                key = dict(text_id=r["id"], scorer=tag, source=src, variant=variant)
                summ_rows.append({**key, **s, "text": t})
                word_rows.append(w.assign(**key))
    if cfg.CHECKPOINT_EVERY and (idx + 1) % cfg.CHECKPOINT_EVERY == 0:
        _save(summ_rows, word_rows, "_partial")          # survives a Colab disconnect

summ = pd.DataFrame(summ_rows)
words_all = pd.concat(word_rows, ignore_index=True)
_save(summ_rows, word_rows)
print("saved:", summ.shape, words_all.shape)
print(summ.groupby(["scorer", "source", "variant"]).size())
m = summ[summ.variant == "main"]
print("\nmass of surprisal on analysed words (should be close to 1):")
print(m.groupby("scorer").mass_on_words.describe()[["mean", "min"]].round(3))
print("\nshare of words where random cuts can differ from aksharas:",
      round(m.frac_words_randcut_informative.mean(), 3))
# CHANGE 12 diagnostics: word boundaries must coincide with token boundaries for the
# correction to be exact. Expect >0.98; words with bow_ok == False can be excluded.
print("\nshare of words with clean token boundaries (bow_ok):")
print(m.groupby("scorer").frac_words_bow_ok.mean().round(4))
wm_ = words_all[words_all.variant.isin(["main", "raw"])]
pv_ = wm_.pivot_table(index=["scorer", "source", "text_id", "widx"], columns="variant", values="S")
print("\nmean per-word change from the correction, corrected - raw (bits):")
print((pv_["main"] - pv_["raw"]).groupby(level="scorer").agg(["mean", "std"]).round(3)
      if "raw" in pv_ else "raw variant not kept")

# ===================== analysis =====================
import matplotlib.pyplot as plt
if "summ" not in globals():
    summ = pd.read_csv("results/rq2_per_text.csv", encoding="utf-8-sig")
    words_all = pd.read_csv("results/rq2_per_word.csv", encoding="utf-8-sig")

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j]); adj[j] = min(1.0, run)
    return adj

METRICS = {   # column -> what a positive value means
    "logratio_tok_vs_aks":            "per-akshara rate more uniform than per-token",
    "logratio_char_vs_aks":           "per-akshara rate more uniform than per-char",
    "logratio_tok_vs_aks_content":    "same, content words only",
    "logratio_char_vs_aks_content":   "same, content words only",
    "logratio_grain_tok_vs_aks":      "akshara series smoother than token series",
    "logratio_grain_random_vs_aks":      "akshara vs permuted-size chunks (raw; not calibrated)",
    "logratio_grain_randcut_vs_aks":     "akshara vs random cuts (raw; length artifact)",
    "logratio_grain_randcut_null":       "diagnostic: random-cut ratio from segment lengths alone",
    "logratio_grain_randcut_excess":     "diagnostic: random cuts beyond the length-only null",
    "logratio_grain_random_excess":      "PRIMARY: permuted sizes, excess over within-word shuffle null",
    "logratio_grain_randcut_shufexcess": "random cuts, excess over within-word shuffle null",
}
METRICS = {k: v for k, v in METRICS.items() if k in summ.columns}

def stat_table(d, n_boot=cfg.N_BOOTSTRAP, seed=cfg.SEED):
    rng = np.random.default_rng(seed); rows = []
    for (sc, src), g in d.groupby(["scorer", "source"]):
        for c in METRICS:
            x = g[c].dropna().to_numpy()
            if len(x) < 10: continue
            bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
            rows.append(dict(scorer=sc, source=src, metric=c, n=len(x),
                             mean=x.mean(), ci_lo=np.percentile(bs, 2.5),
                             ci_hi=np.percentile(bs, 97.5),
                             d_z=x.mean() / x.std(ddof=1), frac_pos=(x > 0).mean(),
                             p_wilcoxon=stats.wilcoxon(x).pvalue))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p_wilcoxon); return t

main = summ[summ.variant == "main"]
tab_main = stat_table(main)
tab_wf   = stat_table(summ[summ.variant == "with_first"])
tab_main.to_csv("results/rq2_stats_main.csv", index=False)
tab_wf.to_csv("results/rq2_stats_with_first.csv", index=False)
if (summ.variant == "raw").any():                       
    tab_raw = stat_table(summ[summ.variant == "raw"])
    tab_raw.to_csv("results/rq2_stats_raw.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(tab_main.round(4).to_string(index=False))

print("\nShare of token->akshara grain effect attributable to akshara BOUNDARIES:")
for (sc, src), g in main.groupby(["scorer", "source"]):
    base = g.logratio_grain_tok_vs_aks.mean()
    if base <= 0.01:   
        print(f"  {sc:11s} {src:8s} n/a (token grain not rougher than akshara: base={base:+.3f})"); continue
    msg = f"  {sc:11s} {src:8s} permuted-size raw {g.logratio_grain_random_vs_aks.mean()/base:6.1%}"
    if "logratio_grain_random_excess" in g:
        msg += f" | permuted-size excess over shuffle null {g.logratio_grain_random_excess.mean()/base:+7.1%}"
    print(msg)

# --- gold vs article (paired) ----------------------------------------------------
print("\nGold minus article (paired Wilcoxon):")
_ga = []   
for sc, g in main[main.source.isin(["gold", "article"])].groupby("scorer"):
    for c in ["cv_per_akshara", "logratio_tok_vs_aks", "logratio_char_vs_aks"]:
        pv = g.pivot(index="text_id", columns="source", values=c).dropna()
        dlt = pv["gold"] - pv["article"]
        _ga.append(dict(scorer=sc, metric=c, n=len(dlt), delta=dlt.mean(),
                        p_wilcoxon=stats.wilcoxon(dlt).pvalue))
tab_ga = pd.DataFrame(_ga); tab_ga["p_holm"] = holm(tab_ga.p_wilcoxon)
tab_ga.to_csv("results/rq2_gold_vs_article.csv", index=False)
print(tab_ga.round(4).to_string(index=False))

# --- cross-scorer agreement ------------------------------------------------------
print("\nCross-scorer Spearman of per-text cv_per_akshara:")
print(main.pivot_table(index=["text_id", "source"], columns="scorer",
                       values="cv_per_akshara").corr("spearman").round(2))

# --- word level: which length unit predicts surprisal, beyond frequency? -------
try:
    if "FREQ" not in globals():
        _tr = xlsum_split(cfg.DATASET_CFG, cfg.FREQ_SPLIT)
        from collections import Counter
        FREQ = Counter()
        for a in _tr["text"]:
            FREQ.update(DEV_WORD.findall(unicodedata.normalize("NFC", a)))
        print(f"\nfrequency table: {len(FREQ):,} types from {cfg.FREQ_SPLIT}")
except Exception as e:
    FREQ = None; print("frequency table unavailable:", e)

def r2(X, y):
    X = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b

wm = words_all[words_all.variant == "main"].copy()
rows = []
for sc, g in wm.groupby("scorer"):
    y = np.log(g.S.to_numpy())
    feats = {"char": np.log(g.n_char), "aks": np.log(g.n_aks), "tok": np.log(g.n_tok)}
    base = {}
    if FREQ is not None:
        base["logfreq"] = np.log1p(g.word.map(lambda w: FREQ.get(w, 0)).to_numpy())
    B = np.c_[tuple(base.values())] if base else np.empty((len(y), 0))
    r2_base = r2(B, y)[0] if base else 0.0
    row = dict(scorer=sc, r2_base=r2_base)
    for k, f in feats.items():
        row[f"dR2_{k}"] = r2(np.c_[B, f], y)[0] - r2_base
    row["dR2_aks_over_char"] = r2(np.c_[B, feats["char"], feats["aks"]], y)[0] - r2(np.c_[B, feats["char"]], y)[0]
    row["dR2_char_over_aks"] = r2(np.c_[B, feats["char"], feats["aks"]], y)[0] - r2(np.c_[B, feats["aks"]], y)[0]
    row["slope_logS_logaks"] = stats.linregress(feats["aks"], y).slope
    rows.append(row)
tab_len = pd.DataFrame(rows)
tab_len.to_csv("results/rq2_length_units.csv", index=False)
print("\nIncremental R^2 of log word surprisal "
      f"({'over log-frequency' if FREQ is not None else 'no frequency control'}):")
print(tab_len.round(4).to_string(index=False))

def _split_cat(d):
    if "crosses_aks" not in d:            # older CSVs: no alignment column
        return np.where(d.n_tok > 1, "split", "unsplit")
    return np.select([d.n_tok == 1, ~d.crosses_aks.astype(bool)],
                     ["unsplit", "split_aligned"], "split_crossing")

def _boot_mean_by_text(r, tid, n_boot=1000, seed=cfg.SEED):
    g = pd.DataFrame({"r": r, "t": tid}).groupby("t").r.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

wm["cat"] = _split_cat(wm)
if "has_mark" not in wm:                                     # older CSVs
    wm["has_mark"] = wm.word.map(lambda w: bool(_MARK.search(w)))
rows = []
for sc, g in wm.groupby("scorer"):
    y = np.log2(g.S.to_numpy())
    cols = [np.log(g.n_aks), np.log(g.n_char), g.is_func.astype(float)]
    if FREQ is not None:
        cols.insert(0, np.log1p(g.word.map(lambda w: FREQ.get(w, 0)).to_numpy()))
    X = np.c_[np.ones(len(y)), np.column_stack(cols)]
    uns = (g.cat == "unsplit").to_numpy()
    for fit, mask in [("UNSPLIT", uns), ("FULL", np.ones(len(y), bool))]:
        if mask.sum() < 50: continue
        b, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None); res = y - X @ b
        for cat in sorted(g.cat.unique()):
            m = (g.cat == cat).to_numpy()
            lo, hi = _boot_mean_by_text(res[m], g.text_id.to_numpy()[m])
            rows.append(dict(scorer=sc, fit=fit, category=cat, n=int(m.sum()),
                             share=m.mean(), mean_resid_bits=res[m].mean(), ci_lo=lo, ci_hi=hi,
                             share_has_mark=g.has_mark[m].mean(), share_func=g.is_func[m].mean()))
tab_split = pd.DataFrame(rows)
tab_split.to_csv("results/rq2_split_residuals.csv", index=False)
print("\nMean residual of log2 word surprisal by split category "
      f"({'with' if FREQ is not None else 'WITHOUT'} log-frequency control):")
print(tab_split.round(3).to_string(index=False))

key = ["text_id", "source", "widx"] if "widx" in wm else None
if key is not None and cfg.REF_SCORER in set(wm.scorer):
    p = wm.pivot_table(index=key, columns="scorer", values=["S", "n_tok"]).dropna()
    rows = []
    for o in [s for s in wm.scorer.unique() if s != cfg.REF_SCORER]:
        so, sb = p[("n_tok", o)] > 1, p[("n_tok", cfg.REF_SCORER)] > 1
        cat = np.select([~so & ~sb, so & ~sb, so & sb], ["both_unsplit", f"split_only_{o}", "both_split"], f"split_only_{cfg.REF_SCORER}")
        d = (p[("S", o)] - p[("S", cfg.REF_SCORER)]).to_numpy()
        tids = p.index.get_level_values("text_id").to_numpy()
        for c in np.unique(cat):
            m = cat == c
            lo, hi = _boot_mean_by_text(d[m], tids[m])
            rows.append(dict(model=o, vs=cfg.REF_SCORER, category=c, n=int(m.sum()),
                             mean_dS_bits=d[m].mean(), median=np.median(d[m]), ci_lo=lo, ci_hi=hi))
    tab_x = pd.DataFrame(rows); tab_x.to_csv("results/rq2_cross_scorer_dS.csv", index=False)
    print("\nSame word, model minus BLOOM surprisal (bits), by split status:")
    print(tab_x.round(3).to_string(index=False))
else:
    print("\n(cross-scorer delta skipped: rerun scoring so per-word rows carry 'widx')")

# --- FINAL-6: run metadata for the thesis appendix ---------------------------------
import dataclasses
meta = dict(timestamp=datetime.datetime.now().isoformat(), python=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__,
            datasets=datasets.__version__, config=dataclasses.asdict(cfg),
            n_rows_per_text=len(summ), n_rows_per_word=len(words_all),
            scorer_revisions={m: s.revision for m, s in scorers.items()},     # v9: incl. unpinned extras
            freq_control=FREQ is not None)
json.dump(meta, open("results/run_meta.json", "w"), indent=2, default=str, ensure_ascii=False)
print("\nsaved results/run_meta.json")

# --- figure: forest plot of the main log-ratios ----------------------------------
fig, ax = plt.subplots(figsize=(8, 0.32 * len(tab_main) + 1))
lab = tab_main.metric.str.replace("logratio_", "") + " | " + tab_main.scorer + " | " + tab_main.source
yy = np.arange(len(tab_main))[::-1]
ax.errorbar(tab_main["mean"], yy, xerr=[tab_main["mean"] - tab_main.ci_lo,
            tab_main.ci_hi - tab_main["mean"]], fmt="o", ms=3, capsize=2)
ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=7)
ax.set_xlabel("mean log CV ratio (>0 favours akshara), 95% bootstrap CI")
plt.tight_layout(); plt.savefig("plots/rq2_forest.png", dpi=200); plt.show()


import shutil
rng_v6 = np.random.default_rng(cfg.SEED)
B_V6 = 1000                                    

def _text_weights(tids, n_boot=B_V6, seed=cfg.SEED):
    ut, inv = np.unique(tids, return_inverse=True)
    W = np.random.default_rng(seed).integers(0, len(ut), (n_boot, len(ut)))
    C = np.stack([np.bincount(w, minlength=len(ut)) for w in W])      # (n_boot, n_texts) counts
    return inv, C

def _wmean(x, mask, inv, C):
    n = np.bincount(inv, mask.astype(float), C.shape[1]); s = np.bincount(inv, np.where(mask, x, 0.0), C.shape[1])
    return (C @ s) / np.maximum(C @ n, 1e-12)

# ---------- A. what the word-boundary correction changed ----------------------------
if (summ.variant == "raw").any():
    KEY = ["logratio_tok_vs_aks", "logratio_char_vs_aks", "logratio_tok_vs_aks_content",
           "logratio_char_vs_aks_content", "cv_per_akshara", "cv_word_total"]
    rows = []
    for (sc, src), g in summ[summ.variant.isin(["main", "raw"])].groupby(["scorer", "source"]):
        for c in KEY:
            pv = g.pivot(index="text_id", columns="variant", values=c).dropna()
            d = pv["main"] - pv["raw"]
            rows.append(dict(scorer=sc, source=src, metric=c, raw=pv["raw"].mean(),
                             corrected=pv["main"].mean(), delta=d.mean(),
                             p_wilcoxon=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
    tab_bow = pd.DataFrame(rows); tab_bow["p_holm"] = holm(tab_bow.p_wilcoxon)
    tab_bow.to_csv("results/rq2_bow_effect.csv", index=False)
    print("A. Corrected minus raw (per-text metrics):")
    print(tab_bow.round(4).to_string(index=False))
    wr = words_all[words_all.variant.isin(["main", "raw"])]
    pw = wr.pivot_table(index=["scorer", "source", "text_id", "widx"], columns="variant",
                        values="S").dropna().reset_index()
    meta_w = wr[wr.variant == "main"][["scorer", "source", "text_id", "widx", "n_tok", "has_mark"]]
    pw = pw.merge(meta_w, on=["scorer", "source", "text_id", "widx"])
    pw["dS"] = pw["main"] - pw["raw"]; pw["split"] = np.where(pw.n_tok > 1, "split", "unsplit")
    print("\nPer-word change in S from the correction (bits), by split status:")
    print(pw.groupby(["scorer", "split"]).dS.agg(["mean", "std", "count"]).round(3))
else:
    print("A. skipped: no 'raw' variant in this run")

# ---------- B. baseline-centred cross-scorer ΔΔS --------------------------------------
def ddS_table(wv, ref=cfg.REF_SCORER):
    p = wv.pivot_table(index=["text_id", "source", "widx"], columns="scorer",
                       values=["S", "n_tok", "n_aks"]).dropna()
    extra = [c for c in ["is_func", "bow_ok", "word"] if c in wv]
    q = wv.pivot_table(index=["text_id", "source", "widx"], columns="scorer", values=extra,
                       aggfunc="first").reindex(p.index)                       # v7.1: covariates
    tids = p.index.get_level_values("text_id").to_numpy(); inv, C = _text_weights(tids)
    rows = []
    for o in [s for s in wv.scorer.unique() if s != ref]:
        so, sr_ = (p[("n_tok", o)] > 1).to_numpy(), (p[("n_tok", ref)] > 1).to_numpy()
        d = (p[("S", o)] - p[("S", ref)]).to_numpy(); na = p[("n_aks", o)].to_numpy()
        A, Bm = so & ~sr_, ~so & ~sr_                          # split only in o | split in neither
        for stratum, keep in [("all", np.ones(len(d), bool)), ("n_aks=1", na == 1),
                              ("n_aks=2", na == 2), ("n_aks=3", na == 3), ("n_aks>=4", na >= 4)]:
            a_, b_ = A & keep, Bm & keep
            if a_.sum() < 30 or b_.sum() < 30:
                rows.append(dict(model=o, ref=ref, stratum=stratum, n_split_only=int(a_.sum()),
                                 n_both_unsplit=int(b_.sum()), ddS_bits=np.nan, ci_lo=np.nan, ci_hi=np.nan)); continue
            bs = _wmean(d, a_, inv, C) - _wmean(d, b_, inv, C)
            rows.append(dict(model=o, ref=ref, stratum=stratum, n_split_only=int(a_.sum()),
                             n_both_unsplit=int(b_.sum()), raw_dS_split_only=d[a_].mean(),
                             raw_dS_both_unsplit=d[b_].mean(), ddS_bits=d[a_].mean() - d[b_].mean(),
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
        # v7.1 robustness rows
        if "bow_ok" in q:
            ok = (q[("bow_ok", o)] == 1).to_numpy() & (q[("bow_ok", ref)] == 1).to_numpy()
            a_, b_ = A & ok, Bm & ok
            if a_.sum() >= 30 and b_.sum() >= 30:
                bs = _wmean(d, a_, inv, C) - _wmean(d, b_, inv, C)
                rows.append(dict(model=o, ref=ref, stratum="clean boundaries", n_split_only=int(a_.sum()),
                                 n_both_unsplit=int(b_.sum()), ddS_bits=d[a_].mean() - d[b_].mean(),
                                 ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
        m_ = A | Bm
        X = [np.ones(len(d)), A.astype(float), np.log(na), q[("is_func", o)].astype(float).to_numpy()]
        if FREQ is not None:
            X.append(np.log1p(np.array([FREQ.get(x, 0) for x in q[("word", o)]], float)))
        X = np.column_stack(X); ix0 = np.flatnonzero(m_)
        coef = lambda ix: np.linalg.lstsq(X[ix], d[ix], rcond=None)[0][1]
        grp = [ix0[inv[ix0] == k] for k in range(len(np.unique(tids)))]
        bs = [coef(np.concatenate([grp[k] for k in rng_v6.integers(0, len(grp), len(grp))])) for _ in range(500)]
        rows.append(dict(model=o, ref=ref, stratum="adjusted (length, function word"
                         + (", frequency)" if FREQ is not None else ")"),
                         n_split_only=int(A.sum()), n_both_unsplit=int(Bm.sum()), ddS_bits=coef(ix0),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows)

def ddS_all(wv):
    """v9: the original scorers are analysed together exactly as before (same rows, same numbers);
    each extra scorer is paired with the reference scorer only, so it cannot change the others."""
    parts = [ddS_table(wv[wv.scorer.isin(CORE_TAGS)])]
    for o in [s for s in wv.scorer.unique() if s not in CORE_TAGS]:
        parts.append(ddS_table(wv[wv.scorer.isin([o, cfg.REF_SCORER])]))
    return pd.concat(parts, ignore_index=True)

tab_dd = []
for var in [v for v in ["main", "raw"] if (words_all.variant == v).any()]:
    t_ = ddS_all(words_all[words_all.variant == var]); t_.insert(0, "variant", var); tab_dd.append(t_)
tab_dd = pd.concat(tab_dd, ignore_index=True)
tab_dd.to_csv("results/rq2_cross_scorer_ddS.csv", index=False)
print("\nB. ΔΔS = (model − ref | split only in model) − (model − ref | split in neither), bits:")
print(tab_dd.round(3).to_string(index=False))

# ---------- C. cross-scorer agreement per normaliser ---------------------------------
rows = []
for c in ["cv_word_total", "cv_per_char", "cv_per_akshara", "cv_per_token"]:
    pv = main.pivot_table(index=["text_id", "source"], columns="scorer", values=c)
    cr = pv.corr("spearman"); sc_ = list(cr.columns)
    for i in range(len(sc_)):
        for j in range(i + 1, len(sc_)):
            ok = pv[[sc_[i], sc_[j]]].dropna(); tid = ok.index.get_level_values("text_id").to_numpy()
            ut, invt = np.unique(tid, return_inverse=True)
            groups = [np.flatnonzero(invt == k) for k in range(len(ut))]
            vals = ok.to_numpy(); bs = []
            for _ in range(300):
                ix = np.concatenate([groups[k] for k in rng_v6.integers(0, len(ut), len(ut))])
                bs.append(stats.spearmanr(vals[ix, 0], vals[ix, 1]).statistic)
            rows.append(dict(normaliser=c, pair=f"{sc_[i]} ~ {sc_[j]}", spearman=cr.iloc[i, j],
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
tab_agree = pd.DataFrame(rows); tab_agree.to_csv("results/rq2_cross_scorer_agreement.csv", index=False)
print("\nC. Cross-scorer Spearman agreement of per-text CVs:")
print(tab_agree.round(3).to_string(index=False))

# ---------- D. length-unit ΔR² with cluster-bootstrap CIs --------------------------
def _r2_from(X, y):
    X = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rows = []
for sc, g in wm.groupby("scorer"):
    y = np.log(g.S.to_numpy()); F = [np.log(g.n_char), np.log(g.n_aks)]
    base = [np.log1p(g.word.map(lambda w: FREQ.get(w, 0)).to_numpy())] if FREQ is not None else []
    X_char, X_aks, X_both = (np.column_stack(base + [F[0]]), np.column_stack(base + [F[1]]),
                             np.column_stack(base + F))
    def stat(ix):   # (unique ΔR² of aksharas over chars, unique ΔR² of chars over aksharas)
        r_full = _r2_from(X_both[ix], y[ix])
        return r_full - _r2_from(X_char[ix], y[ix]), r_full - _r2_from(X_aks[ix], y[ix])
    est = stat(np.arange(len(y)))
    tid = g.text_id.to_numpy(); ut, invt = np.unique(tid, return_inverse=True)
    groups = [np.flatnonzero(invt == k) for k in range(len(ut))]
    bs = np.array([stat(np.concatenate([groups[k] for k in rng_v6.integers(0, len(ut), len(ut))]))
                   for _ in range(200)])
    rows.append(dict(scorer=sc, dR2_aks_over_char=est[0], lo_aks=np.percentile(bs[:, 0], 2.5),
                     hi_aks=np.percentile(bs[:, 0], 97.5), dR2_char_over_aks=est[1],
                     lo_char=np.percentile(bs[:, 1], 2.5), hi_char=np.percentile(bs[:, 1], 97.5),
                     frac_boot_aks_gt_char=(bs[:, 0] > bs[:, 1]).mean()))
tab_len_ci = pd.DataFrame(rows); tab_len_ci.to_csv("results/rq2_length_units_ci.csv", index=False)
print("\nD. Unique ΔR² of log S: aksharas over chars vs chars over aksharas (95% CI over texts):")
print(tab_len_ci.round(5).to_string(index=False))

# ---------- E. figures for the report ----------------------------------------------
plt.rcParams.update({"font.size": 9})
SC_ORDER = [m.split("/")[-1] for m in cfg.SCORING_MODELS if m.split("/")[-1] in set(main.scorer)]

# E1. compact forest: the four metrics that carry the argument (gold summaries)
KEYM = [("logratio_char_vs_aks", "rate: per-char vs per-akshara"),
        ("logratio_tok_vs_aks", "rate: per-token vs per-akshara"),
        ("logratio_grain_random_excess", "boundary (permuted sizes)\nexcess over shuffle null"),
        ("logratio_grain_randcut_shufexcess", "boundary (random cuts)\nexcess over shuffle null")]
KEYM = [(k, l) for k, l in KEYM if k in set(tab_main.metric)]
fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.8), sharey=True)
for ax in np.ravel(axes)[len(KEYM):]: ax.axis("off")
for ax, (k, lab) in zip(np.ravel(axes), KEYM):
    t_ = tab_main[(tab_main.metric == k)]
    for yi, sc in enumerate(SC_ORDER):
        for off, src, mk in [(-0.12, "gold", "o"), (0.12, "article", "s")]:
            r_ = t_[(t_.scorer == sc) & (t_.source == src)]
            if len(r_):
                r_ = r_.iloc[0]
                ax.errorbar(r_["mean"], yi + off, xerr=[[r_["mean"] - r_.ci_lo], [r_.ci_hi - r_["mean"]]],
                            fmt=mk, ms=4, capsize=2, color="C0" if src == "gold" else "C1",
                            label=src if yi == 0 else None)
    ax.axvline(0, color="grey", lw=0.8); ax.set_title(lab, fontsize=8.5)
    ax.set_yticks(range(len(SC_ORDER))); ax.set_yticklabels(SC_ORDER)
h_, l_ = np.ravel(axes)[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper right", fontsize=7.5, ncol=2, frameon=False)
fig.supxlabel("mean log CV ratio (> 0 favours akshara), 95% CI over texts", fontsize=8.5)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig("plots/fig_key_metrics.png", dpi=200); plt.show()

# E2. what each grain control shows before and after its null
NC = [("logratio_grain_randcut_vs_aks", "random cuts: observed"),
      ("logratio_grain_randcut_null", "random cuts: length-only null"),
      ("logratio_grain_randcut_shufnull", "random cuts: shuffle null"),
      ("logratio_grain_random_vs_aks", "permuted sizes: observed"),
      ("logratio_grain_random_shufnull", "permuted sizes: shuffle null")]
NC = [(c, l) for c, l in NC if c in main]
if NC:
    agg = main.groupby("scorer")[[c for c, _ in NC]].mean().reindex(SC_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 2.8)); x = np.arange(len(agg)); w_ = 0.8 / len(NC)
    for i, (c, l) in enumerate(NC):
        ax.bar(x + (i - (len(NC) - 1) / 2) * w_, agg[c], w_, label=l)
    ax.axhline(0, color="grey", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(agg.index)
    ax.set_ylabel("log CV ratio vs akshara"); ax.legend(fontsize=6.5, ncol=2)
    plt.tight_layout(); plt.savefig("plots/fig_grain_nulls.png", dpi=200); plt.show()

# E3. ΔΔS with CIs (main = corrected; raw shown if present)
t_ = tab_dd[tab_dd.stratum == "all"]
fig, ax = plt.subplots(figsize=(4.2, 2.4))
for i, (var, gg) in enumerate(t_.groupby("variant", sort=False)):
    yy = np.arange(len(gg)) + 0.15 * (i - 0.5)
    ax.errorbar(gg.ddS_bits, yy, xerr=[gg.ddS_bits - gg.ci_lo, gg.ci_hi - gg.ddS_bits],
                fmt="o", capsize=3, label="corrected" if var == "main" else "raw (v5)")
ax.set_yticks(range(len(gg))); ax.set_yticklabels(gg.model + " vs BLOOM")
ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("ΔΔS (bits): split-only − both-unsplit")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("plots/fig_ddS.png", dpi=200); plt.show()

# E4. what decides splitting: words with vs without combining marks
comp = (wm.assign(status=np.select([wm.n_tok == 1, ~wm.crosses_aks.astype(bool)],
                                   ["unsplit", "split, akshara-aligned"], "split, crosses akshara"))
          .groupby(["scorer", "has_mark"]).status.value_counts(normalize=True).unstack(fill_value=0))
comp = comp.reindex(columns=["unsplit", "split, akshara-aligned", "split, crosses akshara"], fill_value=0)
fig, ax = plt.subplots(figsize=(5.6, 2.6))
labels = [f"{sc}\n{'with' if hm else 'no'} mark" for sc, hm in comp.index]
left = np.zeros(len(comp))
for col in comp.columns:
    ax.barh(labels, comp[col], left=left, label=col); left += comp[col].to_numpy()
ax.set_xlabel("share of words"); ax.legend(fontsize=7, loc="lower right"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig("plots/fig_split_by_mark.png", dpi=200); plt.show()

# ---------- F. compact summary to paste back + zip -----------------------------------
summary = {
    "bow_ok_share": main.groupby("scorer").frac_words_bow_ok.mean().round(4).to_dict()
                    if "frac_words_bow_ok" in main else None,
    "key_metrics_gold": tab_main[(tab_main.source == "gold") & tab_main.metric.isin([k for k, _ in KEYM])]
                        [["scorer", "metric", "mean", "ci_lo", "ci_hi", "d_z", "p_holm"]].round(4).to_dict("records"),
    "ddS_all": tab_dd[tab_dd.stratum == "all"][["variant", "model", "ddS_bits", "ci_lo", "ci_hi"]].round(3).to_dict("records"),
}
json.dump(summary, open("results/rq2_v6_summary.json", "w"), indent=1, ensure_ascii=False, default=str)
shutil.make_archive(f"mtp7_{cfg.DATASET_CFG}_results", "zip", ".", "results")
shutil.make_archive(f"mtp7_{cfg.DATASET_CFG}_plots", "zip", ".", "plots")
print(f"\nWrote results/rq2_v6_summary.json and mtp7_{cfg.DATASET_CFG}_results.zip / _plots.zip in {RUN_DIR} -> download and share these.")

print("PIPELINE_DONE", LANG)
