"""mtp_audit: how 21 tokenizers cut words in 8 Indian languages (CPU only, tokenizer files only).
run_audit(base_dir) writes <base_dir>/audit/tokenizer_audit.csv, tokenizer_info.csv, fig_tokenizer_audit.png"""
import os, re, time, unicodedata
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from mtp_common import SPECS, LANG_SCRIPT, xlsum_split
DATASET, SPLIT = "csebuetnlp/xlsum", "test"

def tqdm(it, **kw):
    it = list(it); n = len(it); t0 = time.time()
    for i, x in enumerate(it, 1):
        yield x
        print(f"    tokenizer {i}/{n} done, {time.time() - t0:,.0f}s", flush=True)

# ===================== PART A: TOKENIZER AUDIT  =====================
AUDIT_LANGS = ["hindi", "marathi", "nepali", "bengali", "gujarati", "punjabi", "tamil", "telugu"]
AUDIT_N = 300
AUDIT_TOKENIZERS = [                         
    ("gpt2 (English)", "openai-community/gpt2"),
    ("gpt2-hindi", "surajp/gpt2-hindi"),
    ("gpt2-bengali", "flax-community/gpt2-bengali"),
    ("mGPT", "ai-forever/mGPT"),
    ("BLOOM", "bigscience/bloom-560m"),
    ("XGLM", "facebook/xglm-564M"),
    ("mT5", "google/mt5-small"),
    ("XLM-R", "FacebookAI/xlm-roberta-base"),
    ("MuRIL", "google/muril-base-cased"),
    ("IndicBERT-v2", "ai4bharat/IndicBERTv2-MLM-only"),
    ("Pythia (GPT-NeoX)", "EleutherAI/pythia-160m"),
    ("Phi-3", "microsoft/Phi-3-mini-4k-instruct"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B"),
    ("Qwen3", "Qwen/Qwen3-0.6B"),
    ("gpt-oss (o200k)", "openai/gpt-oss-20b"),
    ("Sarvam-1", "sarvamai/sarvam-1"),
    ("OpenHathi", "sarvamai/OpenHathi-7B-Hi-v0.1-Base"),
    ("Llama-3.2", "meta-llama/Llama-3.2-1B|unsloth/Llama-3.2-1B"),   
    ("Gemma-2", "google/gemma-2-2b|unsloth/gemma-2-2b"),
    ("Gemma-3", "google/gemma-3-1b-pt|unsloth/gemma-3-1b-pt"),
    ("Mistral-7B", "mistralai/Mistral-7B-v0.1"),
]
NOT_CAUSAL = {"mT5", "XLM-R", "MuRIL", "IndicBERT-v2"}   

def _rule_cuts_marks(pre_tok_repr):
    s = str(pre_tok_repr)
    if "use_regex=True" in s: return True                  
    for pat in re.findall(r'Regex\("(.*?)"\)', s):
        if r"\p{L}" in pat and r"\p{M}" not in pat: return True
    return False

def _load_tok(repo):
    errs = []
    for r in repo.split("|"):
        try:
            return AutoTokenizer.from_pretrained(r), r
        except Exception as e:
            errs.append(f"{r}: {type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError(" ; ".join(errs))

def _audit_texts(lang, n):
    ds = xlsum_split(lang, SPLIT)                         
    return [unicodedata.normalize("NFC", _WSA.sub(" ", t).strip()) for t in ds["summary"][:n]]
_WSA = re.compile(r"\s+")

def _audit_one(tok, texts, sp):
    unk = tok.unk_token_id
    c = dict(words=0, tokens=0, split=0, mark_words=0, mark_split=0, nomark_words=0, nomark_split=0,
             aks=0, aks_complex=0, aks_cut=0, marks=0, marks_initial=0, words_cut=0, unk=0, chars=0, bound_ok=0,
             ll=0, ll_cut=0, subchar=0)
    base = sp.cons | sp.vowel
    for text in texts:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = []
        for (a, b), tid in zip(enc["offset_mapping"], enc["input_ids"]):
            while a < b and text[a].isspace(): a += 1
            while b > a and text[b - 1].isspace(): b -= 1
            if b > a: spans.append((a, b, tid))
        starts = {a for a, _, _ in spans}
        prev = None
        for a, b, _ in spans:                              # byte fragments share one character's span
            if prev == (a, b) or b - a == 0: c["subchar"] += 1
            prev = (a, b)
        for m in sp.word_re.finditer(text):
            w0, w1 = m.start(), m.end()
            ts = [(a, b, t) for a, b, t in spans if min(b, w1) - max(a, w0) > 0]
            if not ts: continue
            edges = {e for a, b, _ in ts for e in (a, b) if w0 < e < w1}
            nt = len(ts); has_mark = bool(sp.mark_re.search(m.group()))
            c["words"] += 1; c["tokens"] += nt; c["split"] += nt > 1; c["chars"] += w1 - w0
            c["unk"] += sum(t == unk for _, _, t in ts) if unk is not None else 0
            c["bound_ok"] += ts[0][0] == w0 and ts[-1][1] == w1
            c["mark_words" if has_mark else "nomark_words"] += 1
            c["mark_split" if has_mark else "nomark_split"] += nt > 1
            cut_any = False
            for x, y in sp.spans(m.group()):
                c["aks"] += 1
                if y - x > 1:
                    c["aks_complex"] += 1
                    cut = any(w0 + x < e < w0 + y for e in edges); c["aks_cut"] += cut; cut_any |= cut
            c["words_cut"] += cut_any
            for k in range(1, w1 - w0):                   # start of each run of marks (ै, ैं, ्, ़ा …)
                if sp.mark_re.match(text[w0 + k]) and not sp.mark_re.match(text[w0 + k - 1]):
                    c["marks"] += 1; c["marks_initial"] += (w0 + k) in starts
                if text[w0 + k] in base and text[w0 + k - 1] in base:      
                    c["ll"] += 1; c["ll_cut"] += (w0 + k) in starts
    r = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    return dict(n_words=c["words"], tokens_per_word=r("tokens", "words"), chars_per_token=r("chars", "tokens"),
                frac_words_split=r("split", "words"), frac_split_with_mark=r("mark_split", "mark_words"),
                frac_split_no_mark=r("nomark_split", "nomark_words"),
                tokens_per_akshara=c["tokens"] / c["aks"] if c["aks"] else float("nan"),
                frac_complex_aksharas_cut=r("aks_cut", "aks_complex"), frac_words_cut_inside_akshara=r("words_cut", "words"),
                frac_marks_token_initial=r("marks_initial", "marks"), unk_rate=r("unk", "tokens"),
                frac_letter_bounds_cut=r("ll_cut", "ll"), frac_subchar_tokens=r("subchar", "tokens"),
                frac_word_edges_clean=r("bound_ok", "words"))


def run_audit(base_dir, langs=None, n=None):
    langs = langs or AUDIT_LANGS; n = n or AUDIT_N
    AUDIT_DIR = os.path.join(base_dir, "audit"); os.makedirs(AUDIT_DIR, exist_ok=True)
    texts_by_lang = {}
    for lang in langs:
        try:
            texts_by_lang[lang] = _audit_texts(lang, n); print(f"texts: {lang:9s} {len(texts_by_lang[lang])}")
        except Exception as e:
            print(f"texts: {lang} unavailable ({type(e).__name__}: {e})")
    rows, info = [], []
    for label, repo in tqdm(AUDIT_TOKENIZERS):
        try:
            tok, used = _load_tok(repo)
            if not tok.is_fast: raise RuntimeError("no fast tokenizer (offsets unavailable)")
        except Exception as e:
            info.append(dict(label=label, repo=repo, status=f"skipped: {str(e)[:160]}")); continue
        bt = tok.backend_tokenizer
        info.append(dict(label=label, repo=used, status="ok", vocab=len(tok), model=type(bt.model).__name__,
                         rule_cuts_marks=_rule_cuts_marks(bt.pre_tokenizer), causal_lm=label not in NOT_CAUSAL,
                         pre_tokenizer=str(bt.pre_tokenizer)[:300], normalizer=str(bt.normalizer)[:200]))
        repo = used
        for lang, texts in texts_by_lang.items():
            rows.append(dict(tokenizer=label, repo=repo, language=lang, script=LANG_SCRIPT[lang],
                             **_audit_one(tok, texts, SPECS[LANG_SCRIPT[lang]])))
    audit = pd.DataFrame(rows); audit_info = pd.DataFrame(info)
    ok = audit_info[audit_info.status == "ok"].set_index("label")
    audit["rule_cuts_marks"] = audit.tokenizer.map(ok.rule_cuts_marks)
    audit["causal_lm"] = audit.tokenizer.map(ok.causal_lm)
    # diagnosis per tokenizer x language (thresholds are descriptive, not tuned)
    audit["diagnosis"] = np.select(
        [audit.frac_letter_bounds_cut > 0.9,
         audit.rule_cuts_marks & (audit.frac_marks_token_initial > 0.95),
         audit.frac_marks_token_initial > 0.5],
        ["script ~unknown: character/byte level",
         "rule cuts every vowel sign",
         "learned subwords, many signs cut"],
        "learned subwords")
    audit.to_csv(os.path.join(AUDIT_DIR, "tokenizer_audit.csv"), index=False)
    audit_info.to_csv(os.path.join(AUDIT_DIR, "tokenizer_info.csv"), index=False)
    print(audit_info.reindex(columns=["label", "status", "vocab", "model", "rule_cuts_marks", "causal_lm"]).to_string(index=False))
    print("\ndiagnosis:"); print(audit.pivot(index="tokenizer", columns="language", values="diagnosis")
                                .reindex(columns=AUDIT_LANGS).to_string())
    for col in ["frac_marks_token_initial", "frac_letter_bounds_cut", "frac_subchar_tokens",
                "tokens_per_word", "frac_complex_aksharas_cut", "frac_words_split"]:
        print(f"\n{col}:"); print(audit.pivot(index="tokenizer", columns="language", values=col)
                                  .reindex(columns=AUDIT_LANGS).round(3).to_string())
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.38 * audit.tokenizer.nunique() + 2))
    for ax, (col, ttl, cm) in zip(axes, [("frac_marks_token_initial", "vowel-sign runs that start a new token", "Reds"),
                                         ("frac_letter_bounds_cut", "control: letter|letter boundaries cut", "Greys"),
                                         ("tokens_per_word", "tokens per word (fertility)", "Blues")]):
        pv = audit.pivot(index="tokenizer", columns="language", values=col).reindex(columns=AUDIT_LANGS)
        pv = pv.loc[audit.groupby("tokenizer").frac_marks_token_initial.mean().sort_values().index]
        im = ax.imshow(pv.values, cmap=cm, aspect="auto",
                       vmin=0, vmax=1 if col.startswith("frac") else np.nanmax(pv.values))
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=40, ha="right")
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index); ax.set_title(ttl, fontsize=9)
        for i_ in range(pv.shape[0]):
            for j_ in range(pv.shape[1]):
                v = pv.values[i_, j_]
                if np.isfinite(v): ax.text(j_, i_, f"{v:.2f}" if col.startswith("frac") else f"{v:.1f}",
                                           ha="center", va="center", fontsize=6.5)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(AUDIT_DIR, "fig_tokenizer_audit.png"), dpi=200); plt.close(fig)
    print("\nWrote", AUDIT_DIR)
    return audit, audit_info
