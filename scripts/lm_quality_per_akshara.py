import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="mtp_final")
A = ap.parse_args()
RULE = {"gpt2-hindi", "gpt2-bengali", "mGPT"}
rows = []
for lang in ["hindi", "bengali", "marathi"]:
    f = os.path.join(A.base, f"run_{lang}", "results", "rq2_per_word.csv")
    if not os.path.exists(f):
        f += ".gz"                                                 # gzipped copy (GitHub size limit)
        if not os.path.exists(f): continue
    w = pd.read_csv(f, encoding="utf-8-sig", usecols=["S", "n_aks", "n_tok", "scorer", "source", "variant"])
    w = w[w.variant == "main"]
    for (sc, src), g in w.groupby(["scorer", "source"]):
        rows.append(dict(corpus=f"xlsum-{lang}-{src}", model=sc, bits_per_akshara=g.S.sum() / g.n_aks.sum(),
                         tokens_per_word=g.n_tok.mean(), n_words=len(g)))
mw = os.path.join(A.base, "meco", "meco_words.csv")
if os.path.exists(mw):
    m = pd.read_csv(mw, encoding="utf-8-sig")
    first = m.groupby("text").cumcount() == 0                      # same exclusions as the MECO analysis
    m = m[~first & m.has_dev & (m.n_aks > 0)]
    fit = pd.read_csv(os.path.join(A.base, "meco", "meco_surprisal_ll.csv"))
    fit = fit[(fit.measure == "gaze_ms") & (fit.surprisal == "raw")].set_index("model")
    for c in [c for c in m.columns if c.startswith("S_")]:
        sc = c[2:]
        rows.append(dict(corpus="meco-hindi", model=sc, bits_per_akshara=m[c].sum() / m.n_aks.sum(),
                         tokens_per_word=m[f"ntok_{sc}"].mean(), n_words=len(m),
                         meco_gaze_dll_x1000=fit.dll_x1000.get(sc), meco_ci_lo=fit.ci_lo.get(sc),
                         meco_ci_hi=fit.ci_hi.get(sc)))
t = pd.DataFrame(rows)
t["tokenizer"] = t.model.map(lambda s: "rule (cuts vowel signs)" if s in RULE else "learned subwords")
os.makedirs(os.path.join(A.base, "FINAL"), exist_ok=True)
out = os.path.join(A.base, "FINAL", "table_lm_quality.csv"); t.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False)); print("\nwrote", out)
