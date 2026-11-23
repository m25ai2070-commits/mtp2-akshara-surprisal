import argparse, hashlib, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = {   # language -> splits the study reads
    "hindi": ["test", "train"], "bengali": ["test", "train"], "marathi": ["test", "train"],   # experiment + frequencies
    "nepali": ["test"], "gujarati": ["test"], "punjabi": ["test"], "tamil": ["test"], "telugu": ["test"],  # audit
}
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sample", action="store_true", help="write the small Hindi sample to data/sample/xlsum/")
ap.add_argument("--sample-train-rows", type=int, default=3000, help="train articles kept in the sample (frequencies)")
ap.add_argument("--langs", nargs="+", default=None, help="full mode only: languages to export (default: all)")
A = ap.parse_args()

from datasets import load_dataset

out = os.path.join(ROOT, "data", "sample" if A.sample else "", "xlsum")
plan = {"hindi": ["test", "train"]} if A.sample else {l: FULL[l] for l in (A.langs or FULL)}
manifest = {"_mode": "sample" if A.sample else "full"}
for lang, splits in plan.items():
    for split in splits:
        ds = load_dataset("csebuetnlp/xlsum", lang, split=split, trust_remote_code=True)
        df = ds.to_pandas()
        if A.sample and split == "train":
            df = df.head(A.sample_train_rows)
        d = os.path.join(out, lang); os.makedirs(d, exist_ok=True)
        f = os.path.join(d, f"{split}.jsonl.gz")
        df.to_json(f, orient="records", lines=True, force_ascii=False, compression="gzip")
        manifest[f"{lang}/{split}"] = dict(rows=len(df), rows_in_split=len(ds), md5=hashlib.md5(open(f, "rb").read()).hexdigest(),
                                           file=os.path.relpath(f, ROOT))
        print(f"{lang:9s} {split:5s} {len(df):7,d} of {len(ds):,} rows -> {os.path.relpath(f, ROOT)}")
try:
    from huggingface_hub import HfApi
    manifest["_dataset_commit"] = HfApi().dataset_info("csebuetnlp/xlsum").sha
except Exception as e:
    manifest["_dataset_commit"] = f"unavailable ({type(e).__name__})"
mf = os.path.join(out, "MANIFEST.json")
old = json.load(open(mf)) if os.path.exists(mf) else {}
old.update(manifest); json.dump(old, open(mf, "w"), indent=1)
print("manifest:", os.path.relpath(mf, ROOT))
