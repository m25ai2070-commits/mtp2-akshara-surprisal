# Data

| Folder | Contents | Source and licence |
|---|---|---|
| `sample/xlsum/hindi/test.jsonl.gz`, `train.jsonl.gz` | **Sample dataset** for the end-to-end run (`python run_all.py --sample`): the complete Hindi XL-Sum test split and the first 3,000 Hindi train articles (word frequencies). Columns: id, url, title, summary, text. `MANIFEST.json` gives row counts and checksums | Hasan et al. (2021), Hugging Face `csebuetnlp/xlsum`. CC BY-NC-SA 4.0 (non-commercial) |
| `meco/joint_data_trimmed_wave2_version2.1.rda` | MECO Wave 2 word-level eye-tracking report (the Hindi samples `hi_iiith` and `hi_iitk` are analysed). md5 `55a0fc3291853dd2817903d4038e8c60` | Siegelman et al. (2025), *Scientific Data* 12:1183; OSF https://osf.io/3527a/ (release 2.0, version 2.1). CC BY 4.0 |
| `xlsum/<language>/<split>.jsonl.gz` (optional) | **Full data** for an offline full run: test and train splits for Hindi, Bengali and Marathi, test splits for Nepali, Gujarati, Punjabi, Tamil and Telugu (audit). Complete, original row order | as above |

How to create them (internet needed once, no GPU):

```bash
python scripts/export_xlsum.py --sample     # data/sample/xlsum/  (~10 MB)
python scripts/export_xlsum.py              # data/xlsum/         (~90 MB, full)
```

Without these folders the pipeline downloads the same data from Hugging Face. The sample is used only by
`run_all.py --sample`; a full run uses `data/xlsum/` or the download, never the sample.

The per-word and per-text surprisal values of every model (the derived dataset behind all tables of the thesis)
are in `reference_results/run_<language>/results/` (`rq2_per_word.csv.gz`, `rq2_per_text.csv.gz`).
