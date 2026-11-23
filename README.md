# MTP-2 Replication Package

**Revisiting Uniform Information Density in Hindi: A Diagnostic Study of Tokenization and Akshara-Level Surprisal in Indian Languages**

This package contains the code, a sample dataset and the full results of the thesis. Everything is run with
**one script, `run_all.py`**.

---

## 1. Install

```bash
pip install -r requirements.txt
```

To run only step 2.1 below, these four packages are enough: `pip install "pandas<3" numpy scipy matplotlib`

## 2. Run

There are three ways to run it, from quickest to most complete.

| | What it does | Command | Time | Needs |
|---|---|---|---|---|
| **2.1** | Check the thesis numbers | `python run_all.py --check` | ~1 min | any computer; no GPU, no internet |
| **2.2** | Run the whole pipeline on the sample data | `python run_all.py --sample` | ~10–20 min | GPU recommended; internet (downloads the models, ~7 GB) |
| **2.3** | Reproduce everything from scratch | `python run_all.py` | ~6–7 h | 2 GPUs (e.g. Kaggle T4 ×2); internet |

### 2.1 Check the thesis numbers

This rebuilds every table and finding of the thesis from the saved results in `reference_results/`.
The run should end with:

```
Reproducibility (mtp-8 reference): 248/248 match
Reproducibility (v9 reference): 58/58 match
```

The findings are written to `reference_results/FINAL/FINAL_FINDINGS.md`.

### 2.2 Run the whole pipeline on the sample data

This takes 50 Hindi texts from the bundled sample. It scores them with all five language models, maps surprisal
onto words, tokens, characters and aksharas, runs every statistical test, and writes a report to
`mtp_sample/FINAL/FINAL_FINDINGS.md`.

The numbers will differ from the thesis, because only 50 texts are used, and the report will mark them as
`DIFFERS`. That is expected: this step shows that the pipeline runs end to end. Step 2.1 checks the thesis numbers.

### 2.3 Reproduce everything

This runs the tokenizer audit, Hindi, Bengali and Marathi (1,000 texts each), the eye-tracking analysis and the
final report. The full datasets are downloaded automatically. Results go to `mtp_final/FINAL/`, which should show
the same `match` lines as step 2.1. A Hugging Face token (`export HF_TOKEN=...`) is needed only for four gated
tokenizers in the audit; without one, those are skipped.

### No GPU? Use Kaggle

1. Upload this folder as a Kaggle Dataset.
2. Open `notebooks/run_on_kaggle.ipynb` on Kaggle and add the dataset as input.
3. Turn on Internet and choose **GPU T4 ×2**.
4. Run the cells. Set `RUN_SAMPLE = True` for step 2.2 or `RUN_FULL = True` for step 2.3.

---

## 3. What is in the package

```
README.md            this guide
DETAILS.md           technical details: data sampling, model versions, all options, how results are checked
run_all.py           the one script that runs everything
requirements.txt     software to install
src/                 pipeline code (akshara segmenter, surprisal experiment, tokenizer audit, eye-tracking, report)
scripts/             helper scripts (export the dataset, language-model quality table)
data/sample/         SAMPLE DATASET: Hindi XL-Sum texts (test split + 3,000 training articles)
data/meco/           MECO Hindi eye-tracking data
reference_results/   all results of the thesis run, including per-word surprisal for every model
notebooks/           Kaggle notebook, and the original notebook of the thesis run with its outputs
assets/              font used to measure the printed width of words
```

## 4. Data

| Dataset | In this package | Full dataset | Licence |
|---|---|---|---|
| XL-Sum (Hindi, Bengali, Marathi news texts) | Hindi sample in `data/sample/` | Hugging Face `csebuetnlp/xlsum`, downloaded automatically | CC BY-NC-SA 4.0 |
| MECO Wave 2 (Hindi eye-tracking) | complete file in `data/meco/` | https://osf.io/3527a/ | CC BY 4.0 |
| Pretrained language models (5) | not included | Hugging Face, downloaded automatically, pinned versions | each model's licence |

## 5. Where the thesis results come from

- **All findings, with a verdict per language:** `FINAL/FINAL_FINDINGS.md`
- **The tables behind each chapter:** `FINAL/table_*.csv` (for example `table_scale_control.csv` for the primary result)
- **Figures:** `FINAL/fig_*.png` and `meco/fig_meco.png`

`DETAILS.md` (section 4.1) maps every result in the thesis to its file.
