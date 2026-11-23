# Technical details of the replication package (MTP-2)

*For the short guide, see `README.md`. This file documents the data, models, every option and the reproducibility check.*
*`python run_all.py --check` is the same as `python run_all.py --steps report --base reference_results`.*

This package reproduces every number, table and figure of the MTP-2 thesis *"Revisiting Uniform Information
Density in Hindi: A Diagnostic Study of Tokenization and Akshara-Level Surprisal in Indian Languages"* from public
data and public pretrained models.

## 0. What is in this zip

| Part | Where | What it is |
|---|---|---|
| **Source code** | `run_all.py`, `src/`, `scripts/` | The complete pipeline: one command (`run_all.py`) runs every step |
| **Sample dataset** | `data/sample/xlsum/hindi/` and `data/meco/` | Hindi XL-Sum sample (test split + 3,000 train articles) and the MECO Hindi eye-tracking file: enough to run every step end to end (§1.2) |
| **Full results** | `reference_results/` | Every output of the reported run, incl. per-word surprisal of every model (the derived dataset behind all tables) |
| **Instructions** | this `README.md`, `data/README.md` | How to obtain the data, run each step, and check the results |
| **Notebooks** | `notebooks/` | A Kaggle notebook that runs everything; the notebook of the reported run with its outputs (provenance) |

**Full dataset.** The full datasets are public and are not needed to test the pipeline. They can be shared
separately or downloaded automatically: XL-Sum (Hugging Face `csebuetnlp/xlsum`, CC BY-NC-SA 4.0) and MECO Wave 2
(OSF https://osf.io/3527a/, CC BY 4.0).

## 1. How to reproduce

| Level | Command | Needs | Time |
|---|---|---|---|
| 1. Rebuild every table and verdict from the bundled results | `python run_all.py --steps report --base reference_results` | CPU only, no internet | ~1 min |
| 2. **End-to-end run on the sample dataset** | `python run_all.py --sample` | GPU recommended; internet for the models | ~10–20 min |
| 3. Full reproduction from scratch | `python run_all.py --zip` | 2× T4 GPU (Kaggle), internet | ~6–7 h |

### 1.1 Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -r requirements.txt                          # install the torch build that matches your CUDA driver
export HF_TOKEN=hf_...                                   
```

On Kaggle, use `notebooks/run_on_kaggle.ipynb`

### 1.2 End-to-end run on the sample dataset (level 2)

```bash
python run_all.py --sample
```

What happens, in order (results in `./mtp_sample/`):

1. **checks**: the akshara segmenter is tested against its 190 gold words, and the software versions are recorded.
2. **experiment**: 50 Hindi texts are drawn (seed 42) from the sample's test split. Each gold summary and article
   passage is scored with all five Hindi models (GPT-2 Hindi, mGPT, BLOOM-560m, XGLM-564M, Sarvam-1). Surprisal is
   corrected for word boundaries and mapped onto words, tokens, characters and aksharas. Then all the statistics run:
   evenness per unit, boundary tests with their calibration, split words and length units.
3. **report**: tables, figures and per-language verdicts go to `mtp_sample/FINAL/`.

The models are downloaded on first use, about 7 GB. To include the eye-tracking step as well, using the bundled
MECO file, run:

```bash
python run_all.py --sample --steps checks experiments meco report
```

### 1.3 Rebuild the reported tables (level 1, no GPU, no internet)

```bash
pip install "pandas>=2.2,<3" numpy scipy matplotlib
python run_all.py --steps report --base reference_results      
python scripts/lm_quality_per_akshara.py --base reference_results
```

The console should print `248/248 match` (earlier, independent run) and `58/58 match` (reported run).
`reference_results/FINAL/FINAL_FINDINGS.md` then lists every finding with its verdict per language.

### 1.4 Full reproduction (level 3)

```bash
python run_all.py --zip            
```

The full run uses `data/xlsum/` if it is present, and otherwise downloads XL-Sum. It never uses the sample. At the
end, `mtp_final/FINAL/reproducibility_check.csv` compares the run with 306 reference values.

### 1.6 Useful flags

```bash
python run_all.py --steps experiments --langs hindi --force hindi   
python run_all.py --steps meco report                               
python run_all.py --no-extra                                        
```

Steps whose results already exist are skipped unless you name them in `--force`. If a language run is interrupted,
run the same command again: it resumes from its checkpoints (every 100 texts).

---

## 2. Data, models and other inputs

Bundled in `data/` (see `data/README.md`): the Hindi **sample** (`data/sample/xlsum/`) and the MECO word-level
file (`data/meco/`). Models and tokenizers are downloaded from Hugging Face on first use, and anything else not
bundled is downloaded automatically.

### 2.1 XL-Sum (text)

* Source: Hugging Face dataset `csebuetnlp/xlsum` (Hasan et al., 2021), configs `hindi`, `bengali`, `marathi`.
  It is loaded with `datasets==2.21.0` and `trust_remote_code=True`, because the dataset uses a loading script.
* Licence: CC BY-NC-SA 4.0 (non-commercial).
* Sampling (`src/mtp_pipeline.py`, `load_xlsum`): drop rows with an empty article or summary, then take
  `df.sample(n=1000, random_state=42)` from the **test** split. Two text types are scored per row:
  * `gold`: the human-written summary;
  * `article`: a passage from the article. It starts at the **second** sentence, because the first often
    paraphrases the summary, and adds whole sentences until it has at least 25 words. It is then cut at 60 words
    (`article_passage`, `ARTICLE_MIN_WORDS`, `ARTICLE_MAX_WORDS`).

  Texts with fewer than 5 words are excluded (`RQ2_MIN_WORDS`). Bengali keeps 999 article passages.
* Word frequencies come from the **train** split of the same language (Hindi: 250,496 types from 70,778 articles).
* **Sample:** `data/sample/xlsum/hindi/` holds the complete Hindi test split and the first 3,000 Hindi train articles
  (`python scripts/export_xlsum.py --sample`). Only `run_all.py --sample` reads it.
* **Full offline copy:** `python scripts/export_xlsum.py` writes every split the study reads to
  `data/xlsum/<language>/<split>.jsonl.gz` (about 90 MB compressed), complete and in the original row order, with
  a `MANIFEST.json` of row counts, checksums and the dataset commit. A full run then reads these files instead of
  downloading.
* The reference run did not record the dataset repository's commit. `run_all.py` now records the current commit in
  `mtp_final/environment.json` (`xlsum_dataset_sha`). If the dataset repository changes, the sample may change too.
  Section 5 explains how to check this.

### 2.2 MECO Wave 2 (eye-tracking)

* Source: Multilingual Eye-movement COrpus, L1 Wave 2 (Siegelman et al.), OSF project **https://osf.io/3527a/**.
* Licence: CC BY 4.0.
* **Bundled:** `data/meco/joint_data_trimmed_wave2_version2.1.rda`, the exact file analysed in the reference run
  (redistributed under CC BY 4.0). `run_all.py` uses it automatically. Without it, `run_all.py` walks the OSF folders
  *release 2.x → version (latest) → wave 2 → primary data → eye tracking data* and downloads the word-level report
  into `mtp_final/meco_data/`. The reference run used release 2.0, version 2.1:

  | file | size | md5 |
  |---|---|---|
  | `joint_data_trimmed_wave2_version2.1.rda` (word level, **the one analysed**) | 9.8 MB | `55a0fc3291853dd2817903d4038e8c60` |
  | `joint_sac_trimmed_l1_wave2_version2.1.rda` (downloaded, not used) | 27.8 MB | `71b5fd3d56c12f00a34786ca00cff135` |
  | `joint_readrate_wave2_version2.1.rda` (downloaded, not used) | 30 KB | `68ff0135b7201e042beb59db72e419d8` |

  Check your copy with `md5sum mtp_final/meco_data/*.rda`. If OSF has published a newer version, the downloader
  picks the latest one. To reproduce exactly, put the version 2.1 file in `mtp_final/meco_data/` before running.
* Manual download: on OSF go to Files → release 2.0 → version 2.1 → wave 2 → primary data → eye tracking data, and
  save the word-level `.rda` in `mtp_final/meco_data/` (or pass `--meco-dir`).
* Hindi samples used: `hi_iiith` (54 readers) and `hi_iitk` (57 readers), 12 passages, 2,236 analysed words. The
  first word of each passage and words without Devanagari are excluded.

### 2.3 Pretrained models (surprisal)

All models are loaded at fixed commits (`src/mtp_common.py`, `LANG_PROFILES[...]["revisions"]`) and run in
float32.

| Model | Hugging Face repo | Commit | Languages | Tokenizer type |
|---|---|---|---|---|
| GPT-2 Hindi | `surajp/gpt2-hindi` | `6cd8d84edae1b247b8a2f446421de8f5b9208f18` | hi | rule (pre-tokenizer cuts every vowel sign) |
| GPT-2 Bengali | `flax-community/gpt2-bengali` | `18336d76c4ab3c2972532fd9d4730708bbdd81eb` | bn | rule |
| mGPT | `ai-forever/mGPT` | `40897bd7c8b47a76802c411108ca6220438b8b40` | hi, bn, mr | rule |
| BLOOM-560m | `bigscience/bloom-560m` | `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971` | hi, bn, mr | learned subwords (byte-level BPE) |
| XGLM-564M | `facebook/xglm-564M` | `f3059f01b98ccc877c673149e0178c0e957660f9` | hi, bn | learned subwords (SentencePiece) |
| Sarvam-1 | `sarvamai/sarvam-1` | `e9607337286ddf496d4a2562b194e489dcf3feea` | hi, bn, mr | learned subwords (SentencePiece) |

XGLM is not used for Marathi because it was not trained on Marathi. With two GPUs, XGLM and Sarvam-1 run on the
second GPU in float32, as in the reference run. **With one GPU they run in float16**, so their numbers can differ
slightly from the reference values (see §5).

### 2.4 Tokenizer audit

21 tokenizers (`src/mtp_audit.py`, `AUDIT_TOKENIZERS`) are applied to 300 XL-Sum texts in each of Hindi, Marathi,
Nepali, Bengali, Gujarati, Punjabi, Tamil and Telugu. Only tokenizer files are downloaded, no weights. The
tokenizers are loaded at their current revision, not pinned. In the reference run all 21 loaded. Their exact
repositories are listed in `reference_results/audit/tokenizer_info.csv`.

### 2.5 Font for the visual-width control

`assets/NotoSansDevanagari-Regular.ttf` (Noto Sans Devanagari, SIL Open Font License 1.1, md5
`53ef84da2e0f1a7b49fffe7485395722`) is bundled. That way word widths do not depend on a moving download URL.
`run_all.py` copies it to `mtp_final/meco/` before the MECO step.

---

## 3. What is in this package

```
run_all.py                       one command for every step (--sample for the end-to-end test)
requirements.txt                 pinned packages (see the comments in the file)
src/
  mtp_common.py                  scripts and akshara segmenter (with gold tests), language profiles, model revisions
  mtp_pipeline.py                the surprisal experiment for ONE language (run as a separate process)
  mtp_audit.py                   tokenizer audit
  mtp_meco.py                    MECO Hindi reading-time analysis
  mtp_report.py                  cross-language findings, verdicts, reproducibility check -> FINAL/
scripts/
  export_xlsum.py                writes the XL-Sum sample (--sample) or the full splits for offline runs
  lm_quality_per_akshara.py      tokenizer-independent LM quality (bits per akshara) next to the MECO fit
data/                            sample/xlsum/ (Hindi sample), meco/ (MECO file), xlsum/ (full, optional)
notebooks/
  run_on_kaggle.ipynb            thin Kaggle notebook: install, set token, call run_all.py
  mtp-9_reference_run.ipynb      the notebook that produced the reference results, with its outputs (provenance)
assets/NotoSansDevanagari-Regular.ttf
reference_results/               every output of the reported run, incl. per-word surprisal (§6)
```

---

## 4. Step by step: commands and outputs

All outputs go into the results folder (`--base`, default `./mtp_final`).

| Step | Command | Main outputs |
|---|---|---|
| checks | `python run_all.py --steps checks` | console: akshara segmenter gold tests (Devanagari 125/125, Bengali 38/38, …); `environment.json` (versions, GPUs, `pip freeze`, XL-Sum commit) |
| audit | `--steps audit` | `audit/tokenizer_audit.csv`, `tokenizer_info.csv`, `fig_tokenizer_audit.png` |
| experiments | `--steps experiments [--langs …]` | `run_<lang>/results/rq2_*.csv`, `run_meta.json` (config + model commits actually used), `plots/*.png`, `log.txt` |
| meco | `--steps meco` | `meco/MECO_FINDINGS.md`, `meco_*.csv`, `meco_summary.json`, `fig_meco.png`; `meco_log.txt` |
| report | `--steps report` | `FINAL/FINAL_FINDINGS.md`, `FINAL/table_*.csv`, `fig_cross_language.png`, `reproducibility_check.csv` |
| LM quality | `python scripts/lm_quality_per_akshara.py --base mtp_final` | `FINAL/table_lm_quality.csv` |

`report` must run after `experiments`, `audit` and `meco`, because it checks all of them (`run_all.py` keeps this
order).

### 4.1 Where each result of the thesis comes from

The finding numbers match `FINAL/FINAL_FINDINGS.md`.

| # | Result | Table (FINAL/) | Computed from (run_<lang>/results/ unless stated) |
|---|---|---|---|
| 1 | GPT-2-style pre-tokenizers split every word that contains a vowel sign | `table_splitting.csv` | `rq2_per_word.csv` (`n_tok`) |
| 2 | Per-akshara vs per-character evenness, exponent 1 (no scale control) | `table_char_vs_akshara.csv` | `rq2_stats_main.csv`, metric `logratio_char_vs_aks` |
| 3 | **Primary:** the same after scale control (each unit's best length exponent) | `table_scale_control.csv` (`pct_best`, `ci_lo_best`, `ci_hi_best`); robustness: `table_scale_control_reselect.csv` | `rq2_per_word.csv` |
| 4 | Per-akshara vs whole-word (and vs token) evenness | `table_unit_comparison.csv`, `table_most_even_unit.csv` | `rq2_per_text.csv` |
| 5 | Akshara count vs character count as predictors of word surprisal (unique ΔR², log-log) | `table_length_units.csv` | `rq2_length_units_ci.csv` |
| 6–7 | Cross-model agreement per normaliser | `table_agreement.csv` | `rq2_cross_scorer_agreement.csv` |
| 8–9 | Akshara boundaries inside words (rule / learned-subword tokenizers), calibrated | `table_boundary.csv` | `rq2_stats_main.csv` (`logratio_grain_random_excess`), `rq2_grain_calibration.csv` |
| 10 | Calibration of the shuffle-null tests with random surprisal | (in findings) | `rq2_grain_calibration.csv` |
| 11 | Split penalty vs BLOOM (exploratory) | `table_split_penalty.csv` | `rq2_cross_scorer_ddS.csv` |
| 12 | Words split inside vs between aksharas (learned-subword tokenizers) | `table_split_inside_vs_between.csv` | `rq2_split_residuals.csv` |
| 13 | Tokenizer audit | `table_audit_diagnosis.csv`, `fig_tokenizer_audit.png` | `audit/tokenizer_audit.csv` |
| M-A | MECO native ruler: akshara vs character vs visual width vs token length | — | `meco/meco_word_features_ll.csv`, `meco_ruler_ll.csv`, `meco_by_sample.csv`, `meco_evenness_by_reader_text.csv` |
| M-B | MECO surprisal predictive power per model, raw vs boundary-corrected | — | `meco/meco_surprisal_ll.csv` |
| M-C | MECO residuals of split words | — | `meco/meco_split_residuals.csv` |
| LM | Bits per akshara per model (XL-Sum, MECO) next to the MECO fit | `table_lm_quality.csv` | `rq2_per_word.csv`, `meco/meco_words.csv` |
| Fig. | Cross-language figure; MECO figure; per-language figures | `fig_cross_language.png` | `meco/fig_meco.png`, `run_<lang>/plots/` |

Statistical settings (in `src/mtp_pipeline.py`, class `Config`): seed 42; 5,000 bootstrap resamples over texts for
the per-language statistics; 100 shuffles for the shuffle null; 100 permutations for the grain controls; Holm
correction within each results table. The report's own bootstraps use 2,000 resamples, seed 0. In MECO, held-out
log-likelihood gains are leave-one-passage-out, and 95% CIs bootstrap over the 12 passages.

---

## 5. Checking your run against the reference

`FINAL/reproducibility_check.csv` lists every reference value next to your value. `status` is `match` when
`|yours − reference|` is within the tolerance below, `DIFFERS` when it is not, and `not computed` when that part
was not run. The console and the first lines of `FINAL_FINDINGS.md` summarise the check.

| Reference set (`ref_run`) | Values | What they cover | Tolerance |
|---|---|---|---|
| `mtp-8` (earlier, independent run) | 248 | 64 key statistics of the original scorers (4 metrics × 2 text types × 8 language–model pairs), 8 split rates, 8 split penalties, 168 audit values | stats 0.002, split 0.002, split penalty 0.01, audit 0.01 |
| `v9` (the reference run of this package) | 58 | 40 key statistics + 5 split rates for XGLM/Sarvam-1, 13 MECO log-likelihood gains | as above; MECO 0.5 (×1000 units) |

The reference run (v9, clean from scratch, 2026-11-20) matched all 248 `mtp-8` values. The largest difference was
5×10⁻⁶ for statistics and 5×10⁻⁵ for the audit, even though the CUDA build (cu128 → cu130) and Python patch
version had changed. The `v9` values match the v9 run by construction, so they are an independent check only for
later runs. Its MECO values also agree with the rounded values from the earlier run (akshara over character +76;
BLOOM / gpt2-hindi / mGPT surprisal +83 / +26 / +29).

Expect `DIFFERS` when:
* you run on one GPU (XGLM/Sarvam-1 in float16): small differences in their rows only;
* you use `--n` other than 1000, or `--no-extra` (the extra-model rows are then `not computed`);
* the XL-Sum dataset repository, a model repository or MECO on OSF changed. Compare `environment.json` and
  `run_<lang>/results/run_meta.json` with the reference copies in `reference_results/`. The first lines of each
  `log.txt` show the number of usable rows (Hindi test: 8,847) and an example passage, which detects a changed
  sample immediately.

---

## 6. Reference results

* `reference_results/` holds every output of the reference run: per-language results including the per-word and
  per-text surprisal of every model (`rq2_per_word.csv.gz`, `rq2_per_text.csv.gz`; gzipped to fit GitHub's file-size limit, read directly by the code), logs, `run_meta.json`, plots, the
  tokenizer audit, the MECO outputs and `FINAL/` (rebuilt with this package's report code). It is enough for
  level 1 (§1.3). `reference_results/FINAL_as_run_v9/` keeps `FINAL_FINDINGS.md` and the reproducibility table exactly as
  the v9 notebook wrote them, before the later report-wording fixes.
* Not bundled: the model weights (downloaded from Hugging Face) and two MECO files the analysis does not use.
---
