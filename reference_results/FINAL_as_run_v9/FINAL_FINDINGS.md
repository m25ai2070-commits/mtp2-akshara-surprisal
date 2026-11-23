# Final findings (2026-11-20)

Languages: hindi, bengali, marathi | texts per language: {'hindi': 1000, 'bengali': 1000, 'marathi': 1000} | tokenizer audit: yes

**Reproducibility:** 248 of 248 reference values match the submitted runs within tolerance (0 not computed). Details: `reproducibility_check.csv`.

## Findings

Verdicts are given per language (v9): the tests within a language share texts and correlated models, so a count of passed tests overstates the evidence. 'Passed' is kept for reference.

| # | Finding | Passed | Verdict | Evidence |
|---|---|---|---|---|
| 1 | Tokenizers with a GPT-2-style pre-tokenizer split every word that contains a combining mark | 5/5 | **descriptive (true by construction of the rule)** | words with a mark split: 100.0%–100.0% (learned-subword tokenizers: 16%–58%) |
| 2 | Per-akshara surprisal is more even than per-character surprisal (exponent 1, NO scale control) | 26/26 | **holds: hindi, bengali, marathi — overstated, see next row** | +1.4% to +8.3% |
| 3 | Per-akshara surprisal is more even than per-character surprisal AFTER scale control (best exponent per unit) — PRIMARY | 14/26 | **holds: hindi; mixed: marathi; not supported: bengali** | hindi +0.5% to +2.9% (best exponent: akshara 0.80–0.90); bengali -0.4% to +0.1% (best exponent: akshara 0.30–0.50); marathi +0.3% to +1.1% (best exponent: akshara 0.55–0.65) |
| 4 | Per-akshara surprisal is more even than whole-word surprisal | 16/26 | **holds: hindi, marathi; not supported: bengali** | hindi +9.7% to +15.6%; bengali -7.8% to -1.3%; marathi +1.0% to +7.8% (best exponents < 1: per-unit rates over-correct for length) |
| 5 | Akshara count predicts word surprisal better than character count (unique ΔR², log-log) | 11/13 | **holds: hindi, bengali; mixed: marathi** | unique ΔR² of aksharas 0.0007–0.0061 (i.e. ≤ 0.6% of variance: small); share of bootstrap samples favouring aksharas: hindi/bloom-560m 100%, hindi/gpt2-hindi 100%, hindi/mGPT 100%, hindi/sarvam-1 100%, hindi/xglm-564M 100%, bengali/bloom-560m 100%, bengali/gpt2-bengali 100%, bengali/mGPT 100%, bengali/sarvam-1 100%, bengali/xglm-564M 100%, marathi/bloom-560m 0%, marathi/mGPT 100%, marathi/sarvam-1 0% |
| 6 | Per-token surprisal is the least consistent measure across models | 3/3 | **descriptive (expected: token counts differ by tokenizer)** | mean cross-model agreement, per-token vs others: bengali 0.44 vs 0.70–0.77, hindi 0.49 vs 0.71–0.75, marathi 0.51 vs 0.68–0.71 |
| 7 | Per-akshara measures agree across models better than per-character measures | 0/3 | **not supported: bengali, hindi, marathi** | akshara vs character agreement: bengali 0.74 vs 0.77, hindi 0.71 vs 0.73, marathi 0.68 vs 0.69 |
| 8 | Akshara boundaries matter inside words (rule tokenizers, calibrated test) | 10/10 | **holds: hindi, bengali, marathi** | +1.3% to +5.0% beyond the shuffle null. Caveat: these tokenizers cut between consonant and vowel sign, so this may reflect the tokenizer's cuts. |
| 9 | Akshara boundaries matter inside words (learned-subword tokenizers, calibrated test) | 7/16 | **mixed: bengali; inconclusive (low power): hindi, marathi** | -0.3% to +1.4%; counted only if larger than the model's own calibration bias (+2 s.e.). Only 15%–55% of words have more than one token, so most words carry no within-word information for this test. |
| 10 | Calibration: with random surprisals the shuffle-null tests give no effect (p > 0.05) | 25/26 | **holds: hindi, bengali; mixed: marathi (≈1.3 failures expected by chance)** | smallest p = 0.045 |
| 11 | Split words receive extra surprisal (split penalty vs BLOOM, rule tokenizers) | 3/5 | **exploratory (not identified: for rule tokenizers 'split' = 'has a mark')** | hindi/gpt2-hindi all +0.18, adjusted +0.18 [+0.12, +0.25]; hindi/mGPT all +0.13, adjusted +0.09 [+0.04, +0.13]; bengali/gpt2-bengali all -0.24, adjusted -0.00 [-0.09, +0.07]; bengali/mGPT all +0.04, adjusted +0.16 [+0.07, +0.23]; marathi/mGPT all -0.46, adjusted -0.30 [-0.49, -0.11] |
| 12 | Learned-subword tokenizers: words split inside an akshara are harder to predict than words split between aksharas | 7/8 | **holds: hindi, bengali; mixed: marathi** | hindi/bloom-560m 0.94 vs 0.72 bits, hindi/sarvam-1 1.09 vs 0.67 bits, hindi/xglm-564M 0.86 vs 0.49 bits, bengali/bloom-560m 0.29 vs 0.16 bits, bengali/sarvam-1 0.36 vs 0.18 bits, bengali/xglm-564M 0.49 vs 0.33 bits, marathi/bloom-560m 0.26 vs 0.28 bits, marathi/sarvam-1 0.45 vs 0.35 bits |
| 13 | Tokenizer audit (21 tokenizers, 8 languages): 8 fine everywhere, 4 with the vowel-sign rule | — | **descriptive** | fine everywhere: BLOOM, Gemma-3, IndicBERT-v2, MuRIL, Sarvam-1, XGLM, XLM-R, mT5; rule cuts every vowel sign: Llama-3.2, gpt2-bengali, gpt2-hindi, mGPT; character/byte level in some languages: Gemma-2, Mistral-7B, OpenHathi, Phi-3, Pythia (GPT-NeoX), Qwen2.5, Qwen3, gpt2 (English) |

## Tables

### splitting

| language | scorer | tokens_per_word | frac_words_split | frac_split_with_mark |
|---|---|---|---|---|
| hindi | bloom-560m | 1.219 | 0.147 | 0.160 |
| hindi | gpt2-hindi | 3.262 | 0.899 | 1.000 |
| hindi | mGPT | 3.444 | 0.919 | 1.000 |
| hindi | sarvam-1 | 1.267 | 0.165 | 0.179 |
| hindi | xglm-564M | 1.412 | 0.268 | 0.288 |
| bengali | bloom-560m | 1.395 | 0.271 | 0.290 |
| bengali | gpt2-bengali | 4.614 | 0.923 | 1.000 |
| bengali | mGPT | 4.871 | 0.950 | 1.000 |
| bengali | sarvam-1 | 1.735 | 0.430 | 0.454 |
| bengali | xglm-564M | 1.941 | 0.552 | 0.580 |
| marathi | bloom-560m | 1.547 | 0.376 | 0.386 |
| marathi | mGPT | 5.286 | 0.986 | 1.000 |
| marathi | sarvam-1 | 1.717 | 0.448 | 0.458 |

### char vs akshara

| language | scorer | source | pct | p_holm |
|---|---|---|---|---|
| hindi | bloom-560m | article | 1.360 | 0.029 |
| hindi | bloom-560m | gold | 1.769 | 0.000 |
| hindi | gpt2-hindi | article | 3.086 | 0.000 |
| hindi | gpt2-hindi | gold | 4.154 | 0.000 |
| hindi | mGPT | article | 1.985 | 0.000 |
| hindi | mGPT | gold | 3.526 | 0.000 |
| hindi | sarvam-1 | article | 1.595 | 0.000 |
| hindi | sarvam-1 | gold | 2.627 | 0.000 |
| hindi | xglm-564M | article | 1.506 | 0.000 |
| hindi | xglm-564M | gold | 2.784 | 0.000 |
| bengali | bloom-560m | article | 7.001 | 0.000 |
| bengali | bloom-560m | gold | 5.462 | 0.000 |
| bengali | gpt2-bengali | article | 8.306 | 0.000 |
| bengali | gpt2-bengali | gold | 6.496 | 0.000 |
| bengali | mGPT | article | 7.598 | 0.000 |
| bengali | mGPT | gold | 5.791 | 0.000 |
| bengali | sarvam-1 | article | 7.817 | 0.000 |
| bengali | sarvam-1 | gold | 5.930 | 0.000 |
| bengali | xglm-564M | article | 6.718 | 0.000 |
| bengali | xglm-564M | gold | 5.624 | 0.000 |
| marathi | bloom-560m | article | 2.511 | 0.000 |
| marathi | bloom-560m | gold | 3.206 | 0.000 |
| marathi | mGPT | article | 3.603 | 0.000 |
| marathi | mGPT | gold | 4.872 | 0.000 |
| marathi | sarvam-1 | article | 3.050 | 0.000 |
| marathi | sarvam-1 | gold | 3.038 | 0.000 |

### scale control

| language | scorer | source | n_texts | gamma | best_exp_akshara | best_exp_char | pct_exp1 | ci_lo_exp1 | ci_hi_exp1 | pct_gamma | ci_lo_gamma | ci_hi_gamma | pct_best | ci_lo_best | ci_hi_best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hindi | bloom-560m | article | 1000 | 1.026 | 0.800 | 0.750 | 1.360 | 0.751 | 1.986 | 1.712 | 1.103 | 2.355 | 0.469 | 0.006 | 0.930 |
| hindi | bloom-560m | gold | 1000 | 1.026 | 0.800 | 0.700 | 1.769 | 1.000 | 2.532 | 2.109 | 1.321 | 2.840 | 1.048 | 0.487 | 1.572 |
| hindi | gpt2-hindi | article | 1000 | 1.026 | 0.850 | 0.750 | 3.086 | 2.380 | 3.817 | 3.453 | 2.746 | 4.157 | 1.814 | 1.326 | 2.366 |
| hindi | gpt2-hindi | gold | 1000 | 1.026 | 0.850 | 0.750 | 4.154 | 3.310 | 5.003 | 4.470 | 3.615 | 5.373 | 2.906 | 2.284 | 3.545 |
| hindi | mGPT | article | 1000 | 1.026 | 0.850 | 0.750 | 1.985 | 1.343 | 2.619 | 2.340 | 1.701 | 2.997 | 0.961 | 0.508 | 1.429 |
| hindi | mGPT | gold | 1000 | 1.026 | 0.850 | 0.750 | 3.526 | 2.672 | 4.351 | 3.826 | 2.981 | 4.679 | 2.509 | 1.906 | 3.102 |
| hindi | sarvam-1 | article | 1000 | 1.026 | 0.800 | 0.700 | 1.595 | 0.943 | 2.198 | 1.925 | 1.295 | 2.526 | 0.646 | 0.191 | 1.116 |
| hindi | sarvam-1 | gold | 1000 | 1.026 | 0.800 | 0.700 | 2.627 | 1.970 | 3.329 | 2.940 | 2.224 | 3.589 | 1.587 | 1.067 | 2.100 |
| hindi | xglm-564M | article | 1000 | 1.026 | 0.900 | 0.800 | 1.506 | 0.858 | 2.130 | 1.804 | 1.181 | 2.467 | 0.674 | 0.104 | 1.198 |
| hindi | xglm-564M | gold | 1000 | 1.026 | 0.900 | 0.750 | 2.784 | 2.051 | 3.600 | 3.042 | 2.253 | 3.829 | 1.971 | 1.365 | 2.537 |
| bengali | bloom-560m | article | 999 | 0.821 | 0.450 | 0.400 | 7.001 | 6.167 | 7.902 | -0.633 | -1.285 | 0.032 | -0.333 | -0.605 | -0.058 |
| bengali | bloom-560m | gold | 1000 | 0.821 | 0.450 | 0.400 | 5.462 | 4.487 | 6.369 | -1.246 | -2.031 | -0.530 | -0.311 | -0.622 | -0.011 |
| bengali | gpt2-bengali | article | 999 | 0.821 | 0.400 | 0.300 | 8.306 | 7.392 | 9.182 | 0.175 | -0.490 | 0.847 | -0.075 | -0.284 | 0.140 |
| bengali | gpt2-bengali | gold | 1000 | 0.821 | 0.300 | 0.250 | 6.496 | 5.588 | 7.474 | -0.528 | -1.218 | 0.224 | -0.022 | -0.226 | 0.191 |
| bengali | mGPT | article | 999 | 0.821 | 0.450 | 0.350 | 7.598 | 6.670 | 8.492 | -0.733 | -1.408 | -0.057 | -0.317 | -0.581 | -0.061 |
| bengali | mGPT | gold | 1000 | 0.821 | 0.400 | 0.350 | 5.791 | 4.840 | 6.791 | -1.508 | -2.306 | -0.733 | -0.355 | -0.647 | -0.054 |
| bengali | sarvam-1 | article | 999 | 0.821 | 0.400 | 0.300 | 7.817 | 6.908 | 8.756 | 0.312 | -0.354 | 0.997 | 0.028 | -0.189 | 0.239 |
| bengali | sarvam-1 | gold | 1000 | 0.821 | 0.350 | 0.300 | 5.930 | 5.042 | 6.852 | -0.266 | -0.993 | 0.448 | 0.042 | -0.174 | 0.265 |
| bengali | xglm-564M | article | 999 | 0.821 | 0.500 | 0.400 | 6.718 | 5.856 | 7.556 | -0.854 | -1.522 | -0.161 | -0.227 | -0.536 | 0.086 |
| bengali | xglm-564M | gold | 1000 | 0.821 | 0.500 | 0.400 | 5.624 | 4.680 | 6.583 | -0.761 | -1.567 | 0.010 | 0.078 | -0.288 | 0.449 |
| marathi | bloom-560m | article | 1000 | 0.882 | 0.600 | 0.550 | 2.511 | 1.780 | 3.263 | -0.860 | -1.531 | -0.228 | 0.276 | -0.095 | 0.651 |
| marathi | bloom-560m | gold | 1000 | 0.882 | 0.650 | 0.550 | 3.206 | 2.394 | 4.062 | 0.269 | -0.439 | 1.038 | 0.442 | -0.019 | 0.883 |
| marathi | mGPT | article | 1000 | 0.882 | 0.550 | 0.500 | 3.603 | 2.902 | 4.320 | -0.054 | -0.713 | 0.585 | 0.875 | 0.546 | 1.231 |
| marathi | mGPT | gold | 1000 | 0.882 | 0.550 | 0.450 | 4.872 | 4.071 | 5.738 | 1.457 | 0.726 | 2.183 | 1.115 | 0.757 | 1.459 |
| marathi | sarvam-1 | article | 1000 | 0.882 | 0.550 | 0.450 | 3.050 | 2.432 | 3.671 | 0.052 | -0.490 | 0.633 | 0.703 | 0.409 | 1.001 |
| marathi | sarvam-1 | gold | 1000 | 0.882 | 0.550 | 0.450 | 3.038 | 2.308 | 3.760 | 0.431 | -0.206 | 1.062 | 0.498 | 0.196 | 0.807 |

### unit comparison

| language | scorer | source | unit | mean_cv_unit | mean_cv_akshara | pct | ci_lo | ci_hi | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| hindi | bloom-560m | article | word | 0.909 | 0.801 | 13.431 | 12.413 | 14.375 | 0.000 |
| hindi | bloom-560m | article | character | 0.812 | 0.801 | 1.360 | 0.759 | 1.991 | 0.016 |
| hindi | bloom-560m | article | token | 0.793 | 0.801 | -0.525 | -1.274 | 0.244 | 0.921 |
| hindi | bloom-560m | gold | word | 0.894 | 0.798 | 12.318 | 11.225 | 13.501 | 0.000 |
| hindi | bloom-560m | gold | character | 0.812 | 0.798 | 1.769 | 1.001 | 2.479 | 0.000 |
| hindi | bloom-560m | gold | token | 0.795 | 0.798 | 0.186 | -0.742 | 1.084 | 0.921 |
| hindi | gpt2-hindi | article | word | 0.876 | 0.760 | 15.219 | 14.172 | 16.290 | 0.000 |
| hindi | gpt2-hindi | article | character | 0.785 | 0.760 | 3.086 | 2.401 | 3.780 | 0.000 |
| hindi | gpt2-hindi | article | token | 0.941 | 0.760 | 22.948 | 21.381 | 24.595 | 0.000 |
| hindi | gpt2-hindi | gold | word | 0.868 | 0.756 | 15.296 | 14.171 | 16.441 | 0.000 |
| hindi | gpt2-hindi | gold | character | 0.788 | 0.756 | 4.154 | 3.329 | 5.022 | 0.000 |
| hindi | gpt2-hindi | gold | token | 0.928 | 0.756 | 21.771 | 20.070 | 23.488 | 0.000 |
| hindi | mGPT | article | word | 0.898 | 0.787 | 14.005 | 13.011 | 14.976 | 0.000 |
| hindi | mGPT | article | character | 0.804 | 0.787 | 1.985 | 1.364 | 2.619 | 0.000 |
| hindi | mGPT | article | token | 0.892 | 0.787 | 13.065 | 11.936 | 14.254 | 0.000 |
| hindi | mGPT | gold | word | 0.887 | 0.770 | 15.571 | 14.455 | 16.667 | 0.000 |
| hindi | mGPT | gold | character | 0.797 | 0.770 | 3.526 | 2.695 | 4.375 | 0.000 |
| hindi | mGPT | gold | token | 0.877 | 0.770 | 13.427 | 12.056 | 14.850 | 0.000 |
| hindi | sarvam-1 | article | word | 0.999 | 0.909 | 10.173 | 9.202 | 11.112 | 0.000 |
| hindi | sarvam-1 | article | character | 0.924 | 0.909 | 1.595 | 0.979 | 2.184 | 0.000 |
| hindi | sarvam-1 | article | token | 0.907 | 0.909 | 0.124 | -0.725 | 0.975 | 0.921 |
| hindi | sarvam-1 | gold | word | 0.994 | 0.911 | 9.701 | 8.674 | 10.692 | 0.000 |
| hindi | sarvam-1 | gold | character | 0.933 | 0.911 | 2.627 | 1.919 | 3.261 | 0.000 |
| hindi | sarvam-1 | gold | token | 0.925 | 0.911 | 2.098 | 1.155 | 3.064 | 0.000 |
| hindi | xglm-564M | article | word | 0.930 | 0.807 | 14.846 | 13.814 | 15.804 | 0.000 |
| hindi | xglm-564M | article | character | 0.820 | 0.807 | 1.506 | 0.893 | 2.161 | 0.000 |
| hindi | xglm-564M | article | token | 0.796 | 0.807 | -1.166 | -1.948 | -0.371 | 0.056 |
| hindi | xglm-564M | gold | word | 0.935 | 0.817 | 14.439 | 13.256 | 15.601 | 0.000 |
| hindi | xglm-564M | gold | character | 0.840 | 0.817 | 2.784 | 2.012 | 3.553 | 0.000 |
| hindi | xglm-564M | gold | token | 0.818 | 0.817 | 0.465 | -0.433 | 1.420 | 0.918 |
| bengali | bloom-560m | article | word | 0.681 | 0.698 | -2.200 | -3.244 | -1.176 | 0.009 |
| bengali | bloom-560m | article | character | 0.749 | 0.698 | 7.001 | 6.154 | 7.903 | 0.000 |
| bengali | bloom-560m | article | token | 0.606 | 0.698 | -12.769 | -13.597 | -11.979 | 0.000 |
| bengali | bloom-560m | gold | word | 0.660 | 0.684 | -3.252 | -4.394 | -2.027 | 0.000 |
| bengali | bloom-560m | gold | character | 0.724 | 0.684 | 5.462 | 4.466 | 6.443 | 0.000 |
| bengali | bloom-560m | gold | token | 0.602 | 0.684 | -11.538 | -12.542 | -10.552 | 0.000 |
| bengali | gpt2-bengali | article | word | 0.694 | 0.742 | -5.959 | -6.843 | -5.029 | 0.000 |
| bengali | gpt2-bengali | article | character | 0.807 | 0.742 | 8.306 | 7.444 | 9.191 | 0.000 |
| bengali | gpt2-bengali | article | token | 0.958 | 0.742 | 28.655 | 26.966 | 30.433 | 0.000 |
| bengali | gpt2-bengali | gold | word | 0.687 | 0.750 | -7.797 | -8.883 | -6.692 | 0.000 |
| bengali | gpt2-bengali | gold | character | 0.802 | 0.750 | 6.496 | 5.585 | 7.415 | 0.000 |
| bengali | gpt2-bengali | gold | token | 0.931 | 0.750 | 23.084 | 21.223 | 25.032 | 0.000 |
| bengali | mGPT | article | word | 0.647 | 0.679 | -4.448 | -5.468 | -3.418 | 0.000 |
| bengali | mGPT | article | character | 0.734 | 0.679 | 7.598 | 6.665 | 8.473 | 0.000 |
| bengali | mGPT | article | token | 0.796 | 0.679 | 16.650 | 15.253 | 18.066 | 0.000 |
| bengali | mGPT | gold | word | 0.627 | 0.672 | -6.127 | -7.303 | -4.947 | 0.000 |
| bengali | mGPT | gold | character | 0.713 | 0.672 | 5.791 | 4.792 | 6.805 | 0.000 |
| bengali | mGPT | gold | token | 0.770 | 0.672 | 14.046 | 12.463 | 15.730 | 0.000 |
| bengali | sarvam-1 | article | word | 0.745 | 0.791 | -5.397 | -6.317 | -4.467 | 0.000 |
| bengali | sarvam-1 | article | character | 0.857 | 0.791 | 7.817 | 6.929 | 8.709 | 0.000 |
| bengali | sarvam-1 | article | token | 0.725 | 0.791 | -7.897 | -8.710 | -7.038 | 0.000 |
| bengali | sarvam-1 | gold | word | 0.745 | 0.796 | -5.903 | -6.925 | -4.854 | 0.000 |
| bengali | sarvam-1 | gold | character | 0.847 | 0.796 | 5.930 | 5.009 | 6.848 | 0.000 |
| bengali | sarvam-1 | gold | token | 0.746 | 0.796 | -5.646 | -6.650 | -4.573 | 0.000 |
| bengali | xglm-564M | article | word | 0.674 | 0.684 | -1.347 | -2.499 | -0.183 | 0.291 |
| bengali | xglm-564M | article | character | 0.733 | 0.684 | 6.718 | 5.883 | 7.633 | 0.000 |
| bengali | xglm-564M | article | token | 0.617 | 0.684 | -9.418 | -10.308 | -8.483 | 0.000 |
| bengali | xglm-564M | gold | word | 0.674 | 0.686 | -1.530 | -2.898 | -0.071 | 0.056 |
| bengali | xglm-564M | gold | character | 0.725 | 0.686 | 5.624 | 4.664 | 6.601 | 0.000 |
| bengali | xglm-564M | gold | token | 0.624 | 0.686 | -8.522 | -9.637 | -7.452 | 0.000 |
| marathi | bloom-560m | article | word | 0.716 | 0.685 | 5.080 | 4.025 | 6.144 | 0.000 |
| marathi | bloom-560m | article | character | 0.701 | 0.685 | 2.511 | 1.769 | 3.237 | 0.000 |
| marathi | bloom-560m | article | token | 0.605 | 0.685 | -11.670 | -12.501 | -10.858 | 0.000 |
| marathi | bloom-560m | gold | word | 0.706 | 0.660 | 7.802 | 6.526 | 9.025 | 0.000 |
| marathi | bloom-560m | gold | character | 0.682 | 0.660 | 3.206 | 2.367 | 4.060 | 0.000 |
| marathi | bloom-560m | gold | token | 0.596 | 0.660 | -9.750 | -10.781 | -8.735 | 0.000 |
| marathi | mGPT | article | word | 0.705 | 0.688 | 2.681 | 1.685 | 3.627 | 0.000 |
| marathi | mGPT | article | character | 0.714 | 0.688 | 3.603 | 2.863 | 4.316 | 0.000 |
| marathi | mGPT | article | token | 0.738 | 0.688 | 6.627 | 5.557 | 7.835 | 0.000 |
| marathi | mGPT | gold | word | 0.694 | 0.675 | 2.896 | 1.813 | 3.969 | 0.000 |
| marathi | mGPT | gold | character | 0.709 | 0.675 | 4.872 | 4.035 | 5.702 | 0.000 |
| marathi | mGPT | gold | token | 0.737 | 0.675 | 8.542 | 7.414 | 9.673 | 0.000 |
| marathi | sarvam-1 | article | word | 0.832 | 0.825 | 0.981 | 0.092 | 1.883 | 0.030 |
| marathi | sarvam-1 | article | character | 0.851 | 0.825 | 3.050 | 2.458 | 3.693 | 0.000 |
| marathi | sarvam-1 | article | token | 0.773 | 0.825 | -6.392 | -7.181 | -5.581 | 0.000 |
| marathi | sarvam-1 | gold | word | 0.835 | 0.827 | 1.162 | 0.215 | 2.127 | 0.007 |
| marathi | sarvam-1 | gold | character | 0.855 | 0.827 | 3.038 | 2.341 | 3.766 | 0.000 |
| marathi | sarvam-1 | gold | token | 0.792 | 0.827 | -3.990 | -4.837 | -3.102 | 0.000 |

### most even unit

| language | scorer | source | character | token | word | akshara | most_even |
|---|---|---|---|---|---|---|---|
| bengali | bloom-560m | article | 0.749 | 0.606 | 0.681 | 0.698 | token |
| bengali | bloom-560m | gold | 0.724 | 0.602 | 0.660 | 0.684 | token |
| bengali | gpt2-bengali | article | 0.807 | 0.958 | 0.694 | 0.742 | word |
| bengali | gpt2-bengali | gold | 0.802 | 0.931 | 0.687 | 0.750 | word |
| bengali | mGPT | article | 0.734 | 0.796 | 0.647 | 0.679 | word |
| bengali | mGPT | gold | 0.713 | 0.770 | 0.627 | 0.672 | word |
| bengali | sarvam-1 | article | 0.857 | 0.725 | 0.745 | 0.791 | token |
| bengali | sarvam-1 | gold | 0.847 | 0.746 | 0.745 | 0.796 | word |
| bengali | xglm-564M | article | 0.733 | 0.617 | 0.674 | 0.684 | token |
| bengali | xglm-564M | gold | 0.725 | 0.624 | 0.674 | 0.686 | token |
| hindi | bloom-560m | article | 0.812 | 0.793 | 0.909 | 0.801 | token |
| hindi | bloom-560m | gold | 0.812 | 0.795 | 0.894 | 0.798 | token |
| hindi | gpt2-hindi | article | 0.785 | 0.941 | 0.876 | 0.760 | akshara |
| hindi | gpt2-hindi | gold | 0.788 | 0.928 | 0.868 | 0.756 | akshara |
| hindi | mGPT | article | 0.804 | 0.892 | 0.898 | 0.787 | akshara |
| hindi | mGPT | gold | 0.797 | 0.877 | 0.887 | 0.770 | akshara |
| hindi | sarvam-1 | article | 0.924 | 0.907 | 0.999 | 0.909 | token |
| hindi | sarvam-1 | gold | 0.933 | 0.925 | 0.994 | 0.911 | akshara |
| hindi | xglm-564M | article | 0.820 | 0.796 | 0.930 | 0.807 | token |
| hindi | xglm-564M | gold | 0.840 | 0.818 | 0.935 | 0.817 | akshara |
| marathi | bloom-560m | article | 0.701 | 0.605 | 0.716 | 0.685 | token |
| marathi | bloom-560m | gold | 0.682 | 0.596 | 0.706 | 0.660 | token |
| marathi | mGPT | article | 0.714 | 0.738 | 0.705 | 0.688 | akshara |
| marathi | mGPT | gold | 0.709 | 0.737 | 0.694 | 0.675 | akshara |
| marathi | sarvam-1 | article | 0.851 | 0.773 | 0.832 | 0.825 | token |
| marathi | sarvam-1 | gold | 0.855 | 0.792 | 0.835 | 0.827 | token |

### length units

| scorer | dR2_aks_over_char | lo_aks | hi_aks | dR2_char_over_aks | lo_char | hi_char | frac_boot_aks_gt_char | language |
|---|---|---|---|---|---|---|---|---|
| bloom-560m | 0.004 | 0.003 | 0.005 | 0.001 | 0.000 | 0.001 | 1.000 | hindi |
| gpt2-hindi | 0.002 | 0.002 | 0.003 | 0.000 | 0.000 | 0.000 | 1.000 | hindi |
| mGPT | 0.002 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 | 1.000 | hindi |
| sarvam-1 | 0.006 | 0.005 | 0.007 | 0.000 | 0.000 | 0.001 | 1.000 | hindi |
| xglm-564M | 0.005 | 0.004 | 0.006 | 0.001 | 0.001 | 0.001 | 1.000 | hindi |
| bloom-560m | 0.004 | 0.003 | 0.005 | 0.002 | 0.001 | 0.003 | 1.000 | bengali |
| gpt2-bengali | 0.002 | 0.002 | 0.003 | 0.001 | 0.000 | 0.002 | 1.000 | bengali |
| mGPT | 0.001 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 1.000 | bengali |
| sarvam-1 | 0.006 | 0.005 | 0.007 | 0.003 | 0.002 | 0.004 | 1.000 | bengali |
| xglm-564M | 0.002 | 0.002 | 0.003 | 0.001 | 0.001 | 0.002 | 1.000 | bengali |
| bloom-560m | 0.001 | 0.000 | 0.001 | 0.001 | 0.001 | 0.002 | 0.000 | marathi |
| mGPT | 0.003 | 0.002 | 0.004 | 0.002 | 0.001 | 0.002 | 0.995 | marathi |
| sarvam-1 | 0.002 | 0.002 | 0.003 | 0.004 | 0.003 | 0.004 | 0.000 | marathi |

### agreement

| language | cv_per_akshara | cv_per_char | cv_per_token | cv_word_total |
|---|---|---|---|---|
| bengali | 0.745 | 0.771 | 0.439 | 0.697 |
| hindi | 0.708 | 0.733 | 0.490 | 0.752 |
| marathi | 0.679 | 0.686 | 0.509 | 0.712 |

### boundary

| language | scorer | source | pct | p_holm | cal_bias_pct | frac_words_split | pass |
|---|---|---|---|---|---|---|---|
| hindi | bloom-560m | article | 0.547 | 0.008 | 0.562 | 0.147 | False |
| hindi | bloom-560m | gold | 0.324 | 0.504 | 0.562 | 0.147 | False |
| hindi | gpt2-hindi | article | 2.428 | 0.000 | 0.631 | 0.899 | True |
| hindi | gpt2-hindi | gold | 2.014 | 0.000 | 0.631 | 0.899 | True |
| hindi | mGPT | article | 1.622 | 0.000 | 1.038 | 0.919 | True |
| hindi | mGPT | gold | 1.253 | 0.000 | 1.038 | 0.919 | True |
| hindi | sarvam-1 | article | 0.872 | 0.000 | 0.475 | 0.165 | True |
| hindi | sarvam-1 | gold | 0.848 | 0.003 | 0.475 | 0.165 | True |
| hindi | xglm-564M | article | 0.807 | 0.000 | 0.159 | 0.268 | True |
| hindi | xglm-564M | gold | 0.560 | 0.027 | 0.159 | 0.268 | True |
| bengali | bloom-560m | article | -0.324 | 0.298 | 1.673 | 0.271 | False |
| bengali | bloom-560m | gold | -0.148 | 1.000 | 1.673 | 0.271 | False |
| bengali | gpt2-bengali | article | 4.690 | 0.000 | 0.712 | 0.923 | True |
| bengali | gpt2-bengali | gold | 4.518 | 0.000 | 0.712 | 0.923 | True |
| bengali | mGPT | article | 4.450 | 0.000 | 1.155 | 0.950 | True |
| bengali | mGPT | gold | 4.270 | 0.000 | 1.155 | 0.950 | True |
| bengali | sarvam-1 | article | 0.478 | 0.006 | 1.317 | 0.430 | False |
| bengali | sarvam-1 | gold | 0.310 | 0.908 | 1.317 | 0.430 | False |
| bengali | xglm-564M | article | 0.731 | 0.000 | 0.658 | 0.552 | True |
| bengali | xglm-564M | gold | 0.438 | 0.120 | 0.658 | 0.552 | False |
| marathi | bloom-560m | article | 1.021 | 0.000 | 1.729 | 0.376 | False |
| marathi | bloom-560m | gold | 1.407 | 0.000 | 1.729 | 0.376 | False |
| marathi | mGPT | article | 5.016 | 0.000 | 1.509 | 0.986 | True |
| marathi | mGPT | gold | 4.450 | 0.000 | 1.509 | 0.986 | True |
| marathi | sarvam-1 | article | 1.360 | 0.000 | 0.882 | 0.448 | True |
| marathi | sarvam-1 | gold | 1.182 | 0.000 | 0.882 | 0.448 | True |

### split penalty

| language | model | ddS_all | lo_all | hi_all | ddS_clean | lo_clean | hi_clean | ddS_adjusted | lo_adjusted | hi_adjusted | signs_agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hindi | gpt2-hindi | 0.184 | 0.124 | 0.240 | 0.120 | 0.059 | 0.177 | 0.181 | 0.117 | 0.247 | True |
| hindi | mGPT | 0.128 | 0.084 | 0.168 | 0.106 | 0.062 | 0.147 | 0.087 | 0.038 | 0.132 | True |
| bengali | gpt2-bengali | -0.241 | -0.317 | -0.163 | -0.297 | -0.374 | -0.227 | -0.005 | -0.088 | 0.069 | True |
| bengali | mGPT | 0.040 | -0.040 | 0.116 | -0.005 | -0.081 | 0.070 | 0.156 | 0.072 | 0.234 | True |
| marathi | mGPT | -0.461 | -0.658 | -0.258 | -0.557 | -0.754 | -0.354 | -0.298 | -0.487 | -0.113 | True |

### split inside vs between

| language | scorer | aligned | crossing | pass |
|---|---|---|---|---|
| hindi | bloom-560m | 0.719 | 0.944 | True |
| hindi | sarvam-1 | 0.675 | 1.086 | True |
| hindi | xglm-564M | 0.491 | 0.860 | True |
| bengali | bloom-560m | 0.157 | 0.294 | True |
| bengali | sarvam-1 | 0.184 | 0.359 | True |
| bengali | xglm-564M | 0.326 | 0.488 | True |
| marathi | bloom-560m | 0.283 | 0.263 | False |
| marathi | sarvam-1 | 0.354 | 0.451 | True |

### tokenizer audit diagnosis

| tokenizer | bengali | gujarati | hindi | marathi | nepali | punjabi | tamil | telugu |
|---|---|---|---|---|---|---|---|---|
| BLOOM | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| Gemma-2 | learned subwords, many signs cut | script ~unknown: character/byte level | learned subwords | learned subwords | learned subwords | script ~unknown: character/byte level | learned subwords | script ~unknown: character/byte level |
| Gemma-3 | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| IndicBERT-v2 | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| Llama-3.2 | script ~unknown: character/byte level | script ~unknown: character/byte level | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Mistral-7B | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| MuRIL | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| OpenHathi | script ~unknown: character/byte level | script ~unknown: character/byte level | learned subwords | learned subwords | learned subwords | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Phi-3 | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Pythia (GPT-NeoX) | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Qwen2.5 | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Qwen3 | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| Sarvam-1 | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| XGLM | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| XLM-R | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |
| gpt-oss (o200k) | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords, many signs cut | learned subwords | learned subwords |
| gpt2 (English) | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| gpt2-bengali | rule cuts every vowel sign | script ~unknown: character/byte level | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | script ~unknown: character/byte level | rule cuts every vowel sign | script ~unknown: character/byte level |
| gpt2-hindi | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | script ~unknown: character/byte level | script ~unknown: character/byte level | script ~unknown: character/byte level |
| mGPT | rule cuts every vowel sign | script ~unknown: character/byte level | rule cuts every vowel sign | rule cuts every vowel sign | rule cuts every vowel sign | script ~unknown: character/byte level | rule cuts every vowel sign | rule cuts every vowel sign |
| mT5 | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords | learned subwords |

## Run environment

```
{
 "hindi": {
  "timestamp": "2026-11-20T13:15:05.500432",
  "python": "3.13.16",
  "torch": "2.11.0+cu130",
  "transformers": "5.16.1",
  "datasets": "2.21.0"
 },
 "bengali": {
  "timestamp": "2026-11-20T14:56:32.625238",
  "python": "3.13.16",
  "torch": "2.11.0+cu130",
  "transformers": "5.16.1",
  "datasets": "2.21.0"
 },
 "marathi": {
  "timestamp": "2026-11-20T15:59:28.461526",
  "python": "3.13.16",
  "torch": "2.11.0+cu130",
  "transformers": "5.16.1",
  "datasets": "2.21.0"
 }
}
```