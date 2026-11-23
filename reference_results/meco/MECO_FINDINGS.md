# MECO Hindi: reading-time analysis

Readers: 111 (hi_iiith, hi_iitk), passages: 12, analysed words: 2236. Main measure: gaze duration. Between-sample reliability: gaze r = 0.84, total r = 0.91, first_fix r = 0.47.

## A. Native ruler check

* Unique gain of akshara length over character length: +71.92 [+32.01, +100.69] (positive in 11/12 passages); character over akshara: +23.53 [-24.98, +57.42] (positive in 10/12). Regression: Nguyen & Arehalli's length x frequency formula. Without the interaction terms: akshara +39.45 vs character +33.16.

### All units side by side (word = whole word, no length information)

| unit | gain over a word-only model (frequency), x1000 | reading time per unit: CV vs akshara (> 0 = akshara more even) |
|---|---|---|
| word | 0 (reference) | -2.7% (p = 0.12) |
| character | +193.0 [+102.7, +262.0] (11/12) | -1.0% (p = 0.0016) |
| akshara | +241.4 [+181.0, +296.6] (11/12) | reference |
| token (gpt2-hindi) | +121.6 [+57.0, +183.9] (11/12) | +19.6% (p = 6e-20) |
| token (mGPT) | +149.7 [+94.6, +210.1] (11/12) | +16.0% (p = 6.3e-20) |
| token (bloom-560m) | +23.5 [-7.0, +54.3] (7/12) | -6.2% (p = 1.7e-07) |
| token (xglm-564M) | +29.6 [-11.8, +66.5] (8/12) | -2.8% (p = 0.00035) |
| token (sarvam-1) | +6.3 [-15.0, +27.4] (6/12) | -5.3% (p = 4.3e-06) |

### Word-level features (frequency, previous-word frequency, function word, punctuation, sentence-final, position in sentence, position in passage)

| comparison | gain x1000 [95% CI] | passages positive |
|---|---|---|
| all word-level features over intercept | +244.3 [+203.6, +278.1] | 12/12 |
| word feature: logfreq (drop-one) | +30.9 [+10.2, +48.6] | 10/12 |
| word feature: prev_logfreq (drop-one) | -1.2 [-3.5, +0.5] | 7/12 |
| word feature: func (drop-one) | +2.7 [+0.3, +5.6] | 9/12 |
| word feature: punct (drop-one) | +3.8 [-1.4, +9.0] | 8/12 |
| word feature: sent_final (drop-one) | +2.1 [-0.8, +5.2] | 8/12 |
| word feature: log_pos_sent (drop-one) | +0.2 [-3.2, +3.0] | 8/12 |
| word feature: log_pos_text (drop-one) | +13.7 [+8.2, +18.3] | 11/12 |
| character length over word-level features | +200.6 [+103.1, +272.3] | 11/12 |
| akshara length over word-level features | +254.1 [+190.8, +309.8] | 11/12 |
| visual width length over word-level features | +275.8 [+146.9, +367.9] | 11/12 |
| token (gpt2-hindi) length over word-level features | +130.5 [+65.2, +196.0] | 11/12 |
| token (mGPT) length over word-level features | +160.5 [+101.1, +228.3] | 11/12 |
| token (bloom-560m) length over word-level features | +41.9 [-0.8, +86.8] | 7/12 |
| token (xglm-564M) length over word-level features | +43.9 [-5.5, +93.7] | 8/12 |
| token (sarvam-1) length over word-level features | +16.8 [-8.9, +42.7] | 7/12 |
| akshara over word-level features + visual width | -2.3 [-7.0, +2.9] | 5/12 |
| visual width over word-level features + akshara | +19.5 [-55.5, +61.7] | 11/12 |
| akshara over word-level features + visual width + character | -1.8 [-6.9, +3.8] | 5/12 |
| akshara over word-level features + character | +75.7 [+38.2, +105.4] | 11/12 |
| character over word-level features + akshara | +22.2 [-25.1, +58.1] | 10/12 |
| word surprisal (gpt2-hindi) over word features + akshara length | +25.9 [+15.3, +39.0] | 11/12 |
| surprisal per akshara (gpt2-hindi) over word features + akshara length | +18.1 [+13.0, +24.3] | 12/12 |
| word surprisal (mGPT) over word features + akshara length | +29.1 [+14.2, +44.5] | 11/12 |
| surprisal per akshara (mGPT) over word features + akshara length | +24.1 [+17.8, +30.2] | 12/12 |
| word surprisal (bloom-560m) over word features + akshara length | +82.6 [+58.1, +117.2] | 12/12 |
| surprisal per akshara (bloom-560m) over word features + akshara length | +39.6 [+34.9, +45.1] | 12/12 |
| word surprisal (xglm-564M) over word features + akshara length | +87.8 [+55.5, +129.6] | 12/12 |
| surprisal per akshara (xglm-564M) over word features + akshara length | +45.2 [+36.5, +55.2] | 12/12 |
| word surprisal (sarvam-1) over word features + akshara length | +42.4 [+28.3, +56.7] | 11/12 |
| surprisal per akshara (sarvam-1) over word features + akshara length | +22.1 [+11.0, +30.2] | 11/12 |

### Replication within each Hindi sample (independent readers)

| sample | readers | comparison | gain x1000 [95% CI] | passages positive |
|---|---|---|---|---|
| hi_iiith | 54 | akshara over word-level features + character | +71.2 [+31.4, +102.3] | 11/12 |
| hi_iiith | 54 | character over word-level features + akshara | +30.0 [-14.7, +65.2] | 10/12 |
| hi_iiith | 54 | akshara over word-level features + visual width | -1.3 [-7.1, +4.9] | 4/12 |
| hi_iiith | 54 | word surprisal (gpt2-hindi) over word features + akshara length | +21.1 [+11.1, +32.0] | 10/12 |
| hi_iiith | 54 | word surprisal (mGPT) over word features + akshara length | +27.0 [+12.5, +40.9] | 10/12 |
| hi_iiith | 54 | word surprisal (bloom-560m) over word features + akshara length | +66.8 [+48.4, +88.7] | 12/12 |
| hi_iiith | 54 | word surprisal (xglm-564M) over word features + akshara length | +71.7 [+49.6, +97.6] | 12/12 |
| hi_iiith | 54 | word surprisal (sarvam-1) over word features + akshara length | +38.4 [+26.0, +51.8] | 11/12 |
| hi_iitk | 57 | akshara over word-level features + character | +51.6 [+28.2, +69.6] | 11/12 |
| hi_iitk | 57 | character over word-level features + akshara | +4.0 [-27.9, +25.0] | 9/12 |
| hi_iitk | 57 | akshara over word-level features + visual width | -3.2 [-12.0, +3.1] | 6/12 |
| hi_iitk | 57 | word surprisal (gpt2-hindi) over word features + akshara length | +20.8 [+12.3, +30.9] | 12/12 |
| hi_iitk | 57 | word surprisal (mGPT) over word features + akshara length | +19.0 [+8.9, +29.9] | 11/12 |
| hi_iitk | 57 | word surprisal (bloom-560m) over word features + akshara length | +71.6 [+47.3, +102.9] | 12/12 |
| hi_iitk | 57 | word surprisal (xglm-564M) over word features + akshara length | +75.3 [+43.3, +121.1] | 11/12 |
| hi_iitk | 57 | word surprisal (sarvam-1) over word features + akshara length | +32.1 [+21.1, +42.9] | 11/12 |

## B. Surprisal predictive power (held-out log-lik gain per word x1000, over length + frequency)

| model | tokenizer | raw | corrected | corrected − raw |
|---|---|---|---|---|
| gpt2-hindi | rule (cuts marks) | +19.15 [+8.03, +30.67] | +19.56 [+7.50, +32.58] | +0.41 [-0.87, +1.83] |
| mGPT | rule (cuts marks) | +18.07 [+3.07, +32.70] | +18.37 [+1.54, +33.57] | +0.30 [-1.02, +1.71] |
| bloom-560m | learned subwords | +93.51 [+67.57, +127.78] | +93.41 [+67.58, +126.15] | -0.09 [-0.86, +0.62] |
| xglm-564M | learned subwords | +97.04 [+61.37, +141.03] | +97.20 [+63.85, +142.84] | +0.16 [-2.46, +2.76] |
| sarvam-1 | learned subwords | +52.76 [+37.95, +68.34] | +53.35 [+39.68, +67.45] | +0.59 [-1.77, +2.96] |

## C. Split words (residual ms = observed − predicted)

| model | fit | unsplit | split, aligned | split, crosses akshara | note |
|---|---|---|---|---|---|
| gpt2-hindi | UNSPLIT | -0.4 [-5.0, +4.0] (n=227) | — | +22.7 [+15.7, +31.6] (n=2009) | UNSPLIT fit is trained on only 10% of words: it extrapolates, so read the FULL fit |
| gpt2-hindi | FULL | -5.0 [-9.5, +0.0] (n=227) | — | +0.4 [-5.2, +7.0] (n=2009) |  |
| mGPT | UNSPLIT | -0.3 [-5.8, +5.0] (n=187) | +34.0 [+8.5, +77.6] (n=40) | +74.0 [+58.1, +95.2] (n=2009) | UNSPLIT fit is trained on only 8% of words: it extrapolates, so read the FULL fit |
| mGPT | FULL | +0.0 [-4.1, +4.2] (n=187) | -24.5 [-41.6, -9.5] (n=40) | +0.2 [-5.3, +6.9] (n=2009) |  |
| bloom-560m | UNSPLIT | -0.0 [-4.4, +5.0] (n=1930) | +3.1 [-24.5, +39.0] (n=101) | +2.1 [-17.9, +28.6] (n=205) |  |
| bloom-560m | FULL | +0.8 [-3.3, +5.7] (n=1930) | -4.2 [-29.7, +24.5] (n=101) | -4.5 [-24.1, +19.4] (n=205) |  |
| xglm-564M | UNSPLIT | +0.1 [-3.8, +4.1] (n=1628) | -31.0 [-43.7, -13.6] (n=331) | +4.5 [-15.6, +26.7] (n=277) |  |
| xglm-564M | FULL | +0.8 [-3.2, +5.1] (n=1628) | -18.3 [-29.6, -2.8] (n=331) | +17.9 [-2.6, +42.0] (n=277) |  |
| sarvam-1 | UNSPLIT | +0.1 [-5.0, +6.2] (n=1899) | -12.3 [-33.4, +8.7] (n=172) | -15.1 [-45.5, +22.7] (n=165) |  |
| sarvam-1 | FULL | +1.2 [-3.6, +6.8] (n=1899) | -4.4 [-21.3, +12.8] (n=172) | -8.1 [-37.4, +30.9] (n=165) |  |

Log-likelihood gains are leave-one-passage-out; CIs bootstrap over passages (v9; word-level CIs are in the CSVs as ci_lo_word/ci_hi_word). With 12 passages, 'positive in k/12 passages' shows how consistent an effect is across texts.