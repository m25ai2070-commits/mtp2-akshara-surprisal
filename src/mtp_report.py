import os, glob, json, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mtp_common import SPECS, LANG_SCRIPT, RULE_SCORERS

LANGS = ("hindi", "bengali", "marathi")
KEY_METRICS = ["logratio_tok_vs_aks", "logratio_char_vs_aks", "logratio_grain_random_excess",
               "logratio_grain_randcut_shufexcess"]
TOL = {"stat": 0.002, "ddS": 0.01, "split": 0.002, "audit": 0.01, "meco": 0.5}   # |new - reference| allowed
pct = lambda v: (np.exp(v) - 1) * 100                   # log CV ratio -> "% more even"

REFERENCE = [   # (kind, language, scorer/tokenizer, source, item, value) from the submitted runs
    ('stat', 'hindi', 'bloom-560m', 'article', 'logratio_tok_vs_aks', -0.00527),
    ('stat', 'hindi', 'bloom-560m', 'article', 'logratio_char_vs_aks', 0.01351),
    ('stat', 'hindi', 'bloom-560m', 'article', 'logratio_grain_random_excess', 0.00545),
    ('stat', 'hindi', 'bloom-560m', 'article', 'logratio_grain_randcut_shufexcess', 0.00557),
    ('stat', 'hindi', 'bloom-560m', 'gold', 'logratio_tok_vs_aks', 0.00185),
    ('stat', 'hindi', 'bloom-560m', 'gold', 'logratio_char_vs_aks', 0.01753),
    ('stat', 'hindi', 'bloom-560m', 'gold', 'logratio_grain_random_excess', 0.00323),
    ('stat', 'hindi', 'bloom-560m', 'gold', 'logratio_grain_randcut_shufexcess', 0.0038),
    ('stat', 'hindi', 'gpt2-hindi', 'article', 'logratio_tok_vs_aks', 0.20659),
    ('stat', 'hindi', 'gpt2-hindi', 'article', 'logratio_char_vs_aks', 0.03039),
    ('stat', 'hindi', 'gpt2-hindi', 'article', 'logratio_grain_random_excess', 0.02399),
    ('stat', 'hindi', 'gpt2-hindi', 'article', 'logratio_grain_randcut_shufexcess', 0.00951),
    ('stat', 'hindi', 'gpt2-hindi', 'gold', 'logratio_tok_vs_aks', 0.19697),
    ('stat', 'hindi', 'gpt2-hindi', 'gold', 'logratio_char_vs_aks', 0.0407),
    ('stat', 'hindi', 'gpt2-hindi', 'gold', 'logratio_grain_random_excess', 0.01994),
    ('stat', 'hindi', 'gpt2-hindi', 'gold', 'logratio_grain_randcut_shufexcess', 0.00396),
    ('stat', 'hindi', 'mGPT', 'article', 'logratio_tok_vs_aks', 0.12279),
    ('stat', 'hindi', 'mGPT', 'article', 'logratio_char_vs_aks', 0.01966),
    ('stat', 'hindi', 'mGPT', 'article', 'logratio_grain_random_excess', 0.01609),
    ('stat', 'hindi', 'mGPT', 'article', 'logratio_grain_randcut_shufexcess', 7e-05),
    ('stat', 'hindi', 'mGPT', 'gold', 'logratio_tok_vs_aks', 0.12599),
    ('stat', 'hindi', 'mGPT', 'gold', 'logratio_char_vs_aks', 0.03466),
    ('stat', 'hindi', 'mGPT', 'gold', 'logratio_grain_random_excess', 0.01245),
    ('stat', 'hindi', 'mGPT', 'gold', 'logratio_grain_randcut_shufexcess', -0.00561),
    ('ddS', 'hindi', 'gpt2-hindi', '', 'all', 0.1838),
    ('ddS', 'hindi', 'mGPT', '', 'all', 0.1275),
    ('split', 'hindi', 'bloom-560m', '', 'frac_words_split', 0.14664),
    ('split', 'hindi', 'gpt2-hindi', '', 'frac_words_split', 0.89861),
    ('split', 'hindi', 'mGPT', '', 'frac_words_split', 0.91866),
    ('stat', 'bengali', 'bloom-560m', 'article', 'logratio_tok_vs_aks', -0.13661),
    ('stat', 'bengali', 'bloom-560m', 'article', 'logratio_char_vs_aks', 0.06767),
    ('stat', 'bengali', 'bloom-560m', 'article', 'logratio_grain_random_excess', -0.00325),
    ('stat', 'bengali', 'bloom-560m', 'article', 'logratio_grain_randcut_shufexcess', 0.001),
    ('stat', 'bengali', 'bloom-560m', 'gold', 'logratio_tok_vs_aks', -0.1226),
    ('stat', 'bengali', 'bloom-560m', 'gold', 'logratio_char_vs_aks', 0.05318),
    ('stat', 'bengali', 'bloom-560m', 'gold', 'logratio_grain_random_excess', -0.00149),
    ('stat', 'bengali', 'bloom-560m', 'gold', 'logratio_grain_randcut_shufexcess', 0.00077),
    ('stat', 'bengali', 'gpt2-bengali', 'article', 'logratio_tok_vs_aks', 0.25196),
    ('stat', 'bengali', 'gpt2-bengali', 'article', 'logratio_char_vs_aks', 0.07979),
    ('stat', 'bengali', 'gpt2-bengali', 'article', 'logratio_grain_random_excess', 0.04583),
    ('stat', 'bengali', 'gpt2-bengali', 'article', 'logratio_grain_randcut_shufexcess', 0.03129),
    ('stat', 'bengali', 'gpt2-bengali', 'gold', 'logratio_tok_vs_aks', 0.2077),
    ('stat', 'bengali', 'gpt2-bengali', 'gold', 'logratio_char_vs_aks', 0.06294),
    ('stat', 'bengali', 'gpt2-bengali', 'gold', 'logratio_grain_random_excess', 0.04419),
    ('stat', 'bengali', 'gpt2-bengali', 'gold', 'logratio_grain_randcut_shufexcess', 0.02775),
    ('stat', 'bengali', 'mGPT', 'article', 'logratio_tok_vs_aks', 0.15401),
    ('stat', 'bengali', 'mGPT', 'article', 'logratio_char_vs_aks', 0.07323),
    ('stat', 'bengali', 'mGPT', 'article', 'logratio_grain_random_excess', 0.04354),
    ('stat', 'bengali', 'mGPT', 'article', 'logratio_grain_randcut_shufexcess', 0.02755),
    ('stat', 'bengali', 'mGPT', 'gold', 'logratio_tok_vs_aks', 0.13143),
    ('stat', 'bengali', 'mGPT', 'gold', 'logratio_char_vs_aks', 0.0563),
    ('stat', 'bengali', 'mGPT', 'gold', 'logratio_grain_random_excess', 0.04181),
    ('stat', 'bengali', 'mGPT', 'gold', 'logratio_grain_randcut_shufexcess', 0.02437),
    ('ddS', 'bengali', 'gpt2-bengali', '', 'all', -0.2407),
    ('ddS', 'bengali', 'gpt2-bengali', '', 'adjusted', -0.0047),
    ('ddS', 'bengali', 'mGPT', '', 'all', 0.0404),
    ('ddS', 'bengali', 'mGPT', '', 'adjusted', 0.1559),
    ('split', 'bengali', 'bloom-560m', '', 'frac_words_split', 0.27102),
    ('split', 'bengali', 'gpt2-bengali', '', 'frac_words_split', 0.92312),
    ('split', 'bengali', 'mGPT', '', 'frac_words_split', 0.95002),
    ('stat', 'marathi', 'bloom-560m', 'article', 'logratio_tok_vs_aks', -0.12409),
    ('stat', 'marathi', 'bloom-560m', 'article', 'logratio_char_vs_aks', 0.0248),
    ('stat', 'marathi', 'bloom-560m', 'article', 'logratio_grain_random_excess', 0.01016),
    ('stat', 'marathi', 'bloom-560m', 'article', 'logratio_grain_randcut_shufexcess', 0.01349),
    ('stat', 'marathi', 'bloom-560m', 'gold', 'logratio_tok_vs_aks', -0.10258),
    ('stat', 'marathi', 'bloom-560m', 'gold', 'logratio_char_vs_aks', 0.03156),
    ('stat', 'marathi', 'bloom-560m', 'gold', 'logratio_grain_random_excess', 0.01397),
    ('stat', 'marathi', 'bloom-560m', 'gold', 'logratio_grain_randcut_shufexcess', 0.0194),
    ('stat', 'marathi', 'mGPT', 'article', 'logratio_tok_vs_aks', 0.06417),
    ('stat', 'marathi', 'mGPT', 'article', 'logratio_char_vs_aks', 0.0354),
    ('stat', 'marathi', 'mGPT', 'article', 'logratio_grain_random_excess', 0.04894),
    ('stat', 'marathi', 'mGPT', 'article', 'logratio_grain_randcut_shufexcess', 0.03457),
    ('stat', 'marathi', 'mGPT', 'gold', 'logratio_tok_vs_aks', 0.08197),
    ('stat', 'marathi', 'mGPT', 'gold', 'logratio_char_vs_aks', 0.04757),
    ('stat', 'marathi', 'mGPT', 'gold', 'logratio_grain_random_excess', 0.04354),
    ('stat', 'marathi', 'mGPT', 'gold', 'logratio_grain_randcut_shufexcess', 0.0271),
    ('ddS', 'marathi', 'mGPT', '', 'all', -0.4608),
    ('ddS', 'marathi', 'mGPT', '', 'adjusted', -0.2981),
    ('split', 'marathi', 'bloom-560m', '', 'frac_words_split', 0.37635),
    ('split', 'marathi', 'mGPT', '', 'frac_words_split', 0.98565),
    ('audit', 'hindi', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'gpt2 (English)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'gpt2-hindi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'gpt2-bengali', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'mGPT', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'BLOOM', '', 'frac_marks_token_initial', 0.0804),
    ('audit', 'marathi', 'BLOOM', '', 'frac_marks_token_initial', 0.1349),
    ('audit', 'nepali', 'BLOOM', '', 'frac_marks_token_initial', 0.1132),
    ('audit', 'bengali', 'BLOOM', '', 'frac_marks_token_initial', 0.1204),
    ('audit', 'gujarati', 'BLOOM', '', 'frac_marks_token_initial', 0.1475),
    ('audit', 'punjabi', 'BLOOM', '', 'frac_marks_token_initial', 0.0991),
    ('audit', 'tamil', 'BLOOM', '', 'frac_marks_token_initial', 0.1558),
    ('audit', 'telugu', 'BLOOM', '', 'frac_marks_token_initial', 0.2294),
    ('audit', 'hindi', 'XGLM', '', 'frac_marks_token_initial', 0.0806),
    ('audit', 'marathi', 'XGLM', '', 'frac_marks_token_initial', 0.1171),
    ('audit', 'nepali', 'XGLM', '', 'frac_marks_token_initial', 0.0754),
    ('audit', 'bengali', 'XGLM', '', 'frac_marks_token_initial', 0.1192),
    ('audit', 'gujarati', 'XGLM', '', 'frac_marks_token_initial', 0.1356),
    ('audit', 'punjabi', 'XGLM', '', 'frac_marks_token_initial', 0.1515),
    ('audit', 'tamil', 'XGLM', '', 'frac_marks_token_initial', 0.0938),
    ('audit', 'telugu', 'XGLM', '', 'frac_marks_token_initial', 0.1263),
    ('audit', 'hindi', 'mT5', '', 'frac_marks_token_initial', 0.1031),
    ('audit', 'marathi', 'mT5', '', 'frac_marks_token_initial', 0.1537),
    ('audit', 'nepali', 'mT5', '', 'frac_marks_token_initial', 0.1284),
    ('audit', 'bengali', 'mT5', '', 'frac_marks_token_initial', 0.1692),
    ('audit', 'gujarati', 'mT5', '', 'frac_marks_token_initial', 0.1743),
    ('audit', 'punjabi', 'mT5', '', 'frac_marks_token_initial', 0.2265),
    ('audit', 'tamil', 'mT5', '', 'frac_marks_token_initial', 0.0908),
    ('audit', 'telugu', 'mT5', '', 'frac_marks_token_initial', 0.1586),
    ('audit', 'hindi', 'XLM-R', '', 'frac_marks_token_initial', 0.055),
    ('audit', 'marathi', 'XLM-R', '', 'frac_marks_token_initial', 0.0879),
    ('audit', 'nepali', 'XLM-R', '', 'frac_marks_token_initial', 0.0496),
    ('audit', 'bengali', 'XLM-R', '', 'frac_marks_token_initial', 0.0969),
    ('audit', 'gujarati', 'XLM-R', '', 'frac_marks_token_initial', 0.0967),
    ('audit', 'punjabi', 'XLM-R', '', 'frac_marks_token_initial', 0.0985),
    ('audit', 'tamil', 'XLM-R', '', 'frac_marks_token_initial', 0.0502),
    ('audit', 'telugu', 'XLM-R', '', 'frac_marks_token_initial', 0.0793),
    ('audit', 'hindi', 'MuRIL', '', 'frac_marks_token_initial', 0.0303),
    ('audit', 'marathi', 'MuRIL', '', 'frac_marks_token_initial', 0.054),
    ('audit', 'nepali', 'MuRIL', '', 'frac_marks_token_initial', 0.0359),
    ('audit', 'bengali', 'MuRIL', '', 'frac_marks_token_initial', 0.0281),
    ('audit', 'gujarati', 'MuRIL', '', 'frac_marks_token_initial', 0.0803),
    ('audit', 'punjabi', 'MuRIL', '', 'frac_marks_token_initial', 0.0664),
    ('audit', 'tamil', 'MuRIL', '', 'frac_marks_token_initial', 0.0236),
    ('audit', 'telugu', 'MuRIL', '', 'frac_marks_token_initial', 0.0667),
    ('audit', 'hindi', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0232),
    ('audit', 'marathi', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0302),
    ('audit', 'nepali', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0322),
    ('audit', 'bengali', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0328),
    ('audit', 'gujarati', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0352),
    ('audit', 'punjabi', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0253),
    ('audit', 'tamil', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0259),
    ('audit', 'telugu', 'IndicBERT-v2', '', 'frac_marks_token_initial', 0.0257),
    ('audit', 'hindi', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Pythia (GPT-NeoX)', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Phi-3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Qwen2.5', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Qwen3', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.1894),
    ('audit', 'marathi', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.3337),
    ('audit', 'nepali', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.2578),
    ('audit', 'bengali', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.3289),
    ('audit', 'gujarati', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.3),
    ('audit', 'punjabi', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.5172),
    ('audit', 'tamil', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.3423),
    ('audit', 'telugu', 'gpt-oss (o200k)', '', 'frac_marks_token_initial', 0.4231),
    ('audit', 'hindi', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0435),
    ('audit', 'marathi', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0689),
    ('audit', 'nepali', 'Sarvam-1', '', 'frac_marks_token_initial', 0.122),
    ('audit', 'bengali', 'Sarvam-1', '', 'frac_marks_token_initial', 0.1518),
    ('audit', 'gujarati', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0696),
    ('audit', 'punjabi', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0933),
    ('audit', 'tamil', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0849),
    ('audit', 'telugu', 'Sarvam-1', '', 'frac_marks_token_initial', 0.0988),
    ('audit', 'hindi', 'OpenHathi', '', 'frac_marks_token_initial', 0.0726),
    ('audit', 'marathi', 'OpenHathi', '', 'frac_marks_token_initial', 0.1885),
    ('audit', 'nepali', 'OpenHathi', '', 'frac_marks_token_initial', 0.1226),
    ('audit', 'bengali', 'OpenHathi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'OpenHathi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'OpenHathi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'OpenHathi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'OpenHathi', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Llama-3.2', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'hindi', 'Gemma-2', '', 'frac_marks_token_initial', 0.2017),
    ('audit', 'marathi', 'Gemma-2', '', 'frac_marks_token_initial', 0.3093),
    ('audit', 'nepali', 'Gemma-2', '', 'frac_marks_token_initial', 0.2447),
    ('audit', 'bengali', 'Gemma-2', '', 'frac_marks_token_initial', 0.5948),
    ('audit', 'gujarati', 'Gemma-2', '', 'frac_marks_token_initial', 0.5168),
    ('audit', 'punjabi', 'Gemma-2', '', 'frac_marks_token_initial', 0.6791),
    ('audit', 'tamil', 'Gemma-2', '', 'frac_marks_token_initial', 0.3033),
    ('audit', 'telugu', 'Gemma-2', '', 'frac_marks_token_initial', 0.447),
    ('audit', 'hindi', 'Gemma-3', '', 'frac_marks_token_initial', 0.0311),
    ('audit', 'marathi', 'Gemma-3', '', 'frac_marks_token_initial', 0.0563),
    ('audit', 'nepali', 'Gemma-3', '', 'frac_marks_token_initial', 0.0838),
    ('audit', 'bengali', 'Gemma-3', '', 'frac_marks_token_initial', 0.0796),
    ('audit', 'gujarati', 'Gemma-3', '', 'frac_marks_token_initial', 0.1472),
    ('audit', 'punjabi', 'Gemma-3', '', 'frac_marks_token_initial', 0.4294),
    ('audit', 'tamil', 'Gemma-3', '', 'frac_marks_token_initial', 0.0868),
    ('audit', 'telugu', 'Gemma-3', '', 'frac_marks_token_initial', 0.1358),
    ('audit', 'hindi', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'marathi', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'nepali', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'bengali', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'gujarati', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'punjabi', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'tamil', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
    ('audit', 'telugu', 'Mistral-7B', '', 'frac_marks_token_initial', 1.0),
]


REFERENCE_V9 = [   # recorded from the clean v9 run (2026-11-20): extra scorers and MECO. For that run they match by
                   # construction; for any later reproduction they are an independent check. MECO: gaze duration,
                   # held-out log-likelihood gain x1000 (leave-one-passage-out).
    ('stat', 'hindi', 'xglm-564M', 'article', 'logratio_tok_vs_aks', -0.01172),
    ('stat', 'hindi', 'xglm-564M', 'article', 'logratio_char_vs_aks', 0.01495),
    ('stat', 'hindi', 'xglm-564M', 'article', 'logratio_grain_random_excess', 0.00804),
    ('stat', 'hindi', 'xglm-564M', 'article', 'logratio_grain_randcut_shufexcess', 0.00734),
    ('stat', 'hindi', 'xglm-564M', 'gold', 'logratio_tok_vs_aks', 0.00464),
    ('stat', 'hindi', 'xglm-564M', 'gold', 'logratio_char_vs_aks', 0.02746),
    ('stat', 'hindi', 'xglm-564M', 'gold', 'logratio_grain_random_excess', 0.00558),
    ('stat', 'hindi', 'xglm-564M', 'gold', 'logratio_grain_randcut_shufexcess', 0.00404),
    ('split', 'hindi', 'xglm-564M', '', 'frac_words_split', 0.26825),
    ('stat', 'hindi', 'sarvam-1', 'article', 'logratio_tok_vs_aks', 0.00124),
    ('stat', 'hindi', 'sarvam-1', 'article', 'logratio_char_vs_aks', 0.01583),
    ('stat', 'hindi', 'sarvam-1', 'article', 'logratio_grain_random_excess', 0.00868),
    ('stat', 'hindi', 'sarvam-1', 'article', 'logratio_grain_randcut_shufexcess', 0.00818),
    ('stat', 'hindi', 'sarvam-1', 'gold', 'logratio_tok_vs_aks', 0.02076),
    ('stat', 'hindi', 'sarvam-1', 'gold', 'logratio_char_vs_aks', 0.02593),
    ('stat', 'hindi', 'sarvam-1', 'gold', 'logratio_grain_random_excess', 0.00845),
    ('stat', 'hindi', 'sarvam-1', 'gold', 'logratio_grain_randcut_shufexcess', 0.00597),
    ('split', 'hindi', 'sarvam-1', '', 'frac_words_split', 0.16493),
    ('stat', 'bengali', 'xglm-564M', 'article', 'logratio_tok_vs_aks', -0.09891),
    ('stat', 'bengali', 'xglm-564M', 'article', 'logratio_char_vs_aks', 0.06502),
    ('stat', 'bengali', 'xglm-564M', 'article', 'logratio_grain_random_excess', 0.00728),
    ('stat', 'bengali', 'xglm-564M', 'article', 'logratio_grain_randcut_shufexcess', 0.00593),
    ('stat', 'bengali', 'xglm-564M', 'gold', 'logratio_tok_vs_aks', -0.08907),
    ('stat', 'bengali', 'xglm-564M', 'gold', 'logratio_char_vs_aks', 0.05471),
    ('stat', 'bengali', 'xglm-564M', 'gold', 'logratio_grain_random_excess', 0.00437),
    ('stat', 'bengali', 'xglm-564M', 'gold', 'logratio_grain_randcut_shufexcess', 0.00173),
    ('split', 'bengali', 'xglm-564M', '', 'frac_words_split', 0.55202),
    ('stat', 'bengali', 'sarvam-1', 'article', 'logratio_tok_vs_aks', -0.08227),
    ('stat', 'bengali', 'sarvam-1', 'article', 'logratio_char_vs_aks', 0.07527),
    ('stat', 'bengali', 'sarvam-1', 'article', 'logratio_grain_random_excess', 0.00476),
    ('stat', 'bengali', 'sarvam-1', 'article', 'logratio_grain_randcut_shufexcess', 0.00283),
    ('stat', 'bengali', 'sarvam-1', 'gold', 'logratio_tok_vs_aks', -0.05811),
    ('stat', 'bengali', 'sarvam-1', 'gold', 'logratio_char_vs_aks', 0.05761),
    ('stat', 'bengali', 'sarvam-1', 'gold', 'logratio_grain_random_excess', 0.00309),
    ('stat', 'bengali', 'sarvam-1', 'gold', 'logratio_grain_randcut_shufexcess', 0.00122),
    ('split', 'bengali', 'sarvam-1', '', 'frac_words_split', 0.42963),
    ('stat', 'marathi', 'sarvam-1', 'article', 'logratio_tok_vs_aks', -0.06606),
    ('stat', 'marathi', 'sarvam-1', 'article', 'logratio_char_vs_aks', 0.03004),
    ('stat', 'marathi', 'sarvam-1', 'article', 'logratio_grain_random_excess', 0.01351),
    ('stat', 'marathi', 'sarvam-1', 'article', 'logratio_grain_randcut_shufexcess', 0.01623),
    ('stat', 'marathi', 'sarvam-1', 'gold', 'logratio_tok_vs_aks', -0.04072),
    ('stat', 'marathi', 'sarvam-1', 'gold', 'logratio_char_vs_aks', 0.02992),
    ('stat', 'marathi', 'sarvam-1', 'gold', 'logratio_grain_random_excess', 0.01175),
    ('stat', 'marathi', 'sarvam-1', 'gold', 'logratio_grain_randcut_shufexcess', 0.0168),
    ('split', 'marathi', 'sarvam-1', '', 'frac_words_split', 0.44809),
    ('meco', 'hindi', '', 'features', 'akshara over word-level features + character', 75.75),
    ('meco', 'hindi', '', 'features', 'character over word-level features + akshara', 22.25),
    ('meco', 'hindi', '', 'features', 'akshara over word-level features + visual width', -2.29),
    ('meco', 'hindi', '', 'features', 'word surprisal (gpt2-hindi) over word features + akshara length', 25.93),
    ('meco', 'hindi', '', 'features', 'word surprisal (mGPT) over word features + akshara length', 29.1),
    ('meco', 'hindi', '', 'features', 'word surprisal (bloom-560m) over word features + akshara length', 82.63),
    ('meco', 'hindi', '', 'features', 'word surprisal (xglm-564M) over word features + akshara length', 87.84),
    ('meco', 'hindi', '', 'features', 'word surprisal (sarvam-1) over word features + akshara length', 42.41),
    ('meco', 'hindi', 'gpt2-hindi', 'surprisal', 'raw', 19.15),
    ('meco', 'hindi', 'mGPT', 'surprisal', 'raw', 18.07),
    ('meco', 'hindi', 'bloom-560m', 'surprisal', 'raw', 93.51),
    ('meco', 'hindi', 'xglm-564M', 'surprisal', 'raw', 97.04),
    ('meco', 'hindi', 'sarvam-1', 'surprisal', 'raw', 52.76),
]


def _read(path, **kw):
    for p in (path, path + ".gz"):          # large result files may be stored gzipped (GitHub size limit)
        if os.path.exists(p):
            return pd.read_csv(p, **kw)
    return None

def load_results(base_dir, langs=LANGS):
    R = {}
    for lang in langs:
        d = os.path.join(base_dir, f"run_{lang}", "results")
        if not os.path.exists(os.path.join(d, "rq2_stats_main.csv")):
            continue
        r = {k: _read(os.path.join(d, f"rq2_{k}.csv")) for k in
             ["stats_main", "cross_scorer_ddS", "cross_scorer_agreement", "length_units_ci",
              "grain_calibration", "split_residuals"]}
        w = _read(os.path.join(d, "rq2_per_word.csv"), encoding="utf-8-sig",
                  usecols=["scorer", "variant", "n_tok", "word", "text_id", "source", "S", "n_char", "n_aks"])
        w = w[w.variant == "main"].copy()
        r["words"] = w                                   # v9: per-word values for the scale control
        mk = SPECS[LANG_SCRIPT[lang]].mark_re
        w["has_mark"] = w.word.map(lambda x: bool(mk.search(x)))
        r["splitting"] = pd.DataFrame([dict(scorer=sc, tokens_per_word=g.n_tok.mean(),
                                            frac_words_split=(g.n_tok > 1).mean(),
                                            frac_split_with_mark=(g[g.has_mark].n_tok > 1).mean())
                                       for sc, g in w.groupby("scorer")])
        pt = _read(os.path.join(d, "rq2_per_text.csv"), encoding="utf-8-sig",
                   usecols=lambda c: c in {"scorer", "source", "variant", "text_id", "cv_word_total",
                                           "cv_per_akshara", "cv_per_char", "cv_per_token"})
        r["per_text"] = pt[pt.variant == "main"] if pt is not None else None
        meta = os.path.join(d, "run_meta.json")
        r["meta"] = json.load(open(meta)) if os.path.exists(meta) else {}
        R[lang] = r
    audit = _read(os.path.join(base_dir, "audit", "tokenizer_audit.csv"))
    return R, audit

def _md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                 ("" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j]); adj[j] = min(1.0, run)
    return adj

def unit_comparison(R, n_boot=2000, seed=0):
   
    from scipy import stats
    rng = np.random.default_rng(seed); rows = []
    units = {"word": "cv_word_total", "character": "cv_per_char", "token": "cv_per_token"}
    for l, r in R.items():
        pt = r.get("per_text")
        if pt is None: continue
        for (sc, src), g in pt.groupby(["scorer", "source"]):
            for u, col in units.items():
                v = np.log(g[col] / g["cv_per_akshara"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
                rows.append(dict(language=l, scorer=sc, source=src, unit=u, n_texts=len(v),
                                 mean_cv_unit=g[col].mean(), mean_cv_akshara=g["cv_per_akshara"].mean(),
                                 pct=pct(v.mean()), ci_lo=pct(np.percentile(bs, 2.5)), ci_hi=pct(np.percentile(bs, 97.5)),
                                 p=stats.wilcoxon(v).pvalue if (v != 0).any() else 1.0))
    t = pd.DataFrame(rows)
    if len(t): t["p_holm"] = _holm(t.p)
    return t

def scale_control(R, n_boot=2000, seed=0):
    
    rng = np.random.default_rng(seed); rows = []; B = np.round(np.arange(0, 1.2001, 0.05), 2)
    for l, r in R.items():
        w = r.get("words")
        if w is None or not len(w): continue
        w = w[w.S > 0]
        gam = float(np.log(w.n_aks).std() / np.log(w.n_char).std())
        for (sc, src), g in w.groupby(["scorer", "source"]):
            tid = pd.factorize(g.text_id)[0]; nt = tid.max() + 1
            cnt = np.bincount(tid, minlength=nt).astype(float); S = g.S.to_numpy(float)
            def cvs(n, b):
                x = S / n ** b; m = np.bincount(tid, x, nt) / cnt
                v = np.bincount(tid, x * x, nt) / cnt - m * m
                return np.sqrt(np.maximum(v, 0)) / m
            na, nc = g.n_aks.to_numpy(float), g.n_char.to_numpy(float)
            ok = cnt >= 5
            ma = [np.nanmean(cvs(na, b)[ok]) for b in B]; mc = [np.nanmean(cvs(nc, b)[ok]) for b in B]
            ba, bc = B[int(np.nanargmin(ma))], B[int(np.nanargmin(mc))]
            def _pct(c_char, c_aks):
                v = np.log(c_char / c_aks)[ok]; v = v[np.isfinite(v)]
                bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
                f = lambda z: (np.exp(z) - 1) * 100
                return f(v.mean()), f(np.percentile(bs, 2.5)), f(np.percentile(bs, 97.5))
            p1 = _pct(cvs(nc, 1.0), cvs(na, 1.0)); pg = _pct(cvs(nc, gam), cvs(na, 1.0))
            pb = _pct(cvs(nc, bc), cvs(na, ba))
            rows.append(dict(language=l, scorer=sc, source=src, n_texts=int(ok.sum()), gamma=gam,
                             best_exp_akshara=ba, best_exp_char=bc,
                             pct_exp1=p1[0], ci_lo_exp1=p1[1], ci_hi_exp1=p1[2],
                             pct_gamma=pg[0], ci_lo_gamma=pg[1], ci_hi_gamma=pg[2],
                             pct_best=pb[0], ci_lo_best=pb[1], ci_hi_best=pb[2]))
    return pd.DataFrame(rows)

def scale_control_reselect(R, n_boot=2000, seed=0):
    
    rng = np.random.default_rng(seed); rows = []; B = np.round(np.arange(0, 1.2001, 0.05), 2)
    for l, r in R.items():
        w = r.get("words")
        if w is None or not len(w): continue
        w = w[w.S > 0]
        for (sc, src), g in w.groupby(["scorer", "source"]):
            tid = pd.factorize(g.text_id)[0]; nt = tid.max() + 1
            cnt = np.bincount(tid, minlength=nt).astype(float); S = g.S.to_numpy(float)
            def cvs(n, b):
                x = S / n ** b; m = np.bincount(tid, x, nt) / cnt
                v = np.bincount(tid, x * x, nt) / cnt - m * m
                return np.sqrt(np.maximum(v, 0)) / m
            ok = cnt >= 5
            A = np.stack([cvs(g.n_aks.to_numpy(float), b)[ok] for b in B], 1)
            C = np.stack([cvs(g.n_char.to_numpy(float), b)[ok] for b in B], 1)
            def stat(idx):
                a, c = A[idx], C[idx]
                ia, ic = np.nanargmin(np.nanmean(a, 0)), np.nanargmin(np.nanmean(c, 0))
                v = np.log(c[:, ic] / a[:, ia]); v = v[np.isfinite(v)]
                return (np.exp(v.mean()) - 1) * 100, B[ia], B[ic]
            est, ba, bc = stat(np.arange(len(A)))
            bs = np.array([stat(rng.integers(0, len(A), len(A)))[0] for _ in range(n_boot)])
            rows.append(dict(language=l, scorer=sc, source=src, best_exp_akshara=ba, best_exp_char=bc,
                             pct_best=est, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows)

def _lang_verdict(t, col="pass", low_power=None):
    
    if t is None or not len(t): return "not computed"
    g = t.groupby("language", sort=False)[col].agg(["sum", "count"])
    out = {"holds": [], "mixed": [], "not supported": [], "inconclusive (low power)": []}
    for l, (s, c) in g.iterrows():
        k = "holds" if s == c else "not supported" if s == 0 else "mixed"
        if k == "not supported" and low_power is not None and low_power.get(l): k = "inconclusive (low power)"
        out[k].append(l)
    return "; ".join(f"{k}: {', '.join(v)}" for k, v in out.items() if v)

def _verdict(n_pass, n):
    if n_pass is None: return "descriptive"
    if n == 0: return "not computed"
    f = n_pass / n
    return "robust" if f == 1 else "mostly holds" if f >= 0.75 else "partial" if f >= 0.4 else "not supported"

def reproducibility(R, audit, meco=None):
    
    rows = []
    refs = [r + ("mtp-8",) for r in REFERENCE] + [r + ("v9",) for r in REFERENCE_V9]
    for kind, lang, sc, src, item, ref, ref_run in refs:
        new = np.nan
        if kind == "meco":
            if meco is not None and meco.get(src) is not None:
                m = meco[src]; m = m[m.measure == "gaze_ms"]
                m = m[m.comparison == item] if src == "features" else m[(m.model == sc) & (m.surprisal == item)]
                if len(m): new = float(m.dll_x1000.iloc[0])
        elif kind == "audit":
            if audit is not None:
                m = audit[(audit.language == lang) & (audit.tokenizer == sc)]
                if len(m): new = float(m[item].iloc[0])
        elif lang in R:
            r = R[lang]
            if kind == "stat":
                m = r["stats_main"]; m = m[(m.scorer == sc) & (m.source == src) & (m.metric == item)]
                if len(m): new = float(m["mean"].iloc[0])
            elif kind == "ddS" and r["cross_scorer_ddS"] is not None:
                m = r["cross_scorer_ddS"]; m = m[(m.variant == "main") & (m.model == sc)]
                m = m[m.stratum.str.startswith("adjusted")] if item == "adjusted" else m[m.stratum == "all"]
                if len(m): new = float(m.ddS_bits.iloc[0])
            elif kind == "split":
                m = r["splitting"]; m = m[m.scorer == sc]
                if len(m): new = float(m[item].iloc[0])
        status = "not computed" if not np.isfinite(new) else \
                 ("match" if abs(new - ref) <= TOL[kind] else "DIFFERS")
        rows.append(dict(kind=kind, ref_run=ref_run, language=lang, scorer=sc, source=src, item=item,
                         reference=ref, this_run=new, abs_diff=abs(new - ref) if np.isfinite(new) else np.nan,
                         status=status))
    return pd.DataFrame(rows)

def findings(R, audit):
    try:
        return _findings(R, audit)
    except Exception as e:                       # never lose the report because one table is missing
        import traceback; traceback.print_exc()
        return [(f"findings could not be computed: {type(e).__name__}: {e}", 0, 0, "not computed", "")], {}

def _findings(R, audit):
    """Each finding: (name, tests passed, tests, verdict, evidence). v9: verdicts are per language."""
    F, T = [], {}
    rule = lambda s: s.isin(RULE_SCORERS)
    sp = pd.concat([r["splitting"].assign(language=l) for l, r in R.items()])
    t = sp[rule(sp.scorer)]; lp = sp[~rule(sp.scorer)]
    F.append(("Tokenizers with a GPT-2-style pre-tokenizer split every word that contains a combining mark",
              int((t.frac_split_with_mark >= 0.999).sum()), len(t), "descriptive (true by construction of the rule)",
              f"words with a mark split: {t.frac_split_with_mark.min():.1%}–{t.frac_split_with_mark.max():.1%} "
              f"(learned-subword tokenizers: {lp.frac_split_with_mark.min():.0%}–{lp.frac_split_with_mark.max():.0%})"))
    T["splitting"] = sp[["language", "scorer", "tokens_per_word", "frac_words_split", "frac_split_with_mark"]]
    st = pd.concat([r["stats_main"].assign(language=l) for l, r in R.items()])
    t = st[st.metric == "logratio_char_vs_aks"].assign(**{"pass": lambda d: (d["mean"] > 0) & (d.p_holm < 0.05)})
    F.append(("Per-akshara surprisal is more even than per-character surprisal (exponent 1, NO scale control)",
              int(t["pass"].sum()), len(t), _lang_verdict(t) + " — overstated, see next row",
              f"{pct(t['mean'].min()):+.1f}% to {pct(t['mean'].max()):+.1f}%"))
    T["char_vs_akshara"] = t.assign(pct=pct(t["mean"]))[["language", "scorer", "source", "pct", "p_holm"]]
    SC = scale_control(R)
    if len(SC):
        SC["pass"] = SC.ci_lo_best > 0
        by = SC.groupby("language", sort=False)
        F.append(("Per-akshara surprisal is more even than per-character surprisal AFTER scale control "
                  "(best exponent per unit) — PRIMARY", int(SC["pass"].sum()), len(SC), _lang_verdict(SC),
                  "; ".join(f"{l} {g.pct_best.min():+.1f}% to {g.pct_best.max():+.1f}% "
                            f"(best exponent: akshara {g.best_exp_akshara.min():.2f}–{g.best_exp_akshara.max():.2f})"
                            for l, g in by)))
        T["scale_control"] = SC.drop(columns="pass")
        RS = scale_control_reselect(R)
        if len(RS):
            RS["pass"] = RS.ci_lo > 0
            same = _lang_verdict(RS) == _lang_verdict(SC)
            F[-1] = F[-1][:4] + (F[-1][4] + f". Robustness: with the exponent re-chosen inside every bootstrap sample, "
                                 f"{int(RS['pass'].sum())}/{len(RS)} pass ({_lang_verdict(RS)}; "
                                 f"{'same verdicts' if same else 'VERDICTS CHANGE'}); table_scale_control_reselect.csv",)
            T["scale_control_reselect"] = RS.drop(columns="pass")
    U = unit_comparison(R)
    if len(U):
        w = U[U.unit == "word"].assign(**{"pass": lambda d: (d.pct > 0) & (d.p_holm < 0.05)})
        by = w.groupby("language", sort=False).pct.agg(["min", "max"])
        F.append(("Per-akshara surprisal is more even than whole-word surprisal",
                  int(w["pass"].sum()), len(w), _lang_verdict(w),
                  "; ".join(f"{l} {a:+.1f}% to {b:+.1f}%" for l, (a, b) in by.iterrows())
                  + " (best exponents < 1: per-unit rates over-correct for length)"))
        T["unit_comparison"] = U[["language", "scorer", "source", "unit", "mean_cv_unit", "mean_cv_akshara",
                                  "pct", "ci_lo", "ci_hi", "p_holm"]]
        best = U.assign(_v=U.mean_cv_unit).pivot_table(index=["language", "scorer", "source"], columns="unit",
                                                        values="_v", aggfunc="first")
        best["akshara"] = U.groupby(["language", "scorer", "source"]).mean_cv_akshara.first()
        T["most_even_unit"] = best.assign(most_even=best.idxmin(axis=1)).reset_index()
    # 4. regression: akshara vs character count (log-log, so immune to scale compression) -------
    t = pd.concat([r["length_units_ci"].assign(language=l) for l, r in R.items() if r["length_units_ci"] is not None])
    t["pass"] = t.frac_boot_aks_gt_char >= 0.975
    rev = t[t.frac_boot_aks_gt_char <= 0.025]             # v9.1: character count clearly better
    F.append(("Akshara count predicts word surprisal better than character count (unique ΔR², log-log)",
              int(t["pass"].sum()), len(t), _lang_verdict(t) + ("" if rev.empty else
              "; reversed (character count better) for " + ", ".join(f"{a}/{b}" for a, b in zip(rev.language, rev.scorer))),
              f"unique ΔR² of aksharas {t.dR2_aks_over_char.min():.4f}–{t.dR2_aks_over_char.max():.4f} "
              f"(i.e. ≤ {100 * t.dR2_aks_over_char.max():.1f}% of variance: small); share of bootstrap samples "
              "favouring aksharas: " + ", ".join(f"{l}/{s} {v:.0%}" for l, s, v in
                                                  zip(t.language, t.scorer, t.frac_boot_aks_gt_char))))
    T["length_units"] = t.drop(columns="pass")
    # 5-6. cross-model agreement --------------------------------------------------------------
    ag = pd.concat([r["cross_scorer_agreement"].assign(language=l) for l, r in R.items()])
    m = ag.groupby(["language", "normaliser"]).spearman.mean().unstack()
    tok_low = (m["cv_per_token"] < m.drop(columns="cv_per_token").min(axis=1))
    F.append(("Per-token surprisal is the least consistent measure across models",
              int(tok_low.sum()), len(m), "descriptive (expected: token counts differ by tokenizer)",
              "mean cross-model agreement, per-token vs others: " +
              ", ".join(f"{l} {m.loc[l, 'cv_per_token']:.2f} vs {m.loc[l].drop('cv_per_token').min():.2f}–"
                        f"{m.loc[l].drop('cv_per_token').max():.2f}" for l in m.index)))
    t = pd.DataFrame({"language": m.index, "pass": (m["cv_per_akshara"] > m["cv_per_char"] + 0.02).to_numpy()})
    F.append(("Per-akshara measures agree across models better than per-character measures",
              int(t["pass"].sum()), len(t), _lang_verdict(t), "akshara vs character agreement: " +
              ", ".join(f"{l} {m.loc[l, 'cv_per_akshara']:.2f} vs {m.loc[l, 'cv_per_char']:.2f}" for l in m.index)))
    T["agreement"] = m.reset_index()
    # 7-9. akshara boundaries inside words + calibration ---------------------------------------
    cal = pd.concat([r["grain_calibration"].assign(language=l) for l, r in R.items()])
    calx = cal[cal.metric == "logratio_grain_random_excess"].set_index(["language", "scorer"])
    t = st[st.metric == "logratio_grain_random_excess"].copy()
    t["cal_bias"] = [calx.loc[(l, s), "mean"] + 2 * calx.loc[(l, s), "se"] if (l, s) in calx.index else np.nan
                     for l, s in zip(t.language, t.scorer)]
    t["pass"] = (t["mean"] > 0) & (t.p_holm < 0.05) & (t["mean"] > t.cal_bias)
    t = t.merge(sp[["language", "scorer", "frac_words_split"]], on=["language", "scorer"], how="left")
    rl = rule(t.scorer)
    F.append(("Akshara boundaries matter inside words (rule tokenizers, calibrated test)",
              int(t[rl]["pass"].sum()), int(rl.sum()), _lang_verdict(t[rl]),
              f"{pct(t[rl]['mean'].min()):+.1f}% to {pct(t[rl]['mean'].max()):+.1f}% beyond the shuffle null. Caveat: "
              "these tokenizers cut between consonant and vowel sign, so this may reflect the tokenizer's cuts."))
    ls = t[~rl]
    lowp = ls.groupby("language").frac_words_split.max().lt(0.5).to_dict()
    per_model = "; ".join(f"{l}: " + ", ".join(f"{s} {int(g2['pass'].sum())}/{len(g2)}"
                                                  for s, g2 in g.groupby("scorer", sort=False))
                          for l, g in ls.groupby("language", sort=False))
    F.append(("Akshara boundaries matter inside words (learned-subword tokenizers, calibrated test)",
              int(ls["pass"].sum()), len(ls), _lang_verdict(ls, low_power=lowp),
              f"passes per model (gold + article): {per_model}. "
              f"{pct(ls['mean'].min()):+.1f}% to {pct(ls['mean'].max()):+.1f}%; counted only if larger than the "
              f"model's own calibration bias (+2 s.e.). Only {ls.frac_words_split.min():.0%}–"
              f"{ls.frac_words_split.max():.0%} of words have more than one token, so most words carry no "
              "within-word information for this test."))
    T["boundary"] = t.assign(pct=pct(t["mean"]), cal_bias_pct=pct(t.cal_bias))[
        ["language", "scorer", "source", "pct", "p_holm", "cal_bias_pct", "frac_words_split", "pass"]]
    c = cal[cal.metric.isin(["logratio_grain_random_excess", "logratio_grain_randcut_shufexcess"])].copy()
    c["pass"] = c.p_wilcoxon > 0.05
    F.append(("Calibration: with random surprisals the shuffle-null tests give no effect (p > 0.05)",
              int(c["pass"].sum()), len(c), _lang_verdict(c) + f" (≈{0.05 * len(c):.1f} failures expected by chance)",
              f"smallest p = {c.p_wilcoxon.min():.3f}"))
    # 10. split penalty: exploratory -----------------------------------------------------------
    rows = []
    for l, r in R.items():
        d = r["cross_scorer_ddS"]
        if d is None: continue
        d = d[d.variant == "main"]
        for mdl, g in d.groupby("model"):
            if mdl not in RULE_SCORERS: continue
            est = {}
            for k, sel in [("all", g.stratum == "all"), ("clean", g.stratum == "clean boundaries"),
                           ("adjusted", g.stratum.str.startswith("adjusted"))]:
                if sel.any(): est[k] = g[sel].iloc[0]
            row = dict(language=l, model=mdl)
            for k, e in est.items():
                row.update({f"ddS_{k}": e.ddS_bits, f"lo_{k}": e.ci_lo, f"hi_{k}": e.ci_hi})
            sig = [np.sign(e.ddS_bits) for e in est.values() if e.ci_lo > 0 or e.ci_hi < 0]
            row["signs_agree"] = len(set(sig)) <= 1
            rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        F.append(("Split words receive extra surprisal (split penalty vs BLOOM, rule tokenizers)",
                  int((t.lo_adjusted > 0).sum()) if "lo_adjusted" in t else 0, len(t),
                  "exploratory (not identified: for rule tokenizers 'split' = 'has a mark')",
                  "; ".join(f"{a}/{b} all {c:+.2f}, adjusted {d:+.2f} [{e:+.2f}, {f:+.2f}]{'' if g else ' (sign depends on estimator)'}"
                            for a, b, c, d, e, f, g in zip(t.language, t.model, t.ddS_all, t.ddS_adjusted,
                                                           t.lo_adjusted, t.hi_adjusted, t.signs_agree))))
        T["split_penalty"] = t
    # 11. learned-subword tokenizers: split inside vs between aksharas -------------------------
    rows = []
    for l, r in R.items():
        s0 = r["split_residuals"]
        for sc, s in s0[(s0.fit == "UNSPLIT") & ~rule(s0.scorer)].groupby("scorer"):
            s = s.set_index("category")
            if {"split_aligned", "split_crossing"} <= set(s.index):
                rows.append(dict(language=l, scorer=sc, aligned=s.loc["split_aligned", "mean_resid_bits"],
                                 crossing=s.loc["split_crossing", "mean_resid_bits"],
                                 **{"pass": s.loc["split_crossing", "ci_lo"] > s.loc["split_aligned", "ci_hi"]}))
    t = pd.DataFrame(rows, columns=["language", "scorer", "aligned", "crossing", "pass"])
    F.append(("Learned-subword tokenizers: words split inside an akshara are harder to predict than words split "
              "between aksharas", int(t["pass"].sum()), len(t), _lang_verdict(t),
              ", ".join(f"{a}/{s} {b:.2f} vs {c:.2f} bits" for a, s, b, c in
                        zip(t.language, t.scorer, t.crossing, t.aligned)) or "not computed"))
    T["split_inside_vs_between"] = t
    # 12. tokenizer audit (descriptive) ---------------------------------------------------------
    if audit is not None:
        g = audit.groupby("tokenizer").diagnosis.agg(lambda x: sorted(set(x)))
        fine = [k for k, v in g.items() if v == ["learned subwords"]]
        rl_ = [k for k, v in g.items() if "rule cuts every vowel sign" in v]
        char = [k for k, v in g.items() if k not in rl_ and "script ~unknown: character/byte level" in v]
        cut = [k for k, v in g.items() if k not in fine + rl_ + char]          # v9.1: e.g. gpt-oss
        F.append((f"Tokenizer audit ({audit.tokenizer.nunique()} tokenizers, {audit.language.nunique()} languages): "
                  f"{len(fine)} fine everywhere, {len(rl_)} with the vowel-sign rule", None, audit.tokenizer.nunique(),
                  "descriptive", f"fine everywhere: {', '.join(fine)}; rule cuts every vowel sign: {', '.join(rl_)}; "
                  f"character/byte level in some languages: {', '.join(char)}"
                  + (f"; learned subwords but many vowel signs cut in some languages: {', '.join(cut)}" if cut else "")))
        T["audit_diagnosis"] = audit.pivot(index="tokenizer", columns="language", values="diagnosis").reset_index()
    return F, T

def _family(s):
    return ("BLOOM" if s.startswith("bloom") else "mGPT" if s == "mGPT" else "XGLM" if s.lower().startswith("xglm")
            else "Sarvam-1" if s.lower().startswith("sarvam") else "monolingual GPT-2")

def figure(R, path, SC=None):
    st = pd.concat([r["stats_main"].assign(language=l) for l, r in R.items()])
    sp = pd.concat([r["splitting"].assign(language=l) for l, r in R.items()])
    st["model"] = st.scorer.map(_family); sp["model"] = sp.scorer.map(_family)
    L = [l for l in LANGS if l in R]
    MOD = [m for m in ["monolingual GPT-2", "mGPT", "BLOOM", "XGLM", "Sarvam-1"] if m in set(st.model)]
    col = dict(zip(["monolingual GPT-2", "mGPT", "BLOOM", "XGLM", "Sarvam-1"], ["C0", "C1", "C2", "C3", "C4"]))
    w = 0.8 / len(MOD); off = lambda j: (j - (len(MOD) - 1) / 2) * w
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for j, m in enumerate(MOD):
        v = [sp[(sp.language == l) & (sp.model == m)].frac_words_split.mean() * 100 for l in L]
        axes[0].bar(np.arange(len(L)) + off(j), v, w, color=col[m], label=m)
    axes[0].set_xticks(range(len(L))); axes[0].set_xticklabels(L); axes[0].set_ylabel("% of words split")
    axes[0].set_title("Words split by the tokenizer", fontsize=10); axes[0].legend(fontsize=7)
    # panel 2 (v9): per-akshara vs per-character AFTER the scale control
    if SC is not None and len(SC):
        S2 = SC.assign(model=SC.scorer.map(_family))
        for j, m in enumerate(MOD):
            for k, src in enumerate(["gold", "article"]):
                sub = S2[(S2.model == m) & (S2.source == src)]
                for i, l in enumerate(L):
                    r = sub[sub.language == l]
                    if len(r):
                        r = r.iloc[0]
                        axes[1].errorbar(i + off(j) + (k - 0.5) * w / 3, r.pct_best,
                                         yerr=[[r.pct_best - r.ci_lo_best], [r.ci_hi_best - r.pct_best]],
                                         fmt="o" if src == "gold" else "s", color=col[m], ms=4, capsize=2,
                                         mfc=col[m] if src == "gold" else "white",
                                         label=f"{m} ({src})" if i == 0 else None)
        axes[1].set_title("Per-akshara vs per-character after scale control\n(% more even, > 0 favours akshara)", fontsize=10)
        axes[1].legend(fontsize=6, ncol=2)
    ax = axes[2]
    for j, m in enumerate(MOD):
        for k, src in enumerate(["gold", "article"]):
            sub = st[(st.metric == "logratio_grain_random_excess") & (st.model == m) & (st.source == src)]
            for i, l in enumerate(L):
                r = sub[sub.language == l]
                if len(r):
                    r = r.iloc[0]; y = pct(r["mean"])
                    ax.errorbar(i + off(j) + (k - 0.5) * w / 3, y, yerr=[[y - pct(r.ci_lo)], [pct(r.ci_hi) - y]],
                                fmt="o" if src == "gold" else "s", color=col[m], ms=4, capsize=2,
                                mfc=col[m] if src == "gold" else "white")
    ax.set_title("Akshara boundaries inside words\n(% beyond calibrated null)", fontsize=10)
    for a in axes[1:]:
        a.axhline(0, color="grey", lw=0.8); a.set_xticks(range(len(L))); a.set_xticklabels(L)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(fig)

def build_report(base_dir, langs=LANGS):
    out = os.path.join(base_dir, "FINAL"); os.makedirs(out, exist_ok=True)
    R, audit = load_results(base_dir, langs)
    if not R:
        raise FileNotFoundError("no run_<language>/results folders found; run the experiments first")
    meco = {k: _read(os.path.join(base_dir, "meco", f)) for k, f in
            [("features", "meco_word_features_ll.csv"), ("surprisal", "meco_surprisal_ll.csv")]}
    repro = reproducibility(R, audit, meco); repro.to_csv(os.path.join(out, "reproducibility_check.csv"), index=False)
    F, T = findings(R, audit)
    for k, df in T.items(): df.to_csv(os.path.join(out, f"table_{k}.csv"), index=False)
    figure(R, os.path.join(out, "fig_cross_language.png"), T.get("scale_control"))
    if audit is not None and os.path.exists(os.path.join(base_dir, "audit", "fig_tokenizer_audit.png")):
        import shutil; shutil.copy(os.path.join(base_dir, "audit", "fig_tokenizer_audit.png"), out)
    # ---------------- FINAL_FINDINGS.md ----------------
    vers = {l: {k: r["meta"].get(k) for k in ["timestamp", "python", "torch", "transformers", "datasets"]}
            for l, r in R.items()}
    n_texts = {l: r["meta"].get("config", {}).get("NUM_SAMPLES") for l, r in R.items()}
    rs = repro[repro.status != "not computed"]
    md = [f"# Final findings ({datetime.date.today()})", "",
          f"Languages: {', '.join(R)} | texts per language: {n_texts} | tokenizer audit: "
          f"{'yes' if audit is not None else 'no'}", "",
          "**Reproducibility:** " + "; ".join(
              f"{int((g.status == 'match').sum())} of {len(g)} reference values from the {k} run match"
              for k, g in rs.groupby("ref_run", sort=False))
          + f" ({int((repro.status == 'not computed').sum())} not computed). The mtp-8 values are an independent "
          "check of a clean run; the v9 values (extra scorers, MECO) were recorded from the v9 run itself, so they "
          "are an independent check only for later reproductions. Details: `reproducibility_check.csv`.", "",
          "## Findings", "",
          "Verdicts are given per language (v9): the tests within a language share texts and correlated models, "
          "so a count of passed tests overstates the evidence. 'Passed' is kept for reference.", "",
          "| # | Finding | Passed | Verdict | Evidence |", "|---|---|---|---|---|"]
    for i, (name, k, n, verdict, ev) in enumerate(F, 1):
        md.append(f"| {i} | {name} | {'—' if k is None else f'{k}/{n}'} | **{verdict}** | {ev} |")
    md += ["", "## Tables", ""]
    for k, df in T.items():
        if k == "audit_diagnosis": continue
        md += [f"### {k.replace('_', ' ')}", "", _md_table(df), ""]
    if "audit_diagnosis" in T:
        md += ["### tokenizer audit diagnosis", "", _md_table(T["audit_diagnosis"]), ""]
    md += ["## Run environment", "", "```", json.dumps(vers, indent=1), "```"]
    open(os.path.join(out, "FINAL_FINDINGS.md"), "w", encoding="utf-8").write("\n".join(md))
    return dict(results=R, audit=audit, repro=repro, findings=F, tables=T, out=out)
