"""mtp_common: scripts, akshara segmenter, hand-checked gold segmentations, language profiles.
Shared by the pipeline (mtp_pipeline.py), the tokenizer audit and the final report."""
import os, re, unicodedata

def xlsum_split(lang, split):
    """v9.1: XL-Sum split as a table with columns id/url/title/summary/text. Reads
    $MTP_DATA_DIR/xlsum/<lang>/<split>.jsonl.gz when it exists (written by scripts/export_xlsum.py,
    same rows in the same order as the Hugging Face dataset), otherwise downloads csebuetnlp/xlsum."""
    d = os.environ.get("MTP_DATA_DIR")
    f = os.path.join(d, "xlsum", lang, f"{split}.jsonl.gz") if d else None
    if f and os.path.exists(f):
        import pandas as _pd
        t = _pd.read_json(f, lines=True, compression="gzip", dtype=False)
        print(f"XL-Sum {lang}/{split}: local file {f} ({len(t):,} rows)")
        return t
    from datasets import load_dataset
    return load_dataset("csebuetnlp/xlsum", lang, split=split, trust_remote_code=True)

# ===================== SCRIPTS & AKSHARA SEGMENTER =====================
SCRIPTS = {
    #             block start  conjuncts  extra consonants / independent letters (offsets in block)
    "devanagari": dict(base=0x0900, joins=True,  cons=range(0x78, 0x80), indep=range(0x72, 0x78)),
    "bengali":    dict(base=0x0980, joins=True,  cons=(0x70, 0x71),      indep=(0x4E,)),   # ৎ khanda ta = own unit
    "gurmukhi":   dict(base=0x0A00, joins=True,  cons=(),                indep=(0x72, 0x73)),
    "gujarati":   dict(base=0x0A80, joins=True,  cons=(0x79,),           indep=()),
    "tamil":      dict(base=0x0B80, joins=False, cons=(),                indep=(0x03,)),   # ஃ aytham
    "telugu":     dict(base=0x0C00, joins=True,  cons=(),                indep=()),
}
LANG_SCRIPT = {"hindi": "devanagari", "marathi": "devanagari", "nepali": "devanagari",
               "bengali": "bengali", "punjabi": "gurmukhi", "gujarati": "gujarati",
               "tamil": "tamil", "telugu": "telugu"}

class ScriptSpec:
    def __init__(self, name):
        sp = SCRIPTS[name]; self.name, self.base, self.joins = name, sp["base"], sp["joins"]
        self.cons, self.vowel, self.dep, self.mod = set(), set(), set(), set()
        self.virama = self.nukta = None
        word_chars, marks = [], []
        for cp in range(self.base, self.base + 0x80):
            ch = chr(cp); cat = unicodedata.category(ch); nm = unicodedata.name(ch, ""); off = cp - self.base
            if cat in ("Mn", "Mc"):
                marks.append(ch); word_chars.append(ch)
                if nm.endswith("SIGN VIRAMA"):                     self.virama = ch
                elif "SIGN NUKTA" in nm:                           self.nukta = ch
                elif "VOWEL SIGN" in nm or "LENGTH MARK" in nm:   self.dep.add(ch)
                else:                                              self.mod.add(ch)  # anusvara, candrabindu, visarga, addak, tippi …
            elif cat in ("Lo", "Lm"):
                word_chars.append(ch)
                if cat == "Lo" and (0x15 <= off <= 0x39 or 0x58 <= off <= 0x5F or off in sp["cons"]):
                    self.cons.add(ch)
                elif cat == "Lo" and (0x04 <= off <= 0x14 or off in (0x60, 0x61) or off in sp["indep"]):
                    self.vowel.add(ch)
        cls = "".join(re.escape(c) for c in word_chars)
        self.word_re = re.compile(f"[{cls}]+")                    # a word = a run of script letters/marks
        self.char_re = re.compile(f"[{cls}]")
        self.mark_re = re.compile("[" + "".join(re.escape(c) for c in marks) + "]")
        # 2nd UTF-8 byte of this block's code points (for byte-fragment tokens)
        self.byte2 = {0x80 | ((cp >> 6) & 0x3F) for cp in range(self.base, self.base + 0x80)}

    def spans(self, text):
        out, i, n = [], 0, len(text)
        C, V, DEP, MOD, VIR, NUK = self.cons, self.vowel, self.dep, self.mod, self.virama, self.nukta
        while i < n:
            c = text[i]
            if c in C:
                j = i + 1
                while True:
                    while j < n and text[j] == NUK: j += 1
                    if self.joins and j + 1 < n and text[j] == VIR and text[j + 1] in C:
                        j += 2; continue                          # conjunct: keep going
                    break
                if j < n and text[j] == VIR: j += 1              # explicit virama at the end
                while j < n and (text[j] in DEP or text[j] in MOD or text[j] == NUK): j += 1
                out.append((i, j)); i = j
            elif c in V:
                j = i + 1
                while j < n and text[j] in MOD: j += 1
                out.append((i, j)); i = j
            else:
                i += 1
        return out

GOLD_HINDI = {
    # original 15
    "देश": ["दे", "श"], "राजधानी": ["रा", "ज", "धा", "नी"], "दिल्ली": ["दि", "ल्ली"],
    "नमस्ते": ["न", "म", "स्ते"], "विश्वविद्यालय": ["वि", "श्व", "वि", "द्या", "ल", "य"],
    "भारत": ["भा", "र", "त"], "पैंथर": ["पैं", "थ", "र"],
    "अफ़ग़ानिस्तान": ["अ", "फ़", "ग़ा", "नि", "स्ता", "न"], "इंसाफ़": ["इं", "सा", "फ़"],
    "उदघाटन": ["उ", "द", "घा", "ट", "न"], "स्त्री": ["स्त्री"], "हँसना": ["हँ", "स", "ना"],
    "दुःख": ["दुः", "ख"], "क्या": ["क्या"], "ऑफ़िस": ["ऑ", "फ़ि", "स"],
    # plain consonant + matra
    "कमल": ["क", "म", "ल"], "घर": ["घ", "र"], "पानी": ["पा", "नी"], "किताब": ["कि", "ता", "ब"],
    "सरकार": ["स", "र", "का", "र"], "कुछ": ["कु", "छ"], "शहर": ["श", "ह", "र"],
    "चुनाव": ["चु", "ना", "व"], "पुलिस": ["पु", "लि", "स"], "रुपये": ["रु", "प", "ये"],
    "अमरीका": ["अ", "म", "री", "का"], "पाकिस्तान": ["पा", "कि", "स्ता", "न"],
    # independent vowels
    "आदमी": ["आ", "द", "मी"], "ईमानदार": ["ई", "मा", "न", "दा", "र"], "ऊपर": ["ऊ", "प", "र"],
    "ऐसा": ["ऐ", "सा"], "औरत": ["औ", "र", "त"], "ओर": ["ओ", "र"], "एक": ["ए", "क"],
    "ऋषि": ["ऋ", "षि"], "ऋतु": ["ऋ", "तु"], "आई": ["आ", "ई"], "गए": ["ग", "ए"],
    "हुआ": ["हु", "आ"], "कई": ["क", "ई"], "भाई": ["भा", "ई"], "नई": ["न", "ई"],
    "जाएगा": ["जा", "ए", "गा"], "रुपए": ["रु", "प", "ए"],
    # vocalic r matra
    "कृपया": ["कृ", "प", "या"], "हृदय": ["हृ", "द", "य"], "कृषि": ["कृ", "षि"],
    "तृतीय": ["तृ", "ती", "य"],
    # conjuncts
    "बच्चा": ["ब", "च्चा"], "अच्छा": ["अ", "च्छा"], "उत्तर": ["उ", "त्त", "र"],
    "प्रधानमंत्री": ["प्र", "धा", "न", "मं", "त्री"], "राष्ट्रपति": ["रा", "ष्ट्र", "प", "ति"],
    "महाराष्ट्र": ["म", "हा", "रा", "ष्ट्र"], "ज्ञान": ["ज्ञा", "न"], "क्षेत्र": ["क्षे", "त्र"],
    "श्रीमती": ["श्री", "म", "ती"], "त्योहार": ["त्यो", "हा", "र"], "मंत्रालय": ["मं", "त्रा", "ल", "य"],
    "विद्यार्थी": ["वि", "द्या", "र्थी"], "स्वास्थ्य": ["स्वा", "स्थ्य"], "स्थिति": ["स्थि", "ति"],
    "द्वारा": ["द्वा", "रा"], "पत्नी": ["प", "त्नी"], "सूर्य": ["सू", "र्य"], "धर्म": ["ध", "र्म"],
    "कर्मचारी": ["क", "र्म", "चा", "री"], "पर्यावरण": ["प", "र्या", "व", "र", "ण"],
    "प्रदेश": ["प्र", "दे", "श"], "ट्रेन": ["ट्रे", "न"], "स्कूल": ["स्कू", "ल"],
    "प्रतिशत": ["प्र", "ति", "श", "त"], "अर्थव्यवस्था": ["अ", "र्थ", "व्य", "व", "स्था"],
    "स्वतंत्रता": ["स्व", "तं", "त्र", "ता"], "जम्मू": ["ज", "म्मू"], "कश्मीर": ["क", "श्मी", "र"],
    "न्यायालय": ["न्या", "या", "ल", "य"], "ध्यान": ["ध्या", "न"], "द्विवेदी": ["द्वि", "वे", "दी"],
    "उन्होंने": ["उ", "न्हों", "ने"], "क्योंकि": ["क्यों", "कि"], "अर्ध": ["अ", "र्ध"],
    "डॉक्टर": ["डॉ", "क्ट", "र"], "हिन्दी": ["हि", "न्दी"],
    # anusvara / chandrabindu / visarga
    "संसद": ["सं", "स", "द"], "हिंदी": ["हिं", "दी"], "सिंह": ["सिं", "ह"], "बैंक": ["बैं", "क"],
    "मैं": ["मैं"], "हैं": ["हैं"], "गाँव": ["गाँ", "व"], "चाँद": ["चाँ", "द"],
    "आँख": ["आँ", "ख"], "हँसी": ["हँ", "सी"], "कहाँ": ["क", "हाँ"], "अतः": ["अ", "तः"],
    "प्रातः": ["प्रा", "तः"], "इंडिया": ["इं", "डि", "या"], "अंग्रेज़ी": ["अं", "ग्रे", "ज़ी"],
    # nukta
    "ज़िंदगी": ["ज़िं", "द", "गी"], "फ़िल्म": ["फ़ि", "ल्म"], "क़ानून": ["क़ा", "नू", "न"],
    "ग़रीब": ["ग़", "री", "ब"], "ख़बर": ["ख़", "ब", "र"], "ज़्यादा": ["ज़्या", "दा"],
    "बड़ा": ["ब", "ड़ा"], "पढ़ाई": ["प", "ढ़ा", "ई"], "लड़की": ["ल", "ड़", "की"],
    "सड़क": ["स", "ड़", "क"],
    # loanword vowel ऑ / ॉ
    "ऑफ़": ["ऑ", "फ़"], "कॉलेज": ["कॉ", "ले", "ज"],
    # word-final halant
    "वाक्": ["वा", "क्"], "जगत्": ["ज", "ग", "त्"],
}

GOLD_MARATHI = {   # Devanagari, Marathi-specific letters and spellings
    "आणि": ["आ", "णि"], "मुंबई": ["मुं", "ब", "ई"], "पुणे": ["पु", "णे"], "मिळाले": ["मि", "ळा", "ले"],
    "शाळा": ["शा", "ळा"], "ॲप": ["ॲ", "प"], "झाले": ["झा", "ले"], "त्यांनी": ["त्यां", "नी"],
    "करण्यात": ["क", "र", "ण्या", "त"], "निवडणूक": ["नि", "व", "ड", "णू", "क"], "मराठी": ["म", "रा", "ठी"],
    "दुसऱ्या": ["दु", "स", "ऱ्या"],
}
GOLD_BENGALI = {
    "বাংলা": ["বাং", "লা"], "ভারত": ["ভা", "র", "ত"], "দেশ": ["দে", "শ"], "সরকার": ["স", "র", "কা", "র"],
    "প্রধানমন্ত্রী": ["প্র", "ধা", "ন", "ম", "ন্ত্রী"], "বিশ্ববিদ্যালয়": ["বি", "শ্ব", "বি", "দ্যা", "ল", "য়"],
    "ক্ষমতা": ["ক্ষ", "ম", "তা"], "রাষ্ট্রপতি": ["রা", "ষ্ট্র", "প", "তি"], "উৎসব": ["উ", "ৎ", "স", "ব"],
    "ভবিষ্যৎ": ["ভ", "বি", "ষ্য", "ৎ"], "আমি": ["আ", "মি"], "তিনি": ["তি", "নি"], "কর্মী": ["ক", "র্মী"],
    "চাঁদ": ["চাঁ", "দ"], "দুঃখ": ["দুঃ", "খ"], "হয়েছে": ["হ", "য়ে", "ছে"], "বড়": ["ব", "ড়"],
    "পড়াশোনা": ["প", "ড়া", "শো", "না"], "ঐতিহ্য": ["ঐ", "তি", "হ্য"], "ঔষধ": ["ঔ", "ষ", "ধ"],
    "ঋণ": ["ঋ", "ণ"], "কৃষি": ["কৃ", "ষি"], "স্বাস্থ্য": ["স্বা", "স্থ্য"], "নির্বাচন": ["নি", "র্বা", "চ", "ন"],
    "আন্দোলন": ["আ", "ন্দো", "ল", "ন"], "পুলিশ": ["পু", "লি", "শ"], "ঢাকা": ["ঢা", "কা"],
    "সংসদ": ["সং", "স", "দ"], "যুক্তরাষ্ট্র": ["যু", "ক্ত", "রা", "ষ্ট্র"], "ইংরেজি": ["ইং", "রে", "জি"],
    "গ্রাম": ["গ্রা", "ম"], "সৌদি": ["সৌ", "দি"], "কোভিড": ["কো", "ভি", "ড"], "জ্ঞান": ["জ্ঞা", "ন"],
    "শিক্ষার্থী": ["শি", "ক্ষা", "র্থী"], "অর্থনীতি": ["অ", "র্থ", "নী", "তি"], "মৃত্যু": ["মৃ", "ত্যু"],
    "ব্যবসা": ["ব্য", "ব", "সা"],
}
GOLD_GURMUKHI = {
    "ਪੰਜਾਬ": ["ਪੰ", "ਜਾ", "ਬ"], "ਸਰਕਾਰ": ["ਸ", "ਰ", "ਕਾ", "ਰ"], "ਪ੍ਰਧਾਨ": ["ਪ੍ਰ", "ਧਾ", "ਨ"],
    "ਪੱਕਾ": ["ਪੱ", "ਕਾ"], "ਸਿੰਘ": ["ਸਿੰ", "ਘ"], "ਖ਼ਬਰ": ["ਖ਼", "ਬ", "ਰ"], "ਇੰਡੀਆ": ["ਇੰ", "ਡੀ", "ਆ"],
}
GOLD_GUJARATI = {
    "ગુજરાત": ["ગુ", "જ", "રા", "ત"], "પ્રધાનમંત્રી": ["પ્ર", "ધા", "ન", "મં", "ત્રી"],
    "સરકાર": ["સ", "ર", "કા", "ર"], "અમદાવાદ": ["અ", "મ", "દા", "વા", "દ"], "શિક્ષણ": ["શિ", "ક્ષ", "ણ"],
    "ચૂંટણી": ["ચૂં", "ટ", "ણી"],
}
GOLD_TAMIL = {
    "தமிழ்நாடு": ["த", "மி", "ழ்", "நா", "டு"], "இந்தியா": ["இ", "ந்", "தி", "யா"], "அரசு": ["அ", "ர", "சு"],
    "சென்னை": ["செ", "ன்", "னை"], "கொரோனா": ["கொ", "ரோ", "னா"], "ஃபேஸ்புக்": ["ஃ", "பே", "ஸ்", "பு", "க்"],
    "பள்ளி": ["ப", "ள்", "ளி"], "மக்கள்": ["ம", "க்", "க", "ள்"],
}
GOLD_TELUGU = {
    "తెలుగు": ["తె", "లు", "గు"], "ప్రభుత్వం": ["ప్ర", "భు", "త్వం"], "హైదరాబాద్": ["హై", "ద", "రా", "బా", "ద్"],
    "ఆంధ్రప్రదేశ్": ["ఆం", "ధ్ర", "ప్ర", "దే", "శ్"], "ఎన్నికలు": ["ఎ", "న్ని", "క", "లు"], "రాష్ట్రం": ["రా", "ష్ట్రం"],
}
GOLD_BY_SCRIPT = {"devanagari": {**GOLD_HINDI, **GOLD_MARATHI}, "bengali": GOLD_BENGALI,
                  "gurmukhi": GOLD_GURMUKHI, "gujarati": GOLD_GUJARATI, "tamil": GOLD_TAMIL, "telugu": GOLD_TELUGU}
SPECS = {name: ScriptSpec(name) for name in SCRIPTS}

def validate_script(name, verbose=True):
    sp, N = SPECS[name], (lambda s: unicodedata.normalize("NFC", s))
    gold = GOLD_BY_SCRIPT[name]
    bad = {}
    for w, g in gold.items():
        got = [N(w)[a:b] for a, b in sp.spans(N(w))]
        if got != [N(x) for x in g]: bad[w] = (g, got)
    if verbose:
        print(f"akshara segmenter [{name}]: {len(gold) - len(bad)}/{len(gold)} exact")
        for w, (g, got) in bad.items(): print(f"  MISMATCH {w}: gold={g} got={got}")
    return not bad


def validate_all(verbose=True):
    return all(validate_script(s, verbose) for s in SCRIPTS)

# ---------------------------------------------------------------------------
RULE_SCORERS = ("gpt2-hindi", "gpt2-bengali", "mGPT")
# ---------------------------------------------------------------------------
HINDI_FUNC = set("""के में है की को से ने और का कि पर हैं भी यह इस एक था थी थे लिए तो
नहीं हो कर गया गई गए साथ बाद वह उन उनके उसके इसके उनकी उसकी इसकी उनका उसका इसका रहा रही रहे
जो किया कहा ही वे अपने अपनी अपना तक हुए हुई हुआ होने करने करते करता करती या इन उस जा जाता जाती
जाते जाए जाएगा सकता सकती सकते रूप ये वो कुछ सभी अब जब तब यहाँ वहाँ लेकिन मगर किसी कोई होता
होती होते बहुत ना न दिया दी दे कई""".split())

LANG_PROFILES = {
    "hindi":   dict(models=("surajp/gpt2-hindi", "ai-forever/mGPT", "bigscience/bloom-560m"),
                    revisions={"surajp/gpt2-hindi": "6cd8d84edae1b247b8a2f446421de8f5b9208f18",
                               "ai-forever/mGPT": "40897bd7c8b47a76802c411108ca6220438b8b40",
                               "bigscience/bloom-560m": "ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971",
                               "facebook/xglm-564M": "f3059f01b98ccc877c673149e0178c0e957660f9",
                               "sarvamai/sarvam-1": "e9607337286ddf496d4a2562b194e489dcf3feea"},
                    probe_start=" भारत में", probe_word=" किताब", danda=True,
                    probe_words=("विश्वविद्यालय", "प्रधानमंत्री", "अफ़ग़ानिस्तान"), func=HINDI_FUNC,
                    extra=("facebook/xglm-564M", "sarvamai/sarvam-1")),
    # Bengali: monolingual GPT-2 (mC4-bn) + the same two multilingual models as Hindi
    "bengali": dict(models=("flax-community/gpt2-bengali", "ai-forever/mGPT", "bigscience/bloom-560m"),
                    revisions={"flax-community/gpt2-bengali": "18336d76c4ab3c2972532fd9d4730708bbdd81eb",
                               "ai-forever/mGPT": "40897bd7c8b47a76802c411108ca6220438b8b40",
                               "bigscience/bloom-560m": "ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971",
                               "facebook/xglm-564M": "f3059f01b98ccc877c673149e0178c0e957660f9",
                               "sarvamai/sarvam-1": "e9607337286ddf496d4a2562b194e489dcf3feea"},
                    probe_start=" ভারত এবং", probe_word=" বিশ্ববিদ্যালয়", danda=True,
                    probe_words=("বিশ্ববিদ্যালয়", "প্রধানমন্ত্রী", "বাংলাদেশ"), func=None,
                    extra=("facebook/xglm-564M", "sarvamai/sarvam-1")),
    # Marathi: no monolingual GPT-2 used; mGPT and BLOOM both cover Marathi
    "marathi": dict(models=("ai-forever/mGPT", "bigscience/bloom-560m"),
                    revisions={"ai-forever/mGPT": "40897bd7c8b47a76802c411108ca6220438b8b40",
                               "bigscience/bloom-560m": "ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971",
                               "sarvamai/sarvam-1": "e9607337286ddf496d4a2562b194e489dcf3feea"},
                    probe_start=" भारत आणि", probe_word=" महाराष्ट्र", danda=False,
                    probe_words=("महाराष्ट्र", "मुंबईतील", "निवडणुकीत"), func=None,
                    extra=("sarvamai/sarvam-1",)),        # XGLM was not trained on Marathi
    # Tamil / Telugu work too (mGPT + BLOOM), but read the Tamil note in the script cell first
    "tamil":   dict(models=("ai-forever/mGPT", "bigscience/bloom-560m"), revisions={},
                    probe_start=" இந்தியா மற்றும்", probe_word=" தமிழ்நாடு", danda=False,
                    probe_words=("தமிழ்நாடு", "இந்தியா", "சென்னை"), func=None),
    "telugu":  dict(models=("ai-forever/mGPT", "bigscience/bloom-560m"), revisions={},
                    probe_start=" భారత్ మరియు", probe_word=" ప్రభుత్వం", danda=False,
                    probe_words=("ప్రభుత్వం", "తెలుగు", "హైదరాబాద్"), func=None),
}
