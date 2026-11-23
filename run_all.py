#!/usr/bin/env python3
"""Reproduce every result of the MTP-2 project.

    python run_all.py --check     # 1. check the thesis numbers from the bundled results   (CPU, no internet, ~1 min)
    python run_all.py --sample    # 2. run the whole pipeline on the bundled sample data   (GPU recommended, ~10-20 min)
    python run_all.py             # 3. reproduce everything from scratch                   (2x T4 GPU, ~6-7 h)

Advanced: --steps, --langs, --force (see DETAILS.md).

Steps (run in this order; a step whose results exist is skipped unless --force names it):
    checks       akshara segmenter gold tests + environment manifest            (CPU, seconds)
    audit        tokenizer audit: 21 tokenizers x 8 languages                   (CPU, ~15 min)
    experiments  surprisal experiment, one process per language                 (GPU, ~1-1.7 h per language)
    meco         MECO Hindi eye-tracking analysis (downloads data from OSF)     (GPU, ~20-30 min)
    report       FINAL/: cross-language tables, figures, verdicts, reproducibility check   (CPU, ~1 min)

This script does what the cells of the v9 notebook did; the computation lives in src/ and is unchanged
from the v9 run except for the pinned XGLM/Sarvam-1 revisions and the report fixes listed in DETAILS.md.
"""
import argparse, glob, json, os, platform, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
STEPS = ["checks", "audit", "experiments", "meco", "report"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default=os.path.join(ROOT, "mtp_final"), help="results folder (default: ./mtp_final)")
ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
ap.add_argument("--langs", nargs="+", default=["hindi", "bengali", "marathi"])
ap.add_argument("--n", type=int, default=1000, help="texts per language (reported results: 1000)")
ap.add_argument("--no-extra", action="store_true", help="skip XGLM-564M and Sarvam-1 (original three scorers only)")
ap.add_argument("--force", nargs="*", default=[], help="recompute these even if results exist, e.g. hindi audit meco")
ap.add_argument("--meco-dir", default=None, help="folder with the MECO .rda files (default: <base>/meco_data)")
ap.add_argument("--zip", action="store_true", help="zip the results folder at the end (mtp_final_results.zip)")
ap.add_argument("--check", action="store_true",
                help="check the thesis numbers: rebuild all tables from reference_results/ (CPU, no internet, ~1 min)")
ap.add_argument("--sample", action="store_true",
                help="end-to-end test: checks, experiment and report for Hindi on 50 texts, into ./mtp_sample "
                     "(numbers will NOT match the full run)")
ap.add_argument("--data", default=os.path.join(ROOT, "data"),
                help="folder with bundled data (xlsum/ from scripts/export_xlsum.py, meco/); default ./data")
A = ap.parse_args()
if A.check:
    A.base, A.steps = os.path.join(ROOT, "reference_results"), ["report"]
if A.sample:
    A.n, A.langs = 50, ["hindi"]
    if A.steps == STEPS: A.steps = ["checks", "experiments", "report"]
    if A.base == os.path.join(ROOT, "mtp_final"): A.base = os.path.join(ROOT, "mtp_sample")

BASE = os.path.abspath(A.base); os.makedirs(BASE, exist_ok=True)
DATA = os.path.abspath(A.data)
_meco_default = os.path.join(BASE, "meco_data")
if not A.meco_dir and not glob.glob(os.path.join(_meco_default, "*.rda")) and glob.glob(os.path.join(DATA, "meco", "*.rda")):
    _meco_default = os.path.join(DATA, "meco")              # bundled MECO file
MECO_DIR = os.path.abspath(A.meco_dir or _meco_default)
ENV = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
       "MTP_EXTRA": "0" if A.no_extra else "1"}
# XL-Sum source: the sample run uses data/sample/xlsum (small Hindi sample); a full run uses data/xlsum (complete
# splits from scripts/export_xlsum.py). A full run never reads the sample, so its numbers cannot be affected by it.
if A.sample and os.path.isdir(os.path.join(DATA, "sample", "xlsum")):
    ENV["MTP_DATA_DIR"] = os.path.join(DATA, "sample")
elif not A.sample and os.path.isdir(os.path.join(DATA, "xlsum")):
    ENV["MTP_DATA_DIR"] = DATA
os.environ.update({k: ENV[k] for k in ("MTP_EXTRA", "MTP_DATA_DIR") if k in ENV})


def banner(s):
    print(f"\n{'=' * 25} {s} {'=' * 25}", flush=True)


def run_logged(cmd, log_path, mode="w"):
    """Run a subprocess, stream its output to the console and to a log file; raise on failure."""
    with open(log_path, mode, encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=ENV)
        for line in p.stdout:
            log.write(line); print(line, end="", flush=True)
        return p.wait()


# ----------------------------------------------------------------------------- checks
def step_checks():
    banner("checks")
    import mtp_common
    assert mtp_common.validate_all(verbose=True), "akshara segmenter failed its gold tests"
    man = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), python=platform.python_version(),
               platform=platform.platform(), extra_scorers=not A.no_extra, n=A.n, langs=A.langs)
    try:
        import torch, transformers, datasets
        man.update(torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__,
                   gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError as e:
        man["import_error"] = str(e)
    try:                                         # which commit of the XL-Sum dataset repo is current
        from huggingface_hub import HfApi
        man["xlsum_dataset_sha"] = HfApi().dataset_info("csebuetnlp/xlsum").sha
    except Exception as e:
        man["xlsum_dataset_sha"] = f"unavailable ({type(e).__name__})"
    man["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                       text=True).stdout.splitlines()
    json.dump(man, open(os.path.join(BASE, "environment.json"), "w"), indent=1)
    print({k: v for k, v in man.items() if k != "pip_freeze"})
    print("XL-Sum source:", f"bundled files in {ENV['MTP_DATA_DIR']}/xlsum" if "MTP_DATA_DIR" in ENV else "Hugging Face download",
          "| MECO folder:", MECO_DIR)
    print("reference run: python 3.13.16, torch 2.11.0+cu130, transformers 5.16.1, datasets 2.21.0, 2x Tesla T4")
    if not man.get("gpus"):
        print("WARNING: no GPU found. 'report' works without one; 'experiments' and 'meco' need a GPU.")
    elif len(man["gpus"]) == 1 and not A.no_extra:
        print("NOTE: one GPU only. XGLM-564M and Sarvam-1 will run in float16 (the reference run used float32 on a "
              "second GPU), so their numbers can differ slightly from the reference values.")


# ----------------------------------------------------------------------------- audit
def step_audit():
    banner("tokenizer audit")
    done = os.path.exists(os.path.join(BASE, "audit", "tokenizer_audit.csv"))
    if done and "audit" not in A.force:
        print("audit: results exist, skipped"); return
    import mtp_audit
    t0 = time.time(); mtp_audit.run_audit(BASE)
    print(f"audit finished in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------- experiments
def step_experiments():
    for lang in A.langs:
        banner(f"experiment: {lang}")
        run_dir = os.path.join(BASE, f"run_{lang}")
        done = os.path.exists(os.path.join(run_dir, "results", "rq2_v6_summary.json"))
        if done and lang not in A.force:
            print(f"{lang}: results exist, skipped"); continue
        if lang in A.force and os.path.exists(run_dir):
            shutil.rmtree(run_dir)               # otherwise keep partial checkpoints so an interrupted run resumes
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.join(SRC, "mtp_pipeline.py"), lang, "--n", str(A.n), "--base", BASE]
        t0 = time.time(); rc = run_logged(cmd, os.path.join(run_dir, "log.txt"))
        if rc != 0 or not os.path.exists(os.path.join(run_dir, "results", "rq2_v6_summary.json")):
            raise RuntimeError(f"{lang} failed (exit code {rc}); see {run_dir}/log.txt")
        print(f"{lang} finished in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------- MECO
def _osf_list(url):
    import requests
    items = []
    while url:
        r = requests.get(url, timeout=60); r.raise_for_status(); j = r.json()
        items += j["data"]; url = j["links"].get("next")
    return items


def download_meco(dest):
    """Walk OSF node 3527a: release 2.x / version (latest) / wave 2 / primary data / eye tracking data,
    and download the word-level report(s). Same logic as the v9 notebook."""
    import requests
    os.makedirs(dest, exist_ok=True)
    items = _osf_list("https://api.osf.io/v2/nodes/3527a/files/osfstorage/")
    for step in [["release 2"], ["version"], ["wave 2"], ["primary data"], ["eye tracking", "eye-tracking"]]:
        cands = [it for it in items if it["attributes"]["kind"] == "folder"
                 and any(k in it["attributes"]["name"].lower() for k in step)]
        if not cands:
            raise RuntimeError(f"OSF folder matching {step} not found; available: "
                               f"{[it['attributes']['name'] for it in items]}")
        pick = sorted(cands, key=lambda it: it["attributes"]["name"])[-1]
        print("  ->", pick["attributes"]["name"])
        items = _osf_list(pick["relationships"]["files"]["links"]["related"]["href"])
    files = []
    for it in items:
        if it["attributes"]["kind"] == "file": files.append(it)
        else: files += [x for x in _osf_list(it["relationships"]["files"]["links"]["related"]["href"])
                        if x["attributes"]["kind"] == "file"]
    want = [f for f in files if f["attributes"]["name"].lower().endswith((".rda", ".rdata", ".csv"))
            and "word" in f["attributes"]["name"].lower()]
    if not want:
        want = [f for f in files if f["attributes"]["name"].lower().endswith((".rda", ".rdata", ".csv"))
                and not any(k in f["attributes"]["name"].lower() for k in ["fix", "sent", "passage", "comp"])]
    for f in want:
        name = f["attributes"]["name"]; out = os.path.join(dest, name)
        if os.path.exists(out): print("  have", name); continue
        print("  downloading", name, f"({(f['attributes'].get('size') or 0) / 1e6:.0f} MB)")
        with requests.get(f["links"]["download"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 20): fh.write(chunk)


def step_meco():
    banner("MECO Hindi reading times")
    done = os.path.exists(os.path.join(BASE, "meco", "meco_summary.json"))
    if done and "meco" not in A.force:
        print("MECO: results exist, skipped"); return
    have = glob.glob(os.path.join(MECO_DIR, "**", "*.rda"), recursive=True) + \
           glob.glob(os.path.join(MECO_DIR, "**", "*.csv"), recursive=True)
    if not have:
        print("Downloading MECO Wave 2 data from OSF …")
        try:
            download_meco(MECO_DIR)
        except Exception as e:
            sys.exit(f"Automatic download failed ({e}). Download manually (see README, 'Data') into {MECO_DIR}")
    # the font used for the visual-width control is bundled, so widths do not depend on a moving URL
    os.makedirs(os.path.join(BASE, "meco"), exist_ok=True)
    font = os.path.join(ROOT, "assets", "NotoSansDevanagari-Regular.ttf")
    if os.path.exists(font): shutil.copy(font, os.path.join(BASE, "meco"))
    script = os.path.join(SRC, "mtp_meco.py"); log = os.path.join(BASE, "meco_log.txt")
    for extra, mode in [(["--inspect"], "w"), ([], "a")]:          # inspect first (prints what was found), then run
        rc = run_logged([sys.executable, "-u", script, "--meco", MECO_DIR, "--base", BASE] + extra, log, mode)
        if rc != 0: raise RuntimeError(f"MECO step failed; see {log}")


# ----------------------------------------------------------------------------- report
def step_report():
    banner("final report")
    import mtp_report
    rep = mtp_report.build_report(BASE, A.langs)
    if A.sample: print("NOTE: sample run (50 texts) - DIFFERS against the reference values is expected.")
    r = rep["repro"]; done = r[r.status != "not computed"]
    for k, g in done.groupby("ref_run", sort=False):
        print(f"Reproducibility ({k} reference): {int((g.status == 'match').sum())}/{len(g)} match")
    print(f"not computed: {int((r.status == 'not computed').sum())}")
    if (done.status == "DIFFERS").any():
        print(done[done.status == "DIFFERS"].to_string(index=False))
    print("findings:", os.path.join(rep["out"], "FINAL_FINDINGS.md"))


if __name__ == "__main__":
    t0 = time.time()
    for s in STEPS:
        if s in A.steps:
            globals()[f"step_{s}"]()
    if A.zip:
        shutil.make_archive(os.path.join(ROOT, "mtp_final_results"), "zip", BASE)
        print("zipped to", os.path.join(ROOT, "mtp_final_results.zip"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
