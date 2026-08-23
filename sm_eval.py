"""
SageMaker GPU eval entry point for the GRSL cross-architecture experiment.

Runs the project's run_eval (static baseline + S1 adaptive) for one detector
backend on the SeaDronesSee val split, on GPU, and writes the results JSON to
SM_MODEL_DIR. Local MPS eval hangs on 4K frames for the two-stage detector, so
the evals run here instead.

Channels:
  images    -> SM_CHANNEL_IMAGES     (val frames)
  saliency  -> SM_CHANNEL_SALIENCY   (xSoyel-Net SAR maps for S1)
  ann       -> SM_CHANNEL_ANN        (instances_val.json)
  weights   -> SM_CHANNEL_WEIGHTS    (detector_best.pt)

Hyperparameters: backend (yolo|rtdetr|frcnn), keep_thr, conf.
"""
import os, sys, subprocess, glob, shutil, json

def sh(cmd):
    print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)

def main():
    hp = json.load(open("/opt/ml/input/config/hyperparameters.json"))
    backend = hp.get("backend", "yolo").strip('"')
    keep_thr = hp.get("keep_thr", "0.05").strip('"')
    conf = hp.get("conf", "0.25").strip('"')
    use_alt = hp.get("alt", "1").strip('"') != "0"

    if backend in ("yolo", "rtdetr"):
        sh("pip install --no-cache-dir 'ultralytics==8.4.66' 'numpy<2' opencv-python-headless")
    else:
        sh("pip install --no-cache-dir 'numpy<2' opencv-python-headless")

    images = os.environ["SM_CHANNEL_IMAGES"]
    sal    = os.environ["SM_CHANNEL_SALIENCY"]
    ann    = glob.glob(os.path.join(os.environ["SM_CHANNEL_ANN"], "*.json"))[0]
    wdir   = os.environ["SM_CHANNEL_WEIGHTS"]
    hits = glob.glob(os.path.join(wdir, "**", "detector_best.pt"), recursive=True)
    if not hits:  # channel delivered model.tar.gz -> extract it
        for tb in glob.glob(os.path.join(wdir, "**", "*.tar.gz"), recursive=True):
            sh(f"tar xzf {tb} -C {wdir}")
        hits = glob.glob(os.path.join(wdir, "**", "detector_best.pt"), recursive=True)
    weights = hits[0]
    out = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), f"{backend}_S1.json")

    # code (tiling/, eval/, data/convert_seadronessee.py) ships in source_dir; make importable
    sys.path.insert(0, os.getcwd()); sys.path.insert(0, "data")
    cmd = (f"python eval/run_eval.py --weights {weights} --backend {backend} "
           f"--val-coco-json {ann} --images-dir {images} --saliency-dir {sal} "
           f"--scheme binary {'--use-altitude' if use_alt else '--no-altitude'} --use-saliency-select --gate-alpha 0 "
           f"--keep-thr {keep_thr} --conf {conf} --out {out}")
    env = dict(os.environ); env["PYTHONPATH"] = os.getcwd() + ":data:" + env.get("PYTHONPATH", "")
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)
    print("RESULT written to", out, flush=True)
    print(open(out).read()[:2000], flush=True)

if __name__ == "__main__":
    main()
