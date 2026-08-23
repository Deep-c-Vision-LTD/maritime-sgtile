"""
Train a YOLO detector on the tiled dataset — SageMaker script-mode entry point.

This trains the *detector* that both the static baseline and the adaptive tiler use
at inference time (they differ only in HOW they tile/gate at test time, not in the
weights). So you train ONCE here, then evaluate baseline vs adaptive with run_eval.py.

  * data  : SM_CHANNEL_TRAINING  (tiled dataset root with data.yaml)
  * output: SM_MODEL_DIR         (auto-tarred to s3://.../output/model.tar.gz)

Local smoke:
  python train_tiled.py --train data/sds_tiles --model yolo11s.pt --epochs 1 \
      --model-dir /tmp/det
"""

import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAINING", "data"))
    p.add_argument("--model-dir", dest="model_dir",
                   default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--model_dir", dest="model_dir_s3", default=None)  # SM injects; ignore
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--data-yaml", default="data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--device", default=None,
                   help="ultralytics device, e.g. 0 / cpu / mps (default: auto)")
    return p.parse_args()


def main():
    a = parse_args()
    # Absolute so ultralytics writes weights under <model_dir>/det (a relative
    # `project` is resolved against ultralytics' own runs dir, which then breaks
    # the best.pt lookup below on local runs; SageMaker's SM_MODEL_DIR is absolute).
    a.model_dir = os.path.abspath(a.model_dir)
    os.makedirs(a.model_dir, exist_ok=True)
    from ultralytics import YOLO, RTDETR
    import yaml

    data_yaml_src = os.path.join(a.train, a.data_yaml)
    # Rewrite path: field so Ultralytics resolves train/val dirs correctly whether
    # running locally or on SageMaker (where SM_CHANNEL_TRAINING is an abs path).
    with open(data_yaml_src) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = a.train
    data_yaml = os.path.join(a.model_dir, "data_patched.yaml")
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f)

    print(f"[train] base={a.model} data={data_yaml} epochs={a.epochs} imgsz={a.imgsz}")

    Arch = RTDETR if "rtdetr" in a.model.lower() else YOLO
    print(f"[train] arch={Arch.__name__}")
    model = Arch(a.model)
    results = model.train(
        data=data_yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch_size,
        lr0=a.learning_rate, device=a.device,
        project=a.model_dir, name="det", exist_ok=True,
    )
    best = os.path.join(a.model_dir, "det", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, os.path.join(a.model_dir, "detector_best.pt"))
        print(f"[train] best weights -> {a.model_dir}/detector_best.pt")
    print(f"[train] done. {getattr(results, 'results_dict', '')}")


if __name__ == "__main__":
    main()
