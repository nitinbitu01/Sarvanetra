"""backend/scripts/train_reid_market1501.py — train OSNet-IBN on Market-1501.

WHY THIS EXISTS
  Not a replacement for threshold validation (backend/scripts/reid_validate.py
  stays the gate on WATCHLIST auto-dispatch). This trains a checkpoint so the
  torchreid training pipeline is exercised end-to-end and produces a real,
  understood artifact — a necessary step before that pipeline can ever be
  re-pointed at real Gujarat footage.

  Training on Market-1501 tunes the model to Market-1501's own 6 cameras at a
  Tsinghua campus. Published ReID domain-generalization research is
  consistent: this does NOT reliably transfer better to a new city's cameras
  than the generic ImageNet/cross-domain pretrained weights already loaded by
  ReIDEmbedder. Do not treat a good Market-1501 Rank-1/mAP score here as
  evidence the deployed system got better — it is evidence the PIPELINE works.

DATASET
  torchreid's own auto-download mirror (baked into the installed package) is
  dead as of this writing. Get Market-1501-v15.09.15.zip another way (Kaggle
  mirrors work) and extract it to
  <data-root>/market1501/Market-1501-v15.09.15/{bounding_box_train,query,
  bounding_box_test}/ — torchreid skips its own download whenever that
  directory already exists, so a manually-placed copy is picked up with no
  code change needed.

TIMING — measured on this machine (RTX 4070), not estimated
  Per-epoch training (workers=0, OS file cache warm): ~82s
    First epoch is slower (~126s) — cold file cache.
  Full eval pass (Rank-1/mAP over the 3,368-query x 15,913-gallery split):
    ~135s each. Only runs every --eval-freq epochs, not every epoch — with
    the default eval_freq=10 over a 250-epoch run that's ~25 eval passes.
  250 epochs, eval every 10  : roughly 250x82s + 25x135s =~ 6.6 hours
   60 epochs, eval every 10  : roughly  60x82s +  6x135s =~ 1.6 hours
  A synthetic-tensor microbenchmark of just the GPU forward/backward step
  (no real data loading, no eval) earlier suggested 1.7h for the full 250
  epochs — that number undercounted real disk I/O and eval cost and should
  not be trusted; the estimate above is from an actual completed run.

USAGE
  python -m backend.scripts.train_reid_market1501
  python -m backend.scripts.train_reid_market1501 --epochs 60   # quick pass
  python -m backend.scripts.train_reid_market1501 --data-root D:/reid-data

OUTPUT
  <save-dir>/model.pth.tar-<epoch>  — torchreid checkpoint format.
  ReIDEmbedder does not load this automatically (it calls build_model with
  pretrained=True, which loads torchreid's own default weights). Wiring a
  trained checkpoint into ReIDEmbedder is a deliberate follow-up step, not
  automatic — see the module docstring in backend/services/reid_embedder.py.
"""
from __future__ import annotations

import argparse
import sys
import types

# ReIDEmbedder uses this same shim to import torchreid at all — this
# environment's torch build has no usable tensorboard, and torchreid imports
# SummaryWriter unconditionally at module load time.
#
# reid_embedder.py can get away with `SummaryWriter = object` because it never
# actually instantiates a writer. This script does: torchreid's Engine.run()
# calls `SummaryWriter(log_dir=save_dir)` and then `.add_scalar(...)` /
# `.close()` on it every epoch. A bare `object` takes no constructor
# arguments, so training crashed immediately on the first `engine.run()` call
# with "TypeError: object() takes no arguments" — caught by actually running
# it, not by inspection. This no-op class accepts the same calls and discards
# them, since training progress and per-epoch Rank-1/mAP are already printed
# to stdout by the engine regardless of whether a real writer exists.
if "torch.utils.tensorboard" not in sys.modules:
    class _NoOpSummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    _dummy_tb = types.ModuleType("torch.utils.tensorboard")
    _dummy_tb.SummaryWriter = _NoOpSummaryWriter
    sys.modules["torch.utils.tensorboard"] = _dummy_tb

import torch
import torchreid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/reid-data",
                        help="Where Market-1501 is (or will be) downloaded.")
    parser.add_argument("--save-dir", default="output/reid_train/osnet_ibn_market1501",
                        help="Checkpoint + log output directory.")
    parser.add_argument("--epochs", type=int, default=250,
                        help="250 is torchreid's standard OSNet recipe "
                             "(~1.7h measured on this machine's RTX 4070). "
                             "Use 60 for a quick correctness pass (~25min).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-freq", type=int, default=10,
                        help="Run Rank-1/mAP eval every N epochs.")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker subprocesses. torchreid's "
                             "own default is 4, but on this machine that "
                             "reliably crashed every run with a CUDA "
                             "'out of memory' error on the very first conv "
                             "layer, despite 10+ GiB reported free — the "
                             "signature of a Windows multiprocessing/CUDA-"
                             "context interaction, not real memory pressure. "
                             "0 (load in the main process) is slower but is "
                             "the value actually proven to complete a run on "
                             "this box. Try a higher value only if you can "
                             "watch it fail safely; it does not corrupt data, "
                             "it just crashes.")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}"
          + (f"  ({torch.cuda.get_device_name(0)})" if use_gpu else ""))
    if not use_gpu:
        print("WARNING: no GPU detected — this will take hours, not minutes. "
              "See the timing measured in this session's transcript "
              "(16-67h CPU vs 0.4-1.7h GPU) before proceeding.")

    # height=256, width=128 is the standard ReID crop aspect ratio and matches
    # what backend/services/reid_embedder.py resizes crops to before embedding
    # — keeping this consistent means the trained checkpoint sees the same
    # input shape it would see in production, not a mismatched one.
    datamanager = torchreid.data.ImageDataManager(
        root=args.data_root,
        sources="market1501",
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        num_instances=4,
        train_sampler="RandomIdentitySampler",  # required for triplet loss
        workers=args.workers,
    )

    # loss='triplet' makes the model return (logits, features) — required by
    # ImageTripletEngine below. This combined triplet+softmax recipe is
    # torchreid's own documented best-practice combo (not plain softmax
    # alone), and is what the original OSNet paper's reported Rank-1/mAP
    # numbers are trained with.
    model = torchreid.models.build_model(
        name="osnet_ibn_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=True,
    )
    if use_gpu:
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=0.0003)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=max(args.epochs // 3, 1)
    )

    engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer,
        margin=0.3, weight_t=0.7, weight_x=1,
        scheduler=scheduler, use_gpu=use_gpu,
    )

    engine.run(
        max_epoch=args.epochs,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        print_freq=20,
    )

    print(f"\nDone. Checkpoint + Rank-1/mAP log written to {args.save_dir}")
    print("Reminder: this measures fit to Market-1501's own cameras, not "
          "Gujarat CCTV. Re-validate on real labelled pairs via "
          "backend/scripts/reid_validate.py before this checkpoint informs "
          "any threshold or dispatch decision.")


if __name__ == "__main__":
    main()
