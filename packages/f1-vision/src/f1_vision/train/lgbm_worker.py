"""LightGBM baseline worker — run as its own process, never imports torch.

LightGBM (Homebrew libomp) and PyTorch (bundled libomp) deadlock when both
OpenMP runtimes are loaded into one process on macOS (confirmed by stack
sample: two libomp.dylib copies parked in __kmp_fork_barrier). The study
exchanges arrays with this worker via .npz files — boring and debuggable.

Usage: ``python -m f1_vision.train.lgbm_worker <in.npz> <out.npz>``
where in.npz has x_train, y_pace, y_cliff, x_test, seed.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    import lightgbm as lgb

    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    seed = int(data["seed"])
    reg = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1
    ).fit(data["x_train"], data["y_pace"])
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1
    ).fit(data["x_train"], data["y_cliff"])
    np.savez(
        out_path,
        pace=reg.predict(data["x_test"]),
        cliff=clf.predict_proba(data["x_test"])[:, 1],
    )


if __name__ == "__main__":
    main()
