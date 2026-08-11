"""Adapter for the OFFICIAL Time-Series-Library implementations of PatchTST,
TimeMixer, Crossformer and FEDformer.

The architectures are the upstream code, vendored unmodified at
`third_party/Time-Series-Library` (thuml/Time-Series-Library, pinned commit
`4e938a1`). Nothing in that tree is edited; this file only builds the config
object each `Model(configs)` expects and reshapes our task to its I/O contract.

TASK ADAPTATION -- the one thing that is NOT upstream, stated plainly.
Those models are benchmarked on multivariate long-horizon forecasting: a long
`seq_len` in, `pred_len` future steps of every channel out. This project's task
is a single scalar per anchor -- the h=7 change of the causal Kalman trend --
from an L=30 causal window. So they are configured with `seq_len=30`,
`pred_len=1`, and the model's first output slot is trained by MSE against that
scalar. Consequences worth knowing:

  * The comparison against LR/RF/XGB/LGBM/ARIMA/GRU is exact: identical inputs,
    identical target, identical loss and early-stopping protocol. That is the
    point -- a model comparison in which the models see different targets is not
    a comparison.
  * These are therefore NOT benchmark reproductions, and the numbers must not
    be read against published results on ETT/Weather/Traffic.
  * L=30 is far shorter than the input lengths these models are designed for,
    which degrades some of their mechanisms. Two are worth naming:
    Crossformer hardcodes `seg_len=12`, so a 30-bar window pads to 36 and gives
    only 3 segments; and FEDformer's mode selection is vacuous here, because
    `modes=32` is clamped to `seq_len // 2 = 15`, so it keeps ALL frequencies
    rather than a sparse subset.
  * With upstream defaults these models are large relative to ~1.5-4k training
    rows per fold (Crossformer ~507k parameters). `SCALE` below shrinks
    `d_model`/`d_ff` from the library defaults (512/2048) to 64/128 to keep them
    in the same capacity range as the GRU they are compared with; set
    `SCALE = "library"` to use upstream defaults instead and watch them overfit.

IMPORT COLLISION. Crossformer does `from models.PatchTST import FlattenHead`,
so the library's `models` package must be importable under exactly that name --
which collides with this project's own `models.py`. `_load_official` therefore
swaps `sys.modules["models"]` for the duration of the import and restores it
immediately, keeping the shadow contained to that call. Loading is lazy (first
model construction, then cached) so it can never run while our `models.py` is
still mid-initialisation.
"""
import contextlib
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

TSLIB_DIR = Path(__file__).resolve().parent / "third_party" / "Time-Series-Library"
TSLIB_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"
TSLIB_MODELS = ("PatchTST", "TimeMixer", "Crossformer", "FEDformer",
                "DLinear", "TimeFilter")

# "project" keeps these comparable in capacity to the GRU; "library" uses the
# upstream argparse defaults (d_model 512, d_ff 2048).
SCALE = "project"
_SCALES = {"project": dict(d_model=64, n_heads=4, d_ff=128),
           "library": dict(d_model=512, n_heads=8, d_ff=2048)}

_CACHE = {}


def _ensure_path():
    if not (TSLIB_DIR / "models" / "PatchTST.py").is_file():
        raise RuntimeError(
            f"Time-Series-Library not found at {TSLIB_DIR}.\n"
            f"Vendor it with:\n"
            f"  git clone https://github.com/thuml/Time-Series-Library.git "
            f"{TSLIB_DIR}\n"
            f"  cd {TSLIB_DIR} && git checkout {TSLIB_COMMIT}")
    p = str(TSLIB_DIR)
    if p not in sys.path:
        sys.path.append(p)                 # appended: `layers`/`utils` resolve, ours win
    return p


def _load_official(name):
    """Import one upstream model module, restoring `sys.modules['models']`."""
    if name in _CACHE:
        return _CACHE[name]
    p = _ensure_path()
    ours = sys.modules.pop("models", None)          # our models.py, fully loaded
    sys.path.insert(0, p)                            # TSLib's `models` must win now
    try:
        mod = importlib.import_module(f"models.{name}")
    finally:
        # Drop every `models*` entry this import created, then put ours back.
        for key in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
            del sys.modules[key]
        if ours is not None:
            sys.modules["models"] = ours
        with contextlib.suppress(ValueError):
            sys.path.remove(p)
        if p not in sys.path:
            sys.path.append(p)
    _CACHE[name] = mod
    return mod


def _config(n_ch, n_steps):
    """The argparse-style namespace `Model(configs)` reads, with library defaults
    except for the capacity knobs in `SCALE` and our seq_len/pred_len."""
    s = _SCALES[SCALE]
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=n_steps, label_len=0, pred_len=1,
        enc_in=n_ch, dec_in=n_ch, c_out=n_ch,
        e_layers=2, d_layers=1, factor=1, dropout=0.1, activation="gelu",
        embed="fixed", freq="d", moving_avg=25, num_class=1,
        output_attention=False, **s,
        # TimeMixer-specific (library defaults, except down-sampling which is
        # off upstream and is the model's whole point, so it is enabled).
        channel_independence=0, decomp_method="moving_avg",
        down_sampling_layers=2, down_sampling_method="avg", down_sampling_window=2,
        top_k=5, use_norm=1,
        # TimeFilter-specific (library defaults, except patch_len: upstream's 16
        # would leave a 30-bar window with a single patch, and the model's graph
        # is built ACROSS patches, so it would have nothing to route over).
        patch_len=6, alpha=0.1, top_p=0.5, pos=1,
    )


class TSLibNet(nn.Module):
    """Wraps an upstream `Model` so it behaves like the other nets here:
    (B, L, C) in, one scalar per row out."""

    def __init__(self, name, n_ch=3, n_steps=30):
        super().__init__()
        mod = _load_official(name)
        cfg = _config(n_ch, n_steps)
        # FEDformer prints its Fourier-block setup on construction; silence it so
        # a 5-asset x 5-fold sweep does not emit 25 copies.
        with contextlib.redirect_stdout(io.StringIO()):
            self.net = mod.Model(cfg)
        self.name, self.n_ch, self.label_len, self.pred_len = name, n_ch, 0, 1

    def forward(self, x):                                   # x: (B, L, C)
        dec = torch.zeros(x.shape[0], self.label_len + self.pred_len, self.n_ch,
                          dtype=x.dtype, device=x.device)
        out = self.net(x, None, dec, None)                  # (B, pred_len, C)
        return out[:, -1, 0]                                # supervised output slot


def factory(name):
    """-> cls(n_ch=..., n_steps=...) so models._NETS can hold it like a class."""
    def make(n_ch=3, n_steps=30):
        return TSLibNet(name, n_ch, n_steps)
    make.__name__ = f"TSLib_{name}"
    return make
