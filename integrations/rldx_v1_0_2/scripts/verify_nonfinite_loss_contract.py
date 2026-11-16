#!/usr/bin/env python3
"""Exercise RLDXTrainer's finite/non-finite total-loss contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock


def verify(rldx_root: Path) -> None:
    sys.path.insert(0, str(rldx_root))

    import torch
    from transformers import Trainer

    from rldx.experiment.trainer import RLDXTrainer

    trainer = RLDXTrainer.__new__(RLDXTrainer)
    trainer.state = SimpleNamespace(global_step=7)
    trainer.args = SimpleNamespace(logging_steps=1, local_rank=-1)
    trainer.action_offset = None
    model = SimpleNamespace(training=False)

    finite = torch.tensor(1.25)
    with mock.patch.object(Trainer, "compute_loss", return_value=(finite, {})):
        result = RLDXTrainer.compute_loss(
            trainer, model, {}, return_outputs=True, num_items_in_batch=None
        )
    if result[0] is not finite or result[1] != {} or trainer.loss is not finite:
        raise AssertionError("finite total loss did not pass through unchanged")

    for value in (float("nan"), float("inf"), float("-inf")):
        nonfinite = torch.tensor(value)
        with mock.patch.object(Trainer, "compute_loss", return_value=(nonfinite, {})):
            try:
                RLDXTrainer.compute_loss(
                    trainer, model, {}, return_outputs=True, num_items_in_batch=None
                )
            except FloatingPointError as exc:
                if "global_step=7" not in str(exc):
                    raise AssertionError("fail-fast error omitted the global step") from exc
            else:
                raise AssertionError(f"non-finite total loss {value!r} did not fail fast")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rldx_root", type=Path)
    args = parser.parse_args()
    verify(args.rldx_root.resolve())
    print("RLDX total-loss contract verified: finite passes; NaN/+Inf/-Inf fail fast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
