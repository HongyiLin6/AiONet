"""Stable epoch seeding and portable checkpoint writes."""
from pathlib import Path
import numpy as np


def seed_epoch(seed,epoch):
    import torch
    value=int(seed)+int(epoch)
    np.random.seed(value);torch.manual_seed(value)


def save_checkpoint(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    torch.save(state,temporary);temporary.replace(path)
