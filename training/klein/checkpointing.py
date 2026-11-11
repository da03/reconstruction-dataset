"""Atomic, resumable checkpoints at optimizer boundaries.

Checkpoints are trusted local training artifacts, not inference-only adapters.
The last two committed states are retained; an incomplete write never replaces
the latest pointer. Evaluation status is separate from the committed state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(root, step, state):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = 'step_%09d_%s' % (step, uuid.uuid4().hex[:8])
    staging = root / ('.' + name + '.tmp')
    staging.mkdir()
    payload = staging / 'state.pt'
    with payload.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    sha = digest(payload)
    atomic_json(staging / 'manifest.json', dict(step=step, sha256=sha, bytes=payload.stat().st_size))
    final = root / name
    os.replace(staging, final)
    atomic_json(root / 'latest.json', dict(directory=name, step=step))
    committed = sorted(root.glob('step_*'), key=lambda p: p.stat().st_mtime_ns)
    for old in committed[:-2]:
        shutil.rmtree(old)
    return final


def load_checkpoint(path):
    path = Path(path)
    if (path / 'latest.json').exists():
        path = path / json.loads((path / 'latest.json').read_text())['directory']
    meta = json.loads((path / 'manifest.json').read_text())
    payload = path / 'state.pt'
    if payload.stat().st_size != meta['bytes'] or digest(payload) != meta['sha256']:
        raise ValueError('Checkpoint checksum mismatch: ' + str(path))
    return torch.load(payload, map_location='cpu', weights_only=False), path


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys():
        raise ValueError('Trainable parameter names do not match checkpoint')
    with torch.no_grad():
        for name, param in params.items():
            if param.shape != state[name].shape or param.dtype != state[name].dtype:
                raise ValueError('Checkpoint tensor shape or dtype changed: ' + name)
            param.copy_(state[name])
