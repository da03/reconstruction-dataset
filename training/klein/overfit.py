#!/usr/bin/env python3
"""Fit fixed frames before scaling data. Run with torchrun for multiple GPUs.

One lineage, four stages gated by default: fixed frame/noise/sigma; fresh noise;
multiple sigmas; five frames. Stage transitions retain this run's weights
and Adam state. Initialization is the named foundation checkpoint plus fresh
LoRA weights. No task-trained adapter is loaded on a fresh launch.
An explicit advance after the one-frame budget preserves its failed gate.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist

from checkpointing import (atomic_json, digest, load_checkpoint, restore_rng,
                           restore_trainable, rng_state, save_checkpoint, trainable_state)
from render import BASE, PROMPT, Codec, attention_bias, part_ids_from_map, substrate_image
from train_klein import Frames
from overfit_runtime import adapter_config, cache_case, predict_velocity

STAGES = ('one_fixed', 'one_fresh_noise', 'one_sigma_grid', 'five_sigma_grid')


def validate_continuation(previous, current):
    """Relocation and larger update budgets preserve the scientific recipe."""
    ignored = {'data', 'out', 'budgets'}
    if {k:v for k,v in previous.items() if k not in ignored} != {
            k:v for k,v in current.items() if k not in ignored}:
        raise ValueError('Continuation changed the training recipe')
    old = [int(n) for n in previous['budgets'].split(',')]
    new = [int(n) for n in current['budgets'].split(',')]
    if len(old) != 4 or len(new) != 4 or any(b < a for a,b in zip(old,new)):
        raise ValueError('Continuation budgets must not decrease')


def advance_after_budget(state, checkpoint):
    """Start the requested five-frame stage without relabeling a failed gate."""
    if state['stage'] != 2 or state['stage_step'] < int(state['config']['budgets'].split(',')[2]):
        raise ValueError('Explicit stage advance requires an exhausted one-frame sigma stage')
    result = json.loads((Path(checkpoint)/'evaluation.json').read_text())
    if (result['stage'] != STAGES[2] or result['step'] != state['total'] or
            result['stage_step'] != state['stage_step'] or
            not np.isfinite(result['max_v']) or
            result['max_v'] <= state['config']['threshold']):
        raise ValueError('Expected the committed evaluation of the failed source gate')
    return dict(stage=STAGES[2], step=state['total'], max_v=result['max_v'],
                threshold=state['config']['threshold'], passed=False,
                reason='Explicit advance to the five-frame experiment after the one-frame budget')


def completion_status(continuation, total):
    unpassed = (continuation or {}).get('unpassed_stages', [])
    return dict(status='five_frame_stage_passed' if unpassed else 'all_stages_passed',
                step=total, unpassed_stages=unpassed)


def check_authority(path, generation, job, command_file=None):
    if not path and not command_file:
        return
    if command_file:
        command = json.loads(Path(command_file).read_text())
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('Authority command must be a nonempty JSON argv list')
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        grant = json.loads(result.stdout)
    else:
        grant = json.loads(Path(path).read_text())
    if not (grant.get('release_training') is True and
            grant.get('generation') == generation and
            grant.get('selected_job') == job and job):
        raise RuntimeError('Training authority was revoked or belongs to another job')


def sigma_grid(text):
    values = [float(v) for v in text.split(',')]
    if not values or any(not 0 < v <= 1 for v in values):
        raise ValueError('Every sigma must be in (0, 1]')
    return values


def training_case(stage, step, rank, world, frames, sigmas):
    """Enumerate the complete frame-by-sigma product across all ranks."""
    position = step * world + rank
    if stage < 2:
        return 0, .5
    frame_count = frames if stage == 3 else 1
    position %= frame_count * len(sigmas)
    return position // len(sigmas), sigmas[position % len(sigmas)]


def evaluate_checkpoint(path, evaluate):
    """A failed evaluation leaves its committed training checkpoint intact."""
    path = Path(path)
    result_path = path / 'evaluation.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    result = evaluate()
    atomic_json(result_path, result)
    return result


def flow_loss(pred, clean, noise, sigma):
    target = noise - clean
    v = (pred.float() - target.float()).square().mean()
    noisy = (1 - sigma) * clean.float() + sigma * noise.float()
    x0 = (noisy - sigma * pred.float() - clean.float()).square().mean()
    return v, x0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=BASE)
    ap.add_argument('--revision', required=True, help='Immutable foundation model revision')
    ap.add_argument('--object', required=True)
    ap.add_argument('--variant', type=int, default=0)
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--rank', type=int, default=128)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sigmas', type=sigma_grid, default=sigma_grid('0.1,0.25,0.5,0.75,0.9'))
    ap.add_argument('--budgets', default='2000,4000,6000,10000')
    ap.add_argument('--threshold', type=float, default=1e-3,
                    help='Maximum per-case velocity MSE required to advance a stage')
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--resume-from', help='Continue full native state into a new output directory')
    ap.add_argument('--advance-to-five-frames', action='store_true',
                    help='With resume-from, explicitly leave an exhausted one-frame gate failed')
    ap.add_argument('--authority-file', help='Atomic job-specific grant checked before batches and saves')
    ap.add_argument('--authority-command-file',
                    help='JSON argv for reading the original cross-site grant on every check')
    ap.add_argument('--authority-generation', type=int)
    ap.add_argument('--preflight', action='store_true', help='Check selected files without loading a model')
    a = ap.parse_args()
    if a.resume and a.resume_from:
        ap.error('--resume and --resume-from are mutually exclusive')
    if a.advance_to_five_frames and not a.resume_from:
        ap.error('--advance-to-five-frames requires --resume-from')
    if a.authority_file and a.authority_command_file:
        ap.error('Choose a grant file or a grant-reading command')
    if bool(a.authority_file or a.authority_command_file) != (a.authority_generation is not None):
        ap.error('Provide an authority source and authority-generation together')
    def guard():
        check_authority(a.authority_file,a.authority_generation,os.getenv('SLURM_JOB_ID'),
                        a.authority_command_file)
    guard()
    if not re.fullmatch(r'[0-9a-f]{40}', a.revision):
        raise ValueError('Use a full immutable foundation commit SHA')
    budgets = [int(n) for n in a.budgets.split(',')]
    if len(budgets) != 4 or min(budgets) < 1 or a.frames != 5 or a.eval_every < 1:
        raise ValueError('Expected four positive budgets, five frames and positive evaluation interval')
    local_rank, rank, world = int(os.getenv('LOCAL_RANK', '0')), int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (out / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ds = Frames(a.data, 'train', [a.object], 3, a.res,
                '%d-%d' % (a.variant, a.variant), max_frames=a.frames)
    if len(ds.rows) != a.frames:
        raise ValueError('Need exactly five selected training frames')
    if not ds.objects[a.object].get('n_parts'):
        raise ValueError('This recipe requires a positive n_parts')
    files = set()
    for row in ds.rows:
        files.update(row[k] for k in ('tgt', 'ppr', 'mask'))
        refs, ids = ds.ref_files(row, with_ids=True)
        if any(p is None for p in ids):
            raise ValueError('Part-label maps are required by this recipe')
        files.update(refs + ids)
    manifest = dict(object=a.object, variant=a.variant, frames=ds.rows,
                    files={p: digest(Path(a.data)/p) for p in sorted(files)})
    manifest_path = out / 'data_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Selected training data changed')
    if rank == 0:
        atomic_json(manifest_path, manifest)
    if a.preflight:
        print(json.dumps(dict(preflight='passed', frames=len(ds.rows), files=len(files))), flush=True)
        return
    config_record = {k:v for k,v in vars(a).items() if k not in (
        'resume','resume_from','advance_to_five_frames','preflight','authority_file',
        'authority_command_file','authority_generation')}
    config_record['world_size'] = world
    config_record['versions'] = {
        k:importlib.metadata.version(k) for k in ('torch','diffusers','peft','transformers')}
    latest = out/'checkpoints/latest.json'
    if a.resume_from and (latest.exists() or (out/'config.json').exists()):
        raise ValueError('--resume-from requires a new output directory')
    if not a.resume and latest.exists():
        raise ValueError('Output already has a checkpoint; use --resume')
    if a.resume and not latest.exists():
        raise ValueError('Cannot resume without a complete checkpoint')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text()) != config_record:
            raise ValueError('Configuration differs from the existing run')
    elif rank == 0:
        atomic_json(out/'config.json',config_record)
    torch.cuda.set_device(local_rank)
    dev, dtype = torch.device('cuda', local_rank), torch.bfloat16
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype)
    pipe.text_encoder.to(dev)
    with torch.no_grad():
        pe, tids = pipe.encode_prompt(prompt=PROMPT, device=dev, max_sequence_length=512)
    pipe.text_encoder.to('cpu')
    del pipe.text_encoder
    torch.cuda.empty_cache()
    pipe.vae.to(dev).eval().requires_grad_(False)
    model = pipe.transformer.to(dev).requires_grad_(False)
    config = adapter_config(model, a.rank, a.alpha)
    model.add_adapter(config)
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in n for n in trainable):
        raise ValueError('Expected fresh LoRA parameters only')
    for name, param in trainable.items():
        if 'lora_B' in name and torch.count_nonzero(param).item():
            raise ValueError('LoRA B must be zero at fresh initialization')
        if 'lora_A' in name and not torch.count_nonzero(param).item():
            raise ValueError('LoRA A must have fresh random initialization')
    params = list(trainable.values())
    for p in params:
        p.data = p.data.float()
        if world > 1:
            dist.broadcast(p.data, 0)
    model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0)
    codec = Codec(pipe, dev, dtype)
    P = codec.P
    cache = []
    for i in range(a.frames):
        item = ds.item(i, rng=None)
        cache.append(cache_case(codec, item, pe.shape[1]))
        if rank == 0:
            item['target'].save(out/f'target_{i}.png')
            codec.decode(cache[-1]['clean'],cache[-1]['h'],cache[-1]['w']).save(out/f'vae_roundtrip_{i}.png')
    # Separate generators: validation never changes the optimization RNG stream.
    fixed = torch.Generator(device=dev).manual_seed(a.seed+12345)
    fixed_noise = torch.randn(cache[0]['clean'].shape,generator=fixed,device=dev)
    noise_rng = torch.Generator(device=dev).manual_seed(a.seed+1000+rank)
    if rank == 0 and not (a.resume or a.resume_from):
        atomic_json(out/'initialization.json',dict(model=a.model,revision=a.revision,
            task_parameters='fresh Gaussian LoRA A and verified zero LoRA B',
            trainable=sum(p.numel() for p in params),
            adapter_tensor_sha256={n:hashlib.sha256(
                p.detach().float().cpu().numpy().tobytes()).hexdigest()
                for n,p in trainable.items()},
            versions={k:importlib.metadata.version(k) for k in ('torch','diffusers','peft','transformers')}))
    stage, stage_step, total, pending, cp = 0, 0, 0, True, None
    continuation = None
    if a.resume or a.resume_from:
        state, cp = load_checkpoint(a.resume_from or out/'checkpoints')
        if a.resume_from:
            validate_continuation(state['config'],config_record)
        elif state['config'] != config_record:
            raise ValueError('Resume configuration differs from checkpoint')
        if state['manifest'] != manifest:
            raise ValueError('Resume configuration or data differs from checkpoint')
        restore_trainable(model,state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        stage,stage_step,total,pending=state['stage'],state['stage_step'],state['total'],state['pending_eval']
        restore_rng(state['ranks'][rank]['rng'])
        noise_rng.set_state(state['ranks'][rank]['noise_rng'].cpu())
        continuation = state.get('continuation')
        if a.resume_from:
            unpassed = list((continuation or {}).get('unpassed_stages', []))
            if a.advance_to_five_frames:
                unpassed.append(advance_after_budget(state, cp))
                stage, stage_step = 3, 0
            continuation = dict(source=str(cp), source_manifest=json.loads((cp/'manifest.json').read_text()),
                previous_config=state['config'], config=config_record, previous=continuation,
                unpassed_stages=unpassed)
            if rank==0:
                atomic_json(out/'continuation.json',continuation)
            # Save restored native state here before evaluating or updating it.
            cp=None
            pending=True
        print(f'RESTORED rank={rank} step={total} stage={stage} optimizer_states={len(optimizer.state)}',flush=True)
    stopping = [False]
    def stop(_sig,_frame): stopping[0]=True
    signal.signal(signal.SIGUSR1,stop)
    signal.signal(signal.SIGTERM,stop)

    def predict(case, noise, sigma):
        noisy=(1-sigma)*case['clean']+sigma*noise
        return predict_velocity(model,case,noisy,sigma,pe,tids,dtype)

    def commit():
        nonlocal cp
        guard()
        rs=dict(rng=rng_state(),noise_rng=noise_rng.get_state())
        ranks=[None]*world
        if world>1: dist.all_gather_object(ranks,rs)
        else: ranks=[rs]
        if rank==0:
            cp=save_checkpoint(out/'checkpoints',total,dict(weights=trainable_state(model),
                optimizer=optimizer.state_dict(),ranks=ranks,config=config_record,manifest=manifest,
                stage=stage,stage_step=stage_step,total=total,pending_eval=pending,
                continuation=continuation),before_publish=guard)
        names=[str(cp) if rank==0 else None]
        if world>1: dist.broadcast_object_list(names,src=0)
        cp=Path(names[0])

    def evaluate():
        model.eval()
        cases=[]
        count=5 if stage==3 else 1
        sigmas=[.5] if stage<2 else a.sigmas
        with torch.no_grad():
            for i in range(count):
                c=cache[i]
                for sigma in sigmas:
                    for seed in ([0] if stage==0 else [0,1,2,3]):
                        gen=torch.Generator(device=dev).manual_seed(20000+i*10+seed)
                        noise=fixed_noise if stage==0 else torch.randn(c['clean'].shape,generator=gen,device=dev)
                        pred=predict(c,noise,sigma)
                        v,x0=flow_loss(pred,c['clean'],noise,sigma)
                        if seed == 0 and sigma == .5:
                            noisy=(1-sigma)*c['clean']+sigma*noise
                            codec.decode(noisy-sigma*pred.float(),c['h'],c['w']).save(
                                cp/f'frame_{i}_one_step_x0.png')
                        cases.append(dict(frame=i,sigma=sigma,seed=seed,v_mse=float(v),x0_mse=float(x0)))
        model.train()
        return dict(stage=STAGES[stage],step=total,stage_step=stage_step,cases=cases,
                    mean_v=float(np.mean([c['v_mse'] for c in cases])),
                    max_v=max(c['v_mse'] for c in cases))

    def evaluate_committed():
        result=[None]
        if rank==0:
            result[0]=evaluate_checkpoint(cp,evaluate)
            with (out/'evaluations.jsonl').open('a') as f: f.write(json.dumps(result[0])+'\n')
            print('EVAL '+json.dumps({k:v for k,v in result[0].items() if k!='cases'}),flush=True)
        if world>1: dist.broadcast_object_list(result,src=0)
        return result[0]

    model.train()
    if cp is None: commit()
    started=time.monotonic()
    while stage<len(STAGES):
        guard()
        if pending:
            result=evaluate_committed()
            pending=False
            if stage_step and result['max_v']<=a.threshold:
                if rank==0:
                    atomic_json(out/(STAGES[stage]+'_passed.json'),result)
                stage+=1; stage_step=0
                pending=stage<len(STAGES)
                commit()
                if pending: continue
                break
            if stage_step>=budgets[stage]:
                if rank==0:
                    atomic_json(out/'status.json',dict(status='stopped_at_budget',stage=STAGES[stage],step=total,
                        max_v=result['max_v'],threshold=a.threshold))
                break
        halt=torch.tensor(int(stopping[0]),device=dev)
        if world>1: dist.all_reduce(halt,op=dist.ReduceOp.MAX)
        if halt.item():
            commit()
            if rank==0: atomic_json(out/'status.json',dict(status='stopped_for_signal',step=total))
            break
        i,sigma=training_case(stage,stage_step,rank,world,a.frames,a.sigmas)
        c=cache[i]
        noise=fixed_noise if stage==0 else torch.randn(c['clean'].shape,generator=noise_rng,device=dev)
        optimizer.zero_grad(set_to_none=True)
        pred=predict(c,noise,sigma)
        loss,_=flow_loss(pred,c['clean'],noise,sigma)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward()
        for p in params:
            if p.grad is None: raise RuntimeError('Missing adapter gradient')
            if world>1:
                dist.all_reduce(p.grad)
                p.grad.div_(world)
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        samples=[p.detach().flatten()[::max(1,p.numel()//64)].clone() for p in params]
        guard()
        optimizer.step()
        delta=float(torch.stack([(p.detach().flatten()[::max(1,p.numel()//64)]-old).square().sum()
                  for p,old in zip(params,samples)]).sum().sqrt())
        total+=1; stage_step+=1
        if rank==0 and (total<=5 or total%10==0):
            row=dict(step=total,stage=STAGES[stage],stage_step=stage_step,loss=float(loss.detach()),
                     frame=i,sigma=sigma,grad_norm=float(norm),sampled_parameter_delta=delta,lr=a.lr,
                     elapsed=time.monotonic()-started,gpu=torch.cuda.get_device_name(dev))
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print('TRAIN '+json.dumps(row),flush=True)
        if stage_step%a.eval_every==0 or stage_step==budgets[stage]:
            pending=True
            commit()  # Evaluation must never precede a complete resumable save.
    if stage==len(STAGES) and rank==0:
        atomic_json(out/'status.json',completion_status(continuation,total))
    if world>1: dist.destroy_process_group()


if __name__=='__main__':
    main()
