#!/usr/bin/env python3
"""Evaluate a native overfit checkpoint without changing any training state.

Reproduce fixed-noise validation, generate from pure noise, and compare
BF16 versus FP32 arithmetic on exactly the same weights and encoded inputs.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from checkpointing import atomic_json, digest, load_checkpoint, restore_trainable
from overfit import flow_loss
from overfit_runtime import adapter_config, cache_case, euler_sample, predict_velocity
from render import Codec, PROMPT, compute_empirical_mu
from train_klein import Frames


def image_mse(a, b):
    return float(np.mean(((np.asarray(a, np.float32) - np.asarray(b, np.float32)) / 255) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--precision-check', action='store_true')
    a = ap.parse_args()
    if a.frames not in range(1, 6) or a.steps < 2:
        ap.error('Expected 1..5 frames and at least two sampling steps')
    seeds = [int(s) for s in a.seeds.split(',')]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Use a new output directory; completed diagnostics are immutable')
    state, cp = load_checkpoint(a.checkpoint)
    config = state['config']
    versions = {k: importlib.metadata.version(k) for k in config['versions']}
    if versions != config['versions']:
        raise ValueError(f'Runtime differs from training: {versions}')
    ds = Frames(a.data, 'train', [config['object']], 3, config['res'],
                '%d-%d' % (config['variant'], config['variant']), max_frames=config['frames'])
    if ds.rows != state['manifest']['frames']:
        raise ValueError('Training frames changed')
    for name, sha in state['manifest']['files'].items():
        if digest(Path(a.data)/name) != sha:
            raise ValueError('Changed input: '+name)
    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    dev, dtype = torch.device('cuda'), torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = Flux2KleinPipeline.from_pretrained(config['model'], revision=config['revision'],
                                              torch_dtype=dtype)
    pipe.text_encoder.to(dev)
    with torch.no_grad():
        pe, tids = pipe.encode_prompt(prompt=PROMPT, device=dev, max_sequence_length=512)
    pipe.text_encoder.to('cpu')
    del pipe.text_encoder
    torch.cuda.empty_cache()
    pipe.vae.to(dev).eval().requires_grad_(False)
    model = pipe.transformer.to(dev).requires_grad_(False)
    model.add_adapter(adapter_config(model, config['rank'], config['alpha']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    restore_trainable(model, state['weights'])
    model.eval()
    # Optimizer/RNG state remains untouched in the committed source checkpoint.
    del state['optimizer'], state['ranks'], state['weights']
    codec = Codec(pipe, dev, dtype)
    items = [ds.item(i, rng=None) for i in range(a.frames)]
    cache = [cache_case(codec, item, pe.shape[1]) for item in items]
    report = dict(checkpoint=str(cp), manifest=json.loads((cp/'manifest.json').read_text()),
                  config=config, versions=versions, device=torch.cuda.get_device_name(),
                  total=state['total'], stage=state['stage'], samples=[], denoising=[],
                  target_used_for_sampling=False, sampling_steps=a.steps, sampling_seeds=seeds)
    gallery = []
    started = time.monotonic()
    with torch.no_grad():
        for i, (c, item) in enumerate(zip(cache, items)):
            target = item['target']
            vae = codec.decode(c['clean'], c['h'], c['w'])
            target.save(out/f'target_{i}.png')
            vae.save(out/f'vae_{i}.png')
            item['ctx'][0].save(out/f'blockout_{i}.png')
            row = [('target', target), ('VAE reconstruction', vae)]
            sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            sched.set_timesteps(a.steps, device=dev, sigmas=np.linspace(1, 1/a.steps, a.steps),
                                mu=compute_empirical_mu(c['clean'].shape[1], a.steps))
            schedule = sched.sigmas.detach().cpu().tolist()
            report['sampling_sigmas'] = schedule
            condition = {k: v for k, v in c.items() if k != 'clean'}
            for seed in seeds:
                torch.cuda.synchronize()
                t = time.monotonic()
                generated = euler_sample(
                    lambda x, s: predict_velocity(model, condition, x, s, pe, tids),
                    c['clean'].shape, schedule, seed, dev)
                im = codec.decode(generated, c['h'], c['w'])
                torch.cuda.synchronize()
                im.save(out/f'frame_{i}_seed_{seed}.png')
                row.append((f'pure noise, seed {seed}', im))
                record = dict(frame=i, seed=seed, seconds=time.monotonic()-t,
                    latent_mse=float((generated-c['clean']).square().mean()),
                    pixel_mse=image_mse(im,target), pixel_mse_to_vae=image_mse(im,vae),
                    vae_pixel_mse=image_mse(vae,target))
                report['samples'].append(record)
                atomic_json(out/'progress.json',report)
                print('SAMPLE '+json.dumps(record),flush=True)
            gallery.append(row)
        # Keep target encodings and weights fixed: FP32 modes isolate arithmetic
        # and input rounding, not a different VAE encoding or pretrained model.
        modes = ['bf16', 'fp32_bf16_input', 'fp32'] if a.precision_check else ['bf16']
        for mode in modes:
            if mode != 'bf16':
                model.float()
            for i, c in enumerate(cache):
                for sigma in sorted(set(config['sigmas'] + [.01, .05, 1.0])):
                    for seed in range(4):
                        gen = torch.Generator(device=dev).manual_seed(20000+i*10+seed)
                        noise = torch.randn(c['clean'].shape,generator=gen,device=dev)
                        noisy = (1-sigma)*c['clean']+sigma*noise
                        pred = predict_velocity(model,c,noisy,sigma,pe,tids,
                            torch.bfloat16 if mode=='bf16' else torch.float32,
                            round_input=mode=='fp32_bf16_input')
                        v,x0 = flow_loss(pred,c['clean'],noise,sigma)
                        rounded = noisy.to(torch.bfloat16).float()
                        # Oracle knows the memorized target but observes rounded x_t.
                        oracle = (rounded-c['clean'])/sigma
                        ov,_ = flow_loss(oracle,c['clean'],noise,sigma)
                        target_v = noise-c['clean']
                        oq = float((target_v.to(torch.bfloat16).float()-target_v).square().mean())
                        report['denoising'].append(dict(mode=mode,frame=i,sigma=sigma,seed=seed,
                            v_mse=float(v),x0_mse=float(x0),rounded_input_oracle_v_mse=float(ov),
                            rounded_output_oracle_v_mse=oq))
            atomic_json(out/'progress.json',report)
            print('PRECISION '+mode+' complete',flush=True)
    w,h = config['res'],config['res']+28
    sheet=Image.new('RGB',(w*len(gallery[0]),h*len(gallery)),'white')
    draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(gallery):
        for j,(label,im) in enumerate(row):
            sheet.paste(im.resize((w,w)),(j*w,i*h+28))
            draw.text((j*w+8,i*h+7),f'frame {i}: {label}',fill='black')
    sheet.save(out/'comparison.png')
    report['elapsed_seconds']=time.monotonic()-started
    report['complete']=True
    atomic_json(out/'complete.json',report)
    print('COMPLETE '+json.dumps(dict(step=state['total'],out=str(out))),flush=True)


if __name__=='__main__':
    main()
