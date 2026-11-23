"""Shared model inputs for training-fit diagnostics and sampling."""
from contextlib import nullcontext

import torch

from render import attention_bias, part_ids_from_map, substrate_image


def adapter_config(model, rank, alpha):
    from peft import LoraConfig
    return LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0', 'to_qkv_mlp_proj'] +
        [f'single_transformer_blocks.{i}.attn.to_out'
         for i in range(len(model.single_transformer_blocks))])


def cache_case(codec, item, prompt_tokens):
    clean = codec.encode(item['target'])
    pack = codec.encode(item['ctx'][0])
    refs = [codec.encode(im) for im in item['ctx'][1:]]
    h, w = clean.shape[-2:]
    n = item['n_parts']
    ids = codec.target_ids(clean, item['ctx'][0], n)
    labels = [part_ids_from_map(im, h, w, 0, n) for im in item['ref_ids']]
    sub = codec.encode(substrate_image(item['ctx'][0], item['ctx'][1:], item['ref_ids'], n))
    ctx, ci = codec.context(pack, refs, ids, 'aligned', labels, [sub])
    bias = attention_bias(part_ids_from_map(item['ctx'][0], h, w, 2, n)[0],
                          [x[0] for x in labels], prompt_tokens, 2,
                          device=codec.dev, dtype=codec.wdt)
    return dict(clean=codec.P._pack_latents(clean).float(), ids=torch.cat([ids, ci], 1),
                ctx=ctx, bias=bias, h=h, w=w)


def predict_velocity(model, case, noisy, sigma, pe, tids, dtype=torch.bfloat16,
                     round_input=False):
    """No target image is read here; noisy may be pure Gaussian noise."""
    if round_input:
        noisy = noisy.to(torch.bfloat16).float()
    amp = torch.autocast('cuda', dtype=dtype) if dtype != torch.float32 else nullcontext()
    with amp:
        result = model(hidden_states=torch.cat([noisy.to(dtype), case['ctx'].to(dtype)], 1),
            timestep=torch.as_tensor([sigma], device=noisy.device, dtype=torch.float32),
            encoder_hidden_states=pe.to(dtype), txt_ids=tids, img_ids=case['ids'],
            joint_attention_kwargs={'attention_mask': case['bias'].to(dtype)},
            return_dict=False)[0]
    return result[:, :noisy.shape[1]]


def euler_sample(predict, shape, sigmas, seed, device):
    """Generate from noise. The interface accepts no target pixels or latents."""
    sigmas = [float(s) for s in sigmas]
    if len(sigmas) < 2 or abs(sigmas[0] - 1) > 1e-6 or sigmas[-1] != 0:
        raise ValueError('Sampling must start at sigma=1 and finish at sigma=0')
    if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError('Sigmas must decrease strictly')
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    for current, following in zip(sigmas, sigmas[1:]):
        x = x + (following - current) * predict(x, current).float()
    return x
