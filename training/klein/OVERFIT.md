# Training-set memorization diagnostics

`overfit.py` checks whether the renderer can fit a fixed training example before
increasing noise and data variation. It is a separate diagnostic entry point;
the historical `train_klein.py` experiments and adapter-only resume behavior
are unchanged. These diagnostics measure training fit, not generalization.

The four stages share one optimizer and one training lineage:

1. One target frame, fixed references, fixed Gaussian noise, sigma 0.5.
2. The same frame and references, with fresh noise at sigma 0.5.
3. The same frame at sigma 0.1, 0.25, 0.5, 0.75, and 0.9, with fresh noise.
4. Five fixed frames, covering every frame-by-sigma combination, with fresh noise.

Each stage must reach the configured maximum per-case velocity MSE before the
next starts. The defaults are 0.001 MSE and budgets of 2,000, 4,000, 6,000, and
10,000 optimizer updates respectively. Hitting a budget stops the run and records
the unmet threshold. These are diagnostic settings, not measured results.

The model starts from the specified immutable FLUX.2-Klein base revision and
fresh rank-128 LoRA parameters (alpha 32). No previously trained adapter is
loaded. The script verifies zero-initialized B factors, nonzero A factors and
the trainable parameter names, and records initial tensor hashes. Stage changes
retain the weights and Adam state learned within this run. The conditioning
retains aligned blockout tokens, part-index rotary positions, the analytic
reference-colour substrate and the soft part attention bias used in the
existing recipe. Reference dropout, augmentation, extra background penalties,
weight decay and learning-rate scheduling are disabled.

## Run

Choose an object with at least five training frames in the specified variant.
The frame subset is selected deterministically and recorded with source hashes.
Provide a full foundation-model commit SHA as `MODEL_REVISION`.

```sh
python training/klein/overfit.py \
  --data "$DATA" --out "$OUT" --object "$OBJECT" \
  --revision "$MODEL_REVISION" --preflight

torchrun --standalone --nproc-per-node=2 training/klein/overfit.py \
  --data "$DATA" --out "$OUT" --object "$OBJECT" \
  --revision "$MODEL_REVISION"
```

Set `HF_HUB_CACHE` to a prepopulated model cache when using offline compute
nodes. Run the same command with `--resume` to continue; configuration, package
versions, world size and data hashes must match. The output-directory lock
prevents concurrent writers. Use one shared output directory for all workers
of a run; independent copies of that directory do not coordinate training.

Each GPU processes one example per optimizer update, and gradients are averaged
across GPUs. In the first stage both ranks deliberately use the same example,
noise and sigma; the objective is unchanged by this duplicate computation.

## Outputs and recovery

`train.jsonl` logs loss, gradient norm and sampled parameter movement. Evaluation
uses fixed noise independent of training's RNG and reports each frame, sigma
and noise seed separately in `evaluations.jsonl`. For stages after the first,
four fixed noise draws are evaluated per frame and sigma. Changing the camera,
references or held-out objects is outside this test.

Before evaluation, an atomic, checksummed checkpoint records trainable weights,
Adam moments and counters, all process RNG states, the explicit noise generator,
stage and optimizer cursors, configuration, data manifest and pending evaluation.
The last two committed states are retained. The frozen foundation revision and
hashed source inputs reconstruct the immutable conditioning. Checkpoints occur
at completed optimizer boundaries, with no accumulated gradients to restore.
An interrupted evaluation is retried from the saved state before any further
optimizer update; an already committed evaluation is reused. `SIGUSR1` or
`SIGTERM` requests a save after the current update.

Target images, VAE reconstructions and one-step clean-latent estimates are saved
for inspection. The one-step estimates start from a noised training target;
they are not samples generated from pure noise. A successful loss check does
not establish visual generation quality or real-time performance.

The velocity objective is `v = noise - clean`, with
`noisy = (1 - sigma) * clean + sigma * noise`. A memorized target permits zero
velocity error; there is no unavoidable floor near one. Clean-latent MSE equals
velocity MSE times `sigma**2` in exact arithmetic, so it must not replace the
velocity diagnostic without reporting sigma.

CPU regression tests cover the full frame/sigma schedule, the zero-loss oracle,
exact Adam/RNG continuation, evaluation failure and retry, interrupted writes,
checksum rejection and tensor-shape rejection:

```sh
python -m unittest discover -s training/klein -p 'test_overfit.py' -v
```
