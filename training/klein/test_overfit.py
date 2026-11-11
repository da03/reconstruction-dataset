"""CPU regression tests: run python -m unittest discover -s training/klein."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import checkpointing as ck
from overfit import evaluate_checkpoint, flow_loss, training_case


class OverfitTests(unittest.TestCase):
    def test_every_frame_sees_every_sigma(self):
        sigmas = [.1, .25, .5, .75, .9]
        for world in (1, 2, 3, 5):
            seen = [training_case(3, step, rank, world, 5, sigmas)
                    for step in range(25) for rank in range(world)]
            self.assertEqual(set(seen), {(i, s) for i in range(5) for s in sigmas})
            self.assertEqual(len(set(seen[:25])), 25)

    def test_memorized_velocity_has_no_nonzero_floor(self):
        clean, noise = torch.randn(2, 8, 4), torch.randn(2, 8, 4)
        for sigma in (.001, .1, .5, .99, 1.):
            v, x0 = flow_loss(noise-clean, clean, noise, sigma)
            self.assertEqual(v.item(), 0.)
            self.assertLess(x0.item(), 1e-12)

    def test_exact_adam_and_rng_continuation(self):
        torch.manual_seed(17)
        random.seed(17)
        np.random.seed(17)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        noise_rng = torch.Generator().manual_seed(123)

        def update(m, o, gen):
            x = torch.randn(3, 4, generator=gen)
            y = torch.randn(3, 2) + random.random() + np.random.rand()
            o.zero_grad(set_to_none=True)
            (m(x)-y).square().mean().backward()
            o.step()

        update(model, opt, noise_rng)
        with tempfile.TemporaryDirectory() as root:
            cp = ck.save_checkpoint(root, 1, dict(weights=ck.trainable_state(model),
                optimizer=opt.state_dict(), rng=ck.rng_state(),
                noise_rng=noise_rng.get_state(), cursor=1, pending_eval=True))
            update(model, opt, noise_rng)
            expected = ck.trainable_state(model)
            resumed = torch.nn.Linear(4, 2)
            resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.01)
            state, _ = ck.load_checkpoint(root)
            ck.restore_trainable(resumed, state['weights'])
            resumed_opt.load_state_dict(state['optimizer'])
            ck.restore_rng(state['rng'])
            resumed_rng = torch.Generator()
            resumed_rng.set_state(state['noise_rng'])
            calls = []

            def fail_eval():
                calls.append('failed')
                raise RuntimeError('injected evaluation failure')

            with self.assertRaisesRegex(RuntimeError, 'injected evaluation'):
                evaluate_checkpoint(cp, fail_eval)
            recoverable, _ = ck.load_checkpoint(root)
            self.assertTrue(recoverable['pending_eval'])
            self.assertEqual(recoverable['cursor'], 1)
            self.assertFalse((cp/'evaluation.json').exists())

            def retry_eval():
                calls.append('retried')
                return {'complete': True, 'step': 1}

            evaluate_checkpoint(cp, retry_eval)
            evaluate_checkpoint(cp, fail_eval)  # Committed evaluation is reused.
            self.assertEqual(calls, ['failed', 'retried'])
            update(resumed, resumed_opt, resumed_rng)
            for name, value in expected.items():
                torch.testing.assert_close(value, ck.trainable_state(resumed)[name], rtol=0, atol=0)
            for before, after in zip(opt.state.values(), resumed_opt.state.values()):
                for key in before:
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            first = ck.save_checkpoint(root, 1, {'step': 1})
            with patch.object(ck.torch, 'save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    ck.save_checkpoint(root, 2, {'step': 2})
            state, path = ck.load_checkpoint(root)
            self.assertEqual(state['step'], 1)
            self.assertEqual(path, first)

    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cp = ck.save_checkpoint(root, 1, {'weights': torch.ones(2)})
            with (cp/'state.pt').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ck.load_checkpoint(root)

    def test_shape_mismatch_is_rejected(self):
        m = torch.nn.Linear(4, 2)
        state = ck.trainable_state(m)
        state['weight'] = torch.zeros(1, 4)
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            ck.restore_trainable(m, state)


if __name__ == '__main__':
    unittest.main()
