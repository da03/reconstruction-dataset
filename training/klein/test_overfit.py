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
from overfit import (advance_after_budget, check_authority, completion_status,
                     evaluate_checkpoint, flow_loss, training_case, validate_continuation)
from overfit_runtime import euler_sample, predict_velocity


class OverfitTests(unittest.TestCase):
    def test_stage_advance_preserves_failed_gate_through_checkpoint_and_completion(self):
        state = dict(stage=2,stage_step=6000,total=7800,
                     config=dict(budgets='2000,4000,6000,10000',threshold=.001))
        result = dict(stage='one_sigma_grid',stage_step=6000,step=7800,max_v=.0068)
        with tempfile.TemporaryDirectory() as root:
            cp=Path(root)/'source';cp.mkdir()
            with self.assertRaises(FileNotFoundError):
                advance_after_budget(state,cp)
            ck.atomic_json(cp/'evaluation.json',result)
            failure=advance_after_budget(state,cp)
            continuation=dict(unpassed_stages=[failure])
            ck.save_checkpoint(Path(root)/'next',7800,dict(continuation=continuation))
            restored,_=ck.load_checkpoint(Path(root)/'next')
            status=completion_status(restored['continuation'],9000)
            self.assertEqual(status['status'],'five_frame_stage_passed')
            self.assertFalse(status['unpassed_stages'][0]['passed'])
            self.assertEqual(status['unpassed_stages'][0]['max_v'],.0068)
            self.assertEqual(completion_status(None,9000)['status'],'all_stages_passed')
            for changed in (dict(state,stage=1),dict(state,stage=3),dict(state,stage_step=5900)):
                with self.assertRaises(ValueError):advance_after_budget(changed,cp)
            for changed in (dict(result,step=7700),dict(result,max_v=.0009),dict(result,max_v=float('nan'))):
                ck.atomic_json(cp/'evaluation.json',changed)
                with self.assertRaises(ValueError):advance_after_budget(state,cp)

    def test_continuation_changes_paths_and_budgets_without_changing_recipe(self):
        old = dict(data='old_data',out='old_out',budgets='2,4,6,10',lr=.001,world_size=2)
        new = dict(old,data='new_data',out='new_out',budgets='2,4,16,20')
        validate_continuation(old,new)
        for changed in (dict(new,lr=.002),dict(new,world_size=1),dict(new,budgets='2,4,5,20')):
            with self.assertRaises(ValueError):
                validate_continuation(old,changed)

    def test_revoked_authority_prevents_checkpoint_publication(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'grant.json'
            ck.atomic_json(p,dict(generation=3,selected_job='123',release_training=True))
            guard=lambda:check_authority(p,3,'123')
            guard()
            first=ck.save_checkpoint(Path(root)/'checkpoints',1,{'step':1},before_publish=guard)
            for grant in (dict(generation=4,selected_job='123',release_training=True),
                          dict(generation=3,selected_job='124',release_training=True),
                          dict(generation=3,selected_job='123',release_training=False)):
                ck.atomic_json(p,grant)
                with self.assertRaises(RuntimeError):
                    ck.save_checkpoint(Path(root)/'checkpoints',2,{'step':2},before_publish=guard)
                state,path=ck.load_checkpoint(Path(root)/'checkpoints')
                self.assertEqual(state['step'],1)
                self.assertEqual(path,first)

    def test_sampling_starts_from_reproducible_pure_noise(self):
        seen = []
        def velocity(x, sigma):
            seen.append((x.clone(), sigma))
            return torch.ones_like(x)
        shape = (1, 4, 8)
        out = euler_sample(velocity, shape, [1, .75, .25, 0], 19, 'cpu')
        initial = torch.randn(shape, generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(seen[0][0], initial, rtol=0, atol=0)
        torch.testing.assert_close(out, initial-1)
        self.assertEqual([s for _,s in seen], [1, .75, .25])
        for schedule in ([.9, 0], [1, .5], [1, .5, .6, 0]):
            with self.assertRaises(ValueError):
                euler_sample(velocity, shape, schedule, 19, 'cpu')

    def test_predict_uses_current_latent_and_condition_without_clean_target(self):
        captured = {}
        def model(**kw):
            captured.update(kw)
            return (kw['hidden_states']*2,)
        x = torch.randn(1, 3, 4)
        case = dict(ctx=torch.randn(1, 2, 4),ids=torch.zeros(1, 5, 4),
                    bias=torch.zeros(1, 1, 6, 6))
        pred = predict_velocity(model,case,x,.25,torch.ones(1,1,4),
                                torch.zeros(1,1,4),torch.float32)
        torch.testing.assert_close(pred,2*x,rtol=0,atol=0)
        torch.testing.assert_close(captured['hidden_states'][:,3:],case['ctx'])
        self.assertEqual(captured['timestep'].item(),.25)

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
