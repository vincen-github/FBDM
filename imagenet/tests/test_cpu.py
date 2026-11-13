"""CPU checks; these do not substitute for the four-GPU ImageNet smoke run."""
import ast
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from methods.dm import DM
from methods.utils import repeated_center_pool
from diagnostic_adapters import assert_loss_semantics


class ImageNetRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run.torch.set_num_threads(1)
        cls.policy, cls.cfg, cls.centers = run.input_policy('imagenet')
        cls.model = run.build(cls.cfg, cls.centers).cpu()

    def test_recipe_and_initialization(self):
        self.assertEqual(len(run.digest(self.model.named_parameters())), 64)
        self.assertEqual(self.cfg.velocity_layers, 20)
        self.assertEqual(sum(isinstance(m, run.torch.nn.Linear) for m in self.model.v_net.modules()), 20)
        self.assertEqual(self.cfg.bs, 512)
        self.assertEqual(self.cfg.pool_extra_repeats, 4)
        self.assertEqual(self.cfg.warmup_steps, 10 * 2502)
        self.assertEqual(self.cfg.lambda_param, 10)
        self.assertEqual(self.cfg.fm_weight, 1.3)
        self.assertFalse(any(k.startswith(("assignment_queue", "assignment_epoch_global", "assignment_global_")) for k in vars(self.cfg)))
        self.assertEqual(repeated_center_pool(self.centers, 512, 1, 4).shape[0], 5 * 4096)

    def test_assignment_solver(self):
        self.assertEqual(run.exact_assignment.self_test(), 24)

    def test_fp32_velocity_under_autocast(self):
        z = run.torch.randn(2, 512, dtype=run.torch.bfloat16, requires_grad=True)
        t = run.torch.rand(2, 1).to(run.torch.bfloat16)
        with run.torch.autocast('cpu', dtype=run.torch.bfloat16):
            out = self.model.v_net(z, t)
        self.assertEqual(out.dtype, run.torch.float32)
        out.square().mean().backward()
        self.assertTrue(run.torch.isfinite(z.grad).all())
        self.model.zero_grad(set_to_none=True)

    def test_optimizer(self):
        opt = run.optimizer(self.model, self.cfg)
        opt.load_state_dict(opt.state_dict())
        self.assertIs(opt.param_groups, opt.optim.param_groups)

    def test_cpu_training_objective(self):
        model = self.model
        # Exercise the same objective with local matching; distributed collectives
        # require the separate four-GPU smoke command documented in README.
        model.assign_targets = types.MethodType(DM.assign_targets, model)
        model.train()
        model.reset_epoch_stats(1)
        views = [run.torch.randn(2, 3, 64, 64) for _ in range(2)]
        loss = model(views)
        assert_loss_semantics(model, self.cfg, loss)
        loss.backward()
        self.assertTrue(run.torch.isfinite(loss))
        self.assertTrue(all(run.torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.zero_grad(set_to_none=True)

    def test_removed_implementations(self):
        self.assertFalse((ROOT / 'logical_assignment_batch.py').exists())
        self.assertFalse((ROOT / 'velocity_constraints.py').exists())
        for path in ROOT.rglob('*.py'):
            if 'tests' in path.parts:
                continue
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            self.assertNotIn('energy_penalty', path.read_text(encoding='utf-8'))
            for removed in ('assignment_queue', 'assignment_epoch_global', '_epoch_global_pool'):
                self.assertNotIn(removed, path.read_text(encoding='utf-8'))
        source = (ROOT / 'run.py').read_text()
        self.assertIn('enumerate(loader)', source)
        self.assertNotIn('logical_batches', source)


if __name__ == '__main__':
    unittest.main()

