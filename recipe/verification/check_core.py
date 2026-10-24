"""CPU-only smoke checks of the bundled original functions; no Ray/model/GPU imports."""
from __future__ import annotations
import ast
from collections import defaultdict
from enum import Enum
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_functions(path, names, ns):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            body.append(node)
            found.add(node.name)
    if found != set(names):
        raise RuntimeError(f'Missing functions: {set(names)-found}')
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def main():
    ns = {'torch': torch, 'np': np, 'math': math, 'defaultdict': defaultdict, 'Enum': Enum}
    names = ['compute_opd_advantage', '_get_topd_alpha', '_compute_topd_immediate_rewards',
             '_compute_topd_token_returns', '_normalize_topd_token_returns_by_prompt',
             'compute_topd_advantage', 'compute_naive_topd_advantage', 'AdvantageEstimator']
    load_functions(ROOT/'verl/verl/trainer/ppo/core_algos.py', names, ns)
    old = torch.tensor([[-2., -3., -4.], [-3., -4., -2.]], dtype=torch.float64, device='cpu', requires_grad=True)
    ref = torch.tensor([[-1., -5., -4.], [-2., -3., -2.]], dtype=torch.float64, device='cpu', requires_grad=True)
    mask = torch.tensor([[1.,1.,0.],[1.,1.,1.]], dtype=torch.float64, device='cpu')
    args = dict(old_log_probs=old, ref_log_prob=ref, response_mask=mask,
                token_level_rewards=torch.zeros_like(old), index=np.array(['p','p']))
    opd, _ = ns['compute_opd_advantage'](**args)
    torch.testing.assert_close(opd, (ref-old)*mask)
    assert not opd.requires_grad
    for alpha in (.1,.2,1.):
        reward = ns['_compute_topd_immediate_rewards'](old.detach(),ref.detach(),mask,alpha)
        torch.testing.assert_close(reward, torch.log(alpha*(ref.detach()-old.detach()).exp()+1-alpha)*mask)
    rewards = torch.tensor([[1.,2.,3.,0.]], device='cpu')
    m = torch.tensor([[1.,1.,1.,0.]], device='cpu')
    torch.testing.assert_close(ns['_compute_topd_token_returns'](rewards,m),torch.tensor([[3.5,5.,3.,0.]],device='cpu'))
    full, returns = ns['compute_topd_advantage'](**args, config={'topd_alpha': .1})
    valid = full[mask.bool()]
    assert abs(valid.mean().item()) < 1e-8 and torch.isfinite(full).all()
    assert not full.requires_grad and not returns.requires_grad
    naive, _ = ns['compute_naive_topd_advantage'](**args, config={'topd_alpha': 1.})
    torch.testing.assert_close(naive,opd)
    full_one,_ = ns['compute_topd_advantage'](**args,config={'topd_alpha':1.})
    assert not torch.allclose(full_one,opd)
    try:
        ns['_get_topd_alpha']({'topd_alpha':0.})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid alpha accepted')
    zero = ns['_normalize_topd_token_returns_by_prompt'](torch.zeros(2,3),torch.zeros(2,3),None)
    assert torch.isfinite(zero).all() and zero.eq(0).all()
    metric_ns = {}
    load_functions(ROOT/'verl/verl/trainer/ppo/ray_trainer.py', ['_metric_rollout_n','_add_validation_metric_aliases'], metric_ns)
    metric = {}
    metric_ns['_add_validation_metric_aliases'](metric,'aime24','acc','mean@16',.5)
    assert metric['val/acc/aime24/avg@16']==.5 and metric['val-core/aime24/acc/avg@16']==.5
    print('Bundled original OPD/TOP-D numeric, mask, stop-gradient and avg@16 alias checks passed on CPU.')
    print('GPU training, kernels and environment installation were not run by this check.')


if __name__=='__main__':
    main()
