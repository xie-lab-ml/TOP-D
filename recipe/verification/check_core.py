"""CPU numeric and isolated AST regression checks; no model loading, Ray, or GPU required."""
from __future__ import annotations
import ast
from collections import defaultdict
from enum import Enum
import math
from pathlib import Path
import re
from types import SimpleNamespace
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


def check_fsdp_tokenizer_paths():
    path = ROOT/'verl/verl/workers/fsdp_workers.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    workers = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ActorRolloutRefWorker']
    assert len(workers) == 1, 'Expected exactly one ActorRolloutRefWorker'
    methods = [node for node in workers[0].body
               if isinstance(node, ast.FunctionDef) and node.name == '_build_model_optimizer']
    assert len(methods) == 1, 'Expected exactly one _build_model_optimizer'
    method = methods[0]

    def assignment_index(target):
        matches = [i for i, node in enumerate(method.body)
                   if isinstance(node, ast.Assign)
                   and any(ast.unparse(t) == target for t in node.targets)]
        assert len(matches) == 1, f'Expected one top-level assignment to {target}'
        return matches[0]

    start = assignment_index('local_path')
    tokenizer = assignment_index('self.tokenizer')
    processor = assignment_index('self.processor')
    generation = assignment_index('self.generation_config')
    assert start < tokenizer < processor < generation, 'Path/tokenizer assignment order changed'
    # Extract the original contiguous path-selection branch, not a reimplementation.
    body = method.body[start:processor + 1] + [method.body[generation]]
    fragment = ast.Module(body=body, type_ignores=[])
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(fragment))
    selected = {id(node) for node in ast.walk(fragment)}
    for node in ast.walk(method):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                and node.id in {'local_path', 'tokenizer_path'}):
            assert id(node) in selected, f'Path assignment omitted from fragment: {path}:{node.lineno}'

    for owner, keyword in [('AutoConfig', None),
                           ('actor_module_class', 'pretrained_model_name_or_path')]:
        calls = [node for node in ast.walk(method)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == owner
                 and node.func.attr == 'from_pretrained']
        assert len(calls) == 1, f'Expected one {owner}.from_pretrained call'
        args = calls[0].args[:1] if keyword is None else [
            item.value for item in calls[0].keywords if item.arg == keyword]
        assert len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == 'local_path', (
            f'{owner}.from_pretrained must still use local_path')
    code = compile(ast.fix_missing_locations(fragment), str(path), 'exec')
    teacher = '/stub/teacher-weights'
    student = '/stub/student-tokenizer'
    cached = '/stub/cache/student-tokenizer'
    cases = [
        ('ref explicit student path', 'ref', {'model': {'tokenizer_path': student}}, {}, cached, False),
        ('ref explicit path with use_shm', 'ref', {'model': {'tokenizer_path': student}},
         {'use_shm': True}, cached, True),
        ('ref missing model config', 'ref', {}, {}, teacher, None),
        ('ref null model', 'ref', {'model': None}, {}, teacher, None),
        ('ref empty model', 'ref', {'model': {}}, {}, teacher, None),
        ('ref missing tokenizer override', 'ref', {'model': {'path': teacher}}, {}, teacher, None),
        ('ref null override', 'ref', {'model': {'tokenizer_path': None}}, {}, teacher, None),
        ('ref empty-string override', 'ref', {'model': {'tokenizer_path': ''}}, {}, teacher, None),
        ('actor ignores ref override', 'actor', {'model': {'tokenizer_path': student}}, {}, teacher, None),
    ]
    for label, role, ref_config, model_config, expected_path, expected_shm in cases:
        calls = []

        def copy_stub(path, *, use_shm):
            calls.append(('copy_to_local', path, use_shm))
            return cached

        def loader_stub(name):
            def load(path, *, trust_remote_code):
                calls.append((name, path, trust_remote_code))
                return SimpleNamespace(loader=name, path=path)
            return load

        worker = SimpleNamespace(config=SimpleNamespace(ref=ref_config, model=model_config))
        ns = {'__builtins__': {}, 'self': worker, 'role': role, 'model_path': teacher,
              'trust_remote_code': True, 'copy_to_local': copy_stub,
              'hf_tokenizer': loader_stub('hf_tokenizer'),
              'hf_processor': loader_stub('hf_processor'),
              'get_generation_config': loader_stub('get_generation_config')}
        exec(code, ns)
        assert ns['local_path'] == teacher and ns['model_path'] == teacher, f'{label}: weight path changed'
        assert ns['tokenizer_path'] == expected_path, f'{label}: wrong tokenizer path'
        expected_calls = [] if expected_shm is None else [('copy_to_local', student, expected_shm)]
        expected_calls += [(name, expected_path, True)
                           for name in ('hf_tokenizer', 'hf_processor', 'get_generation_config')]
        assert calls == expected_calls, f'{label}: unexpected calls: {calls!r}'
        for attr, loader in [('tokenizer', 'hf_tokenizer'), ('processor', 'hf_processor'),
                             ('generation_config', 'get_generation_config')]:
            result = getattr(worker, attr)
            assert (result.loader, result.path) == (loader, expected_path), f'{label}: wrong self.{attr}'
        print(f'FSDP AST passed: {label}')
    print('FSDP static checks passed: AutoConfig and model weights still use local_path.')


def check_startup_tokenizer_assertions(common_path, common):
    blocks = re.findall(r'^python3 - <<PY[ \t]*\n(.*?)^PY[ \t]*$', common, re.M | re.S)
    assert len(blocks) == 1, 'Expected one Python startup heredoc in common'
    tree = ast.parse(blocks[0], filename=f'{common_path}:heredoc')
    tokenizers = {'student_tokenizer', 'ref_tokenizer'}
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
                  and any(isinstance(item, ast.Name) and item.id in tokenizers
                          for item in ast.walk(node.test))]
    assert len(assertions) == 3, 'Expected three startup tokenizer compatibility assertions'
    by_field = {}
    for node in assertions:
        names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
        assert names & tokenizers == tokenizers, 'Assertion must check both student and ref'
        fields = {item.attr for item in ast.walk(node.test) if isinstance(item, ast.Attribute)
                  and item.attr in {'eos_token_id', 'pad_token_id'}}
        if 'len' in names:
            fields.add('len')
        assert len(fields) == 1, 'Each assertion must check only EOS, PAD, or len'
        field = fields.pop()
        assert field not in by_field, f'Duplicate startup assertion: {field}'
        by_field[field] = node
    assert set(by_field) == {'eos_token_id', 'pad_token_id', 'len'}
    # Compile only source assertions; skip imports, file reads, and real AutoTokenizer calls.
    code = compile(ast.fix_missing_locations(ast.Module(body=assertions, type_ignores=[])),
                   f'{common_path}:heredoc', 'exec')

    class FakeTokenizer:
        def __init__(self, eos, pad, size):
            self.eos_token_id = eos
            self.pad_token_id = pad
            self.size = size

        def __len__(self):
            return self.size

    cases = [('all equal', (2, 0, 100), None),
             ('EOS mismatch', (3, 0, 100), 'eos_token_id'),
             ('PAD mismatch', (2, 1, 100), 'pad_token_id'),
             ('len mismatch', (2, 0, 101), 'len')]
    for label, ref_values, failing_field in cases:
        ns = {'__builtins__': {'len': len}, 'student_tokenizer': FakeTokenizer(2, 0, 100),
              'ref_tokenizer': FakeTokenizer(*ref_values)}
        try:
            exec(code, ns)
        except AssertionError as error:
            if failing_field is None:
                raise
            traceback = error.__traceback__
            while traceback.tb_next is not None:
                traceback = traceback.tb_next
            assert traceback.tb_lineno == by_field[failing_field].lineno, f'{label}: wrong assertion failed'
            print(f'startup AST passed: {label} rejected as expected ({error})')
        else:
            assert failing_field is None, f'startup incorrectly accepted {label}'
            print(f'startup AST passed: {label}')


def check_entry_batches(common_path, common):
    variables = ('TRAIN_BATCH_SIZE', 'PPO_MINI_BATCH_SIZE',
                 'PPO_MICRO_BATCH_SIZE_PER_GPU', 'LOG_PROB_MICRO_BATCH_SIZE_PER_GPU')
    entries = [('opd_16k_kl0', 'OPD', (128, 128, 1, 1)),
               ('topd_alpha0p1_16k_kl0', 'TOP-D', (128, 32, 1, 1)),
               ('naive_topd_alpha0p1_16k_kl0', 'naive TOP-D', (128, 32, 1, 1))]
    for suffix, label, expected in entries:
        path = common_path.parent/f'run_qwen3_1p7b_base_qwen3_30a3b_teacher_dapo17k_{suffix}.sh'
        source = path.read_text(encoding='utf-8')
        source_call = f'source "${{SCRIPT_DIR}}/{common_path.name}"'
        assert source.splitlines().count(source_call) == 1, f'{path.name}: expected one common source call'
        for variable, value in zip(variables, expected):
            matches = list(re.finditer(rf'^[ \t]*{variable}=([^\n]*)$', source, re.M))
            assert len(matches) == 1 and matches[0].group(1).strip() == str(value), (
                f'{path.name}: {variable} must be assigned once to {value}')
            assert matches[0].start() < source.index(source_call), f'{variable} must be set before sourcing common'
        print(f'batch static checks passed: {label} train/mini/PPOmicro/LOGmicro={expected}')

    required = re.findall(r'^required_entry_vars=\(\s*(.*?)^\)', common, re.M | re.S)
    assert len(required) == 1, 'Expected one required_entry_vars array in common'
    required_vars = re.sub(r'#[^\n]*', '', required[0]).split()
    for variable in variables:
        assert required_vars.count(variable) == 1, f'common required variable missing or duplicated: {variable}'
    guard = re.search(r'^for name in "\$\{required_entry_vars\[@\]\}"; do\n(.*?)^done$',
                      common, re.M | re.S)
    assert guard is not None, 'common must iterate over required variables'
    assert re.search(r'if \[ -z "\$\{!name:-\}" \]; then\s+.*?\bexit 1\s+fi',
                     guard.group(1), re.S), 'common must reject empty required variables'

    command = re.search(r'^python3 -m verl\.trainer\.main_ppo \\\n(?:[^\n]*\\\n)*[^\n]*',
                        common, re.M)
    assert command is not None, 'common training command not found (static check only)'
    forwarding = {
        'data.train_batch_size': 'TRAIN_BATCH_SIZE',
        'actor_rollout_ref.actor.ppo_mini_batch_size': 'PPO_MINI_BATCH_SIZE',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 'PPO_MICRO_BATCH_SIZE_PER_GPU',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 'LOG_PROB_MICRO_BATCH_SIZE_PER_GPU',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 'LOG_PROB_MICRO_BATCH_SIZE_PER_GPU',
        'actor_rollout_ref.model.path': 'MODEL_PATH',
        '+actor_rollout_ref.ref.model.path': 'TEACHER_MODEL_PATH',
        '+actor_rollout_ref.ref.model.tokenizer_path': 'REF_TOKENIZER_PATH',
    }
    for key, variable in forwarding.items():
        values = re.findall(rf'^[ \t]*{re.escape(key)}=([^\n]*?)[ \t]*\\$', command.group(0), re.M)
        assert values == [f'"${{{variable}}}"'], f'common Hydra forwarding wrong or duplicated: {key}={values!r}'
    defaults = re.findall(r'^REF_TOKENIZER_PATH=([^\n]*)$', common[:common.index('cat <<EOF')], re.M)
    assert defaults == ['${REF_TOKENIZER_PATH:-${MODEL_PATH}}'], 'REF_TOKENIZER_PATH must default to MODEL_PATH'
    print('common static checks passed: four required variables and empty-value guards, five batch Hydra mappings, student/teacher/tokenizer paths.')


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
    check_fsdp_tokenizer_paths()
    common_path = ROOT/'recipe/opd/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh'
    common = common_path.read_text(encoding='utf-8')
    check_startup_tokenizer_assertions(common_path, common)
    check_entry_batches(common_path, common)
    print('All CPU numeric and source/config checks passed; no model loading, Ray, GPU training/kernels, or dependency installation was run.')


if __name__=='__main__':
    main()
