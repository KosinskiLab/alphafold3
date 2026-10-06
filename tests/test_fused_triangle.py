"""CPU contracts plus real Pallas interpret-mode adapter parity."""
import subprocess
import sys

from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from alphafold3.model import model_config
from alphafold3.model.components import utils
from alphafold3.model.network import modules
from alphafold3.model.network.fused_triangle import fpf_pallas_serve as dispatch


@pytest.fixture(autouse=True, scope='module')
def parse_absl_flags():
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(['test_fused_triangle'])


def config(on=True, cc='9.0', memory=76, attention='auto'):
    return model_config.GlobalConfig(
        final_init='linear', flash_attention_implementation='xla',
        fused_triangle_multiplication=on,
        fused_triangle_attention=attention if on else 'off',
        fused_triangle_compute_capability=cc, fused_triangle_memory_gib=memory)


@pytest.mark.parametrize('cc,memory,tiles,backend,limit', [
    ('8.0', 38, '8.0', 'pallas', 2048),
    ('8.6', 22, 'safe', 'tokamax', 1536),
    ('8.6', 45, 'safe', 'tokamax', 2048),
    ('8.9', 45, 'safe', 'tokamax', 2048),
    ('9.0', 76, '9.0', 'pallas', 3072),
    ('12.0', 91, '9.0', 'pallas', 3072),
    ('12.0', 15, '9.0', 'pallas', 768),
])
def test_device_and_memory_dispatch(cc, memory, tiles, backend, limit):
    cfg = config(cc=cc, memory=memory)
    policy = dispatch.policy_from_config(cfg)
    assert (policy.tiles, policy.attention, policy.attention_max_n) == (tiles, backend, limit)
    assert dispatch.select(cfg, 'attention', (limit, limit, 128), 'bfloat16', (limit, limit)) == (backend, '')
    n = limit + 64
    assert dispatch.select(cfg, 'attention', (n, n, 128), 'bfloat16', (n, n))[1] == 'size_limit'


@pytest.mark.parametrize('cc,memory', [('7.0', 32), ('10.0', 180), ('13.0', 96), ('', 80), ('9.0', 0), ('9.0', float('nan'))])
def test_unknown_devices_and_budgets_fall_back(cc, memory):
    assert not dispatch.card_policy(cc, memory).enabled


@pytest.mark.parametrize('shape,dtype,mask,heads,reason', [
    ((64, 64, 128), 'float32', (64, 64), 4, 'dtype_not_bfloat16'),
    ((65, 65, 128), 'bfloat16', (65, 65), 4, 'unvalidated_shape'),
    ((64, 64, 32), 'bfloat16', (64, 64), 4, 'unvalidated_shape'),
    ((64, 32, 128), 'bfloat16', (64, 32), 4, 'non_square_pair'),
    ((64, 64, 128), 'bfloat16', (64,), 4, 'mask_shape'),
    ((64, 64, 128), 'bfloat16', (64, 64), 8, 'unvalidated_heads'),
])
def test_fallbacks(shape, dtype, mask, heads, reason):
    assert dispatch.select(config(), 'attention', shape, dtype, mask, num_head=heads) == ('stock', reason)


def test_off_imports_no_fused_package():
    subprocess.run([sys.executable, '-c', '''
import sys
from alphafold3.model.network import modules
from alphafold3.model import model_config
assert model_config.GlobalConfig().fused_triangle_attention == 'off'
assert not any(n.startswith('alphafold3.model.network.fused_triangle') for n in sys.modules)
'''], check=True)


def transformed(variant, cfg):
    def fn(act, mask):
        with utils.bfloat16_context():
            if variant.startswith('trimul'):
                equation = 'ikc,jkc->ijc' if variant == 'trimul_out' else 'kjc,kic->ijc'
                return modules.TriangleMultiplication(
                    modules.TriangleMultiplication.Config(equation=equation), cfg, name='layer')(act, mask)
            return modules.GridSelfAttention(
                modules.GridSelfAttention.Config(), cfg,
                transpose=variant == 'att_end', name='layer')(act, mask)
    return hk.without_apply_rng(hk.transform(fn))


@pytest.mark.parametrize('channels', [64, 128])
@pytest.mark.parametrize('variant', ['trimul_out', 'trimul_in', 'att_start', 'att_end'])
def test_adapter_real_kernels_and_parameter_compatibility(monkeypatch, channels, variant):
    from jax.experimental import pallas as pl
    original = pl.pallas_call
    monkeypatch.setattr(pl, 'pallas_call', lambda *a, **kw: original(*a, **dict(kw, interpret=True)))
    key = jax.random.PRNGKey(channels)
    x = jax.random.normal(key, (64, 64, channels), jnp.float32).astype(jnp.bfloat16)
    valid = jnp.arange(64) < 49
    mask = (valid[:, None] & valid[None, :]).astype(jnp.bfloat16)
    stock = transformed(variant, config(False))
    fused = transformed(variant, config())
    params = stock.init(key, x, mask)
    fused_params = fused.init(key, x, mask)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(fused_params)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(fused_params)):
        np.testing.assert_array_equal(a, b)
    expected = stock.apply(params, x, mask).astype(jnp.float32)
    result = fused.apply(params, x, mask).astype(jnp.float32)
    assert bool(jnp.all(jnp.isfinite(result)))
    e, y = np.asarray(expected[:49, :49]), np.asarray(result[:49, :49])
    assert np.linalg.norm(e) > 1  # No zero-initialised-output parity shortcut.
    assert np.linalg.norm(e-y)/np.linalg.norm(e) < .035
    np.testing.assert_allclose(y, e, atol=.07, rtol=.06)
    np.testing.assert_array_equal(result, fused.apply(params, x, mask).astype(jnp.float32))
    # Padding is finite and cannot affect real outputs.
    altered = jnp.where(mask[:, :, None].astype(bool), x, jnp.bfloat16(7))
    np.testing.assert_array_equal(result[:49, :49], fused.apply(params, altered, mask)[:49, :49].astype(jnp.float32))
    if variant.startswith('att'):
        # The other supported execution path exercises the same adapter/layout.
        tokcore = transformed(variant, config(attention='tokamax'))
        result2 = tokcore.apply(params, x, mask).astype(jnp.float32)
        assert np.linalg.norm(np.asarray(result2[:49, :49])-e)/np.linalg.norm(e) < .035
    # A float32 activation must execute the unchanged body.
    np.testing.assert_array_equal(stock.apply(params, x.astype(jnp.float32), mask.astype(jnp.float32)),
                                  fused.apply(params, x.astype(jnp.float32), mask.astype(jnp.float32)))
