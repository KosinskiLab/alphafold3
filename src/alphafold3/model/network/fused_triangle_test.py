# Copyright 2026 European Molecular Biology Laboratory
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests the fused triangle adapters and their dispatch.

The kernels run in Pallas interpret mode, so the tests run on CPU.
"""

import functools
import math
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3.jax.fused_triangle import dispatch
from alphafold3.model import model_config
from alphafold3.model.components import utils
from alphafold3.model.network import modules
import haiku as hk
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

_NUM_TOKENS = 64
_NUM_VALID_TOKENS = 49
_VARIANTS = (
    'multiplication_outgoing',
    'multiplication_incoming',
    'attention_starting',
    'attention_ending',
)
_ATTENTION_VARIANTS = ('attention_starting', 'attention_ending')
_NUM_CHANNELS = (64, 128)

_pallas_call = pl.pallas_call


def _interpret_pallas_call(*args, **kwargs):
  return _pallas_call(*args, **{**kwargs, 'interpret': True})


def _global_config(
    *, fused: bool = True, attention: str = 'auto'
) -> model_config.GlobalConfig:
  return model_config.GlobalConfig(
      final_init='linear',
      flash_attention_implementation='xla',
      triangle_multiplication_implementation='pallas' if fused else 'default',
      triangle_attention_implementation=attention if fused else 'default',
      fused_triangle_compute_capability='9.0',
      fused_triangle_memory_gib=76,
  )


def _layer(
    variant: str, global_config: model_config.GlobalConfig
) -> hk.Transformed:
  """Returns one triangle module, applied in the bfloat16 context."""

  def forward(act, mask):
    with utils.bfloat16_context():
      if variant.startswith('multiplication'):
        if variant == 'multiplication_outgoing':
          equation = 'ikc,jkc->ijc'
        else:
          equation = 'kjc,kic->ijc'
        layer = modules.TriangleMultiplication(
            modules.TriangleMultiplication.Config(equation=equation),
            global_config,
            name='layer',
        )
      else:
        layer = modules.GridSelfAttention(
            modules.GridSelfAttention.Config(),
            global_config,
            transpose=variant == 'attention_ending',
            name='layer',
        )
      return layer(act, mask)

  return hk.without_apply_rng(hk.transform(forward))


def _inputs(num_channels: int) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns a PRNG key, bfloat16 activations and a mask with padding."""
  key = jax.random.PRNGKey(num_channels)
  act = jax.random.normal(
      key, (_NUM_TOKENS, _NUM_TOKENS, num_channels), jnp.float32
  ).astype(jnp.bfloat16)
  valid = jnp.arange(_NUM_TOKENS) < _NUM_VALID_TOKENS
  mask = (valid[:, None] & valid[None, :]).astype(jnp.bfloat16)
  return key, act, mask


def _valid(output: np.ndarray) -> np.ndarray:
  return output[:_NUM_VALID_TOKENS, :_NUM_VALID_TOKENS]


@functools.cache
def _default_params(variant: str, num_channels: int) -> hk.Params:
  key, act, mask = _inputs(num_channels)
  return _layer(variant, _global_config(fused=False)).init(key, act, mask)


@functools.cache
def _default_output(variant: str, num_channels: int) -> np.ndarray:
  _, act, mask = _inputs(num_channels)
  layer = _layer(variant, _global_config(fused=False))
  output = layer.apply(_default_params(variant, num_channels), act, mask)
  return np.asarray(output.astype(jnp.float32))


@functools.cache
def _fused_output(variant: str, num_channels: int) -> np.ndarray:
  """Must be called with Pallas in interpret mode."""
  _, act, mask = _inputs(num_channels)
  layer = _layer(variant, _global_config())
  output = layer.apply(_default_params(variant, num_channels), act, mask)
  return np.asarray(output.astype(jnp.float32))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
  return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


class DispatchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('sm80_38gib', '8.0', 38, '8.0', 'pallas', 2048),
      ('sm86_22gib', '8.6', 22, 'safe', 'pallas_tokamax_core', 1536),
      ('sm86_45gib', '8.6', 45, 'safe', 'pallas_tokamax_core', 2048),
      ('sm89_45gib', '8.9', 45, 'safe', 'pallas_tokamax_core', 2048),
      ('sm90_76gib', '9.0', 76, '9.0', 'pallas', 3072),
      ('sm120_91gib', '12.0', 91, '9.0', 'pallas', 3072),
      ('sm120_15gib', '12.0', 15, '9.0', 'pallas', 768),
  )
  def test_attention_dispatch_for_device(
      self,
      compute_capability,
      memory_gib,
      tile_table,
      implementation,
      max_tokens,
  ):
    policy = dispatch.device_policy(compute_capability, memory_gib)
    self.assertEqual(
        (
            policy.tile_table,
            policy.attention_implementation,
            policy.attention_max_tokens,
        ),
        (tile_table, implementation, max_tokens),
    )
    for num_tokens, expected in (
        (max_tokens, (implementation, '')),
        (max_tokens + 64, ('default', 'size_limit')),
    ):
      selected = dispatch.select_implementation(
          'triangle_attention',
          'auto',
          policy,
          (num_tokens, num_tokens, 128),
          'bfloat16',
          (num_tokens, num_tokens),
      )
      self.assertEqual(selected, expected)

  @parameterized.named_parameters(
      ('sm70', '7.0', 32),
      ('sm100', '10.0', 180),
      ('sm130', '13.0', 96),
      ('unknown_device', '', 80),
      ('zero_memory', '9.0', 0),
      ('nan_memory', '9.0', math.nan),
  )
  def test_unmeasured_device_disables_kernels(
      self, compute_capability, memory_gib
  ):
    policy = dispatch.device_policy(compute_capability, memory_gib)
    self.assertFalse(policy.enabled)

  @parameterized.named_parameters(
      ('float32', (64, 64, 128), 'float32', (64, 64), 4, 'dtype_not_bfloat16'),
      (
          'unaligned_tokens',
          (65, 65, 128),
          'bfloat16',
          (65, 65),
          4,
          'unvalidated_shape',
      ),
      (
          'unsupported_channels',
          (64, 64, 32),
          'bfloat16',
          (64, 64),
          4,
          'unvalidated_shape',
      ),
      (
          'non_square_pair',
          (64, 32, 128),
          'bfloat16',
          (64, 32),
          4,
          'non_square_pair',
      ),
      ('mask_shape', (64, 64, 128), 'bfloat16', (64,), 4, 'mask_shape'),
      ('num_head', (64, 64, 128), 'bfloat16', (64, 64), 8, 'unvalidated_heads'),
  )
  def test_unsupported_layer_runs_default(
      self, shape, dtype, mask_shape, num_head, reason
  ):
    policy = dispatch.device_policy('9.0', 76)
    selected = dispatch.select_implementation(
        'triangle_attention',
        'auto',
        policy,
        shape,
        dtype,
        mask_shape,
        num_head=num_head,
    )
    self.assertEqual(selected, ('default', reason))


class ImportTest(absltest.TestCase):

  def test_default_config_does_not_import_kernels(self):
    code = """
import sys
from alphafold3.model import model_config
from alphafold3.model.network import modules
global_config = model_config.GlobalConfig()
assert global_config.triangle_multiplication_implementation == 'default'
assert global_config.triangle_attention_implementation == 'default'
assert 'alphafold3.jax.fused_triangle.trimul_pallas' not in sys.modules
assert 'alphafold3.jax.fused_triangle.triattn_pallas' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True)


class AdapterTest(parameterized.TestCase):
  """Compares the real kernels with the original module bodies."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(pl, 'pallas_call', _interpret_pallas_call)
    )

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_init_matches_default(self, variant, num_channels):
    key, act, mask = _inputs(num_channels)
    fused_params = _layer(variant, _global_config()).init(key, act, mask)
    default_params = _default_params(variant, num_channels)
    self.assertEqual(
        jax.tree.structure(fused_params), jax.tree.structure(default_params)
    )
    for fused_leaf, default_leaf in zip(
        jax.tree.leaves(fused_params), jax.tree.leaves(default_params)
    ):
      np.testing.assert_array_equal(fused_leaf, default_leaf)

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_fused_config_runs_pallas_kernels(self, variant, num_channels):
    _, act, mask = _inputs(num_channels)
    layer = _layer(variant, _global_config())
    jaxpr = jax.make_jaxpr(layer.apply)(
        _default_params(variant, num_channels), act, mask
    )
    self.assertIn('pallas_call', str(jaxpr))

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_output_is_finite(self, variant, num_channels):
    self.assertTrue(np.all(np.isfinite(_fused_output(variant, num_channels))))

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_output_matches_default(self, variant, num_channels):
    expected = _valid(_default_output(variant, num_channels))
    actual = _valid(_fused_output(variant, num_channels))
    # The output projections are not zero-initialised, so parity is not
    # trivially met by two zero outputs.
    self.assertGreater(np.linalg.norm(expected), 1)
    self.assertLess(_relative_error(actual, expected), 0.035)
    np.testing.assert_allclose(actual, expected, atol=0.07, rtol=0.06)

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_output_is_deterministic(self, variant, num_channels):
    _, act, mask = _inputs(num_channels)
    layer = _layer(variant, _global_config())
    output = layer.apply(_default_params(variant, num_channels), act, mask)
    np.testing.assert_array_equal(
        output.astype(jnp.float32), _fused_output(variant, num_channels)
    )

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_padding_does_not_change_output(self, variant, num_channels):
    _, act, mask = _inputs(num_channels)
    padded_act = jnp.where(mask[:, :, None].astype(bool), act, jnp.bfloat16(7))
    layer = _layer(variant, _global_config())
    output = layer.apply(
        _default_params(variant, num_channels), padded_act, mask
    )
    np.testing.assert_array_equal(
        _valid(output.astype(jnp.float32)),
        _valid(_fused_output(variant, num_channels)),
    )

  @parameterized.product(
      variant=_ATTENTION_VARIANTS, num_channels=_NUM_CHANNELS
  )
  def test_tokamax_core_matches_default(self, variant, num_channels):
    _, act, mask = _inputs(num_channels)
    layer = _layer(variant, _global_config(attention='pallas_tokamax_core'))
    output = layer.apply(_default_params(variant, num_channels), act, mask)
    self.assertLess(
        _relative_error(
            _valid(np.asarray(output.astype(jnp.float32))),
            _valid(_default_output(variant, num_channels)),
        ),
        0.035,
    )

  @parameterized.product(variant=_VARIANTS, num_channels=_NUM_CHANNELS)
  def test_float32_runs_default_body(self, variant, num_channels):
    _, act, mask = _inputs(num_channels)
    act, mask = act.astype(jnp.float32), mask.astype(jnp.float32)
    params = _default_params(variant, num_channels)
    expected = _layer(variant, _global_config(fused=False)).apply(
        params, act, mask
    )
    output = _layer(variant, _global_config()).apply(params, act, mask)
    np.testing.assert_array_equal(output, expected)


if __name__ == '__main__':
  absltest.main()
