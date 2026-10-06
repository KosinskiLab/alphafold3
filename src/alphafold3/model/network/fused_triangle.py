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

"""Fused Pallas implementations of the triangle modules of the pair stack.

The functions here run the kernels in alphafold3.jax.fused_triangle on the
parameters that TriangleMultiplication and GridSelfAttention create. They
return None whenever the original module body must run instead: during
initialisation, so that parameters are created exactly as before, and for every
layer that dispatch does not select a kernel for. The kernel modules import the
Pallas Triton backend, so they are imported only once a kernel is selected.
"""

from alphafold3.jax.fused_triangle import dispatch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
import haiku as hk
import jax
import jax.numpy as jnp
import tokamax

_SUPPORTED_EQUATIONS = ('ikc,jkc->ijc', 'kjc,kic->ijc')


def _device_policy(
    global_config: model_config.GlobalConfig,
) -> dispatch.DevicePolicy:
  return dispatch.device_policy(
      global_config.fused_triangle_compute_capability,
      global_config.fused_triangle_memory_gib,
  )


def _layer_norm_params(
    name: str, num_channels: int
) -> tuple[jax.Array, jax.Array]:
  """Returns the scale and offset of the hm.LayerNorm called `name`."""
  with hk.name_scope(name):
    scale = hk.get_parameter('scale', (num_channels,), jnp.float32)
    offset = hk.get_parameter('offset', (num_channels,), jnp.float32)
  return scale, offset


def _linear_weights(
    name: str, shape: tuple[int, ...], dtype: jnp.dtype
) -> jax.Array:
  """Returns the weights of the hm.Linear called `name`."""
  with hk.name_scope(name):
    return hk.get_parameter('weights', shape, dtype)


def triangle_multiplication(
    act: jax.Array,
    mask: jax.Array,
    *,
    equation: str,
    use_glu_kernel: bool,
    global_config: model_config.GlobalConfig,
) -> jax.Array | None:
  """Applies TriangleMultiplication with the fused Pallas kernels.

  Must be called inside TriangleMultiplication, whose parameters it reads.

  Args:
    act: Pair activations, shape [N, N, C].
    mask: Pair mask, shape [N, N].
    equation: The einsum equation of the module.
    use_glu_kernel: Whether the module uses the gated linear unit kernel.
    global_config: The global config.

  Returns:
    The module output, shape [N, N, C], or None if the original module body
    must run instead.
  """
  policy = _device_policy(global_config)
  implementation, _ = dispatch.select_implementation(
      'triangle_multiplication',
      global_config.triangle_multiplication_implementation,
      policy,
      act.shape,
      act.dtype,
      mask.shape,
  )
  if (
      implementation == 'default'
      or not use_glu_kernel
      or mask.dtype != act.dtype
      or equation not in _SUPPORTED_EQUATIONS
  ):
    return None
  if hk.running_init():
    # Create the parameters with the original initialisers, order and RNG.
    return None

  # pylint: disable=g-import-not-at-top
  from alphafold3.jax.fused_triangle import trimul_pallas
  # pylint: enable=g-import-not-at-top

  num_channels = act.shape[-1]
  left_norm_scale, left_norm_offset = _layer_norm_params(
      'left_norm_input', num_channels
  )
  center_norm_scale, center_norm_offset = _layer_norm_params(
      'center_norm', num_channels
  )
  weights_projection, _ = hm.haiku_linear_get_params(
      act, num_output=num_channels * 2, name='projection'
  )
  weights_gate, _ = hm.haiku_linear_get_params(
      act, num_output=num_channels * 2, name='gate'
  )
  weights_output, _ = hm.haiku_linear_get_params(
      act, num_output=num_channels, name='output_projection'
  )
  weights_gating_linear, _ = hm.haiku_linear_get_params(
      act, num_output=num_channels, name='gating_linear'
  )
  params = {
      'ln_in_scale': left_norm_scale,
      'ln_in_offset': left_norm_offset,
      'ln_c_scale': center_norm_scale,
      'ln_c_offset': center_norm_offset,
      'w_proj': weights_projection,
      'w_gate': weights_gate,
      'w_out': weights_output,
      'w_gl': weights_gating_linear,
  }
  return trimul_pallas.triangle_multiplication_fused(
      act,
      mask,
      params,
      equation=equation,
      cfg=dispatch.triangle_multiplication_tiles(policy),
  )


def grid_self_attention(
    act: jax.Array,
    pair_mask: jax.Array,
    *,
    num_head: int,
    transpose: bool,
    global_config: model_config.GlobalConfig,
) -> jax.Array | None:
  """Applies GridSelfAttention with the fused Pallas kernels.

  Must be called inside GridSelfAttention, whose parameters it reads.

  Args:
    act: Pair activations, shape [N, N, C].
    pair_mask: Pair mask, shape [N, N].
    num_head: Number of attention heads.
    transpose: The transpose flag of the module; True attends around the
      ending node.
    global_config: The global config.

  Returns:
    The module output, shape [N, N, C], or None if the original module body
    must run instead.
  """
  policy = _device_policy(global_config)
  implementation, _ = dispatch.select_implementation(
      'triangle_attention',
      global_config.triangle_attention_implementation,
      policy,
      act.shape,
      act.dtype,
      pair_mask.shape,
      num_head=num_head,
  )
  if implementation == 'default' or pair_mask.dtype != act.dtype:
    return None
  if hk.running_init():
    # Create the parameters with the original initialisers, order and RNG.
    return None

  # pylint: disable=g-import-not-at-top
  from alphafold3.jax.fused_triangle import triattn_pallas
  # pylint: enable=g-import-not-at-top

  num_tokens, _, num_channels = act.shape
  head_dim = num_channels // num_head
  norm_scale, norm_offset = _layer_norm_params('act_norm', num_channels)
  qk_shape = (num_head, head_dim, num_channels)
  haiku_params = {
      'act_norm': {'scale': norm_scale, 'offset': norm_offset},
      'pair_bias_projection': {
          'weights': _linear_weights(
              'pair_bias_projection', (num_channels, num_head), act.dtype
          )
      },
      'q_projection': {
          'weights': _linear_weights('q_projection', qk_shape, act.dtype)
      },
      'k_projection': {
          'weights': _linear_weights('k_projection', qk_shape, act.dtype)
      },
      'v_projection': {
          'weights': _linear_weights(
              'v_projection', (num_channels, num_head, head_dim), act.dtype
          )
      },
      'gating_query': {
          'weights': _linear_weights(
              'gating_query', (num_head * head_dim, num_channels), act.dtype
          )
      },
      'output_projection': {
          'weights': _linear_weights(
              'output_projection',
              (num_head * head_dim, num_channels),
              act.dtype,
          )
      },
  }
  kernel_params = triattn_pallas.attn_params_from_haiku(haiku_params)
  tiles = dispatch.triangle_attention_tiles(num_tokens, policy)
  if implementation == 'pallas':
    return triattn_pallas.grid_self_attention_fused(
        act,
        pair_mask,
        kernel_params,
        transpose=transpose,
        ending_bias_transposed=False,
        cfg=tiles,
    )

  # 'pallas_tokamax_core': the Pallas prologue and epilogue around the
  # attention core of flash_attention_implementation.
  q, k, v, bias = triattn_pallas.attn_prologue(
      act,
      kernel_params['ln_scale'],
      kernel_params['ln_offset'],
      kernel_params['wq_t'],
      kernel_params['wk_t'],
      kernel_params['wv2'],
      kernel_params['wb16'],
      transpose=transpose,
      t=tiles['t1'],
      num_warps=tiles['w1'],
  )
  bias = jnp.transpose(bias[:, :, :num_head], (2, 0, 1))
  key_mask = jnp.swapaxes(pair_mask, -1, -2) > 0
  qkv_shape = (num_tokens, num_tokens, num_head, head_dim)
  weighted_avg = tokamax.dot_product_attention(
      q.reshape(qkv_shape),
      k.reshape(qkv_shape),
      v.reshape(qkv_shape),
      bias=bias[None],
      mask=key_mask[:, None, None, :],
      implementation=global_config.flash_attention_implementation,
  )
  weighted_avg = weighted_avg.reshape(
      num_tokens, num_tokens, num_head * head_dim
  )
  return triattn_pallas.attn_epilogue(
      weighted_avg,
      act,
      kernel_params['ln_scale'],
      kernel_params['ln_offset'],
      kernel_params['wg_t'],
      kernel_params['wo'],
      transpose=transpose,
      t=tiles['t2'],
      num_warps=tiles['w2'],
  )
