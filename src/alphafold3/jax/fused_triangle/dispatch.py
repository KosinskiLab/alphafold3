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

"""Device and shape dispatch for the fused triangle kernels.

Decides, without importing JAX or querying a device, whether a triangle layer
runs a fused kernel and with which tiles. Only measured GPU architectures are
enabled, and the size limits are conservative: per-layer parity does not
establish the memory capacity of the full model.
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import numpy.typing as npt

Operation: TypeAlias = Literal['triangle_multiplication', 'triangle_attention']

# Increase when a dispatch decision changes, so that recorded provenance can
# tell policies apart.
POLICY_VERSION = 1

# Tile settings of the bf16 kernels, keyed by tile table name. The '8.0' and
# '9.0' rows are the bf16 trimul, attn_default and attn_by_n entries of
# TILE_TABLES, and the 'safe' rows are SAFE_TABLE, copied from
# common/opt_core/opt_core/kernels/fpf_pallas_serve.py of
# https://github.com/anthropics/uplifting-biomolecular-modeling at f4f62fa6
# (Copyright 2026 Anthropic, PBC; Apache License, Version 2.0).
_TILE_TABLES = {
    '8.0': dict(
        trimul=dict(t1=64, w1=4, s1=2, t2=64, w2=4, s2=2, ein='xla'),
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=32, wa=4, sa=3),
        attn_by_n={
            768: dict(bq=128),
            1024: dict(t2=128, w2=8, bk=64),
            1536: dict(bq=128),
            2048: dict(t1=128, bk=64),
        },
    ),
    '9.0': dict(
        trimul=dict(t1=64, w1=8, s1=2, t2=64, w2=8, s2=2, ein='xla'),
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=64, wa=4, sa=3),
        attn_by_n={1024: dict(bq=128, bk=32, wa=4, sa=3)},
    ),
    'safe': dict(
        trimul=dict(t1=32, w1=4, s1=1, t2=32, w2=4, s2=1, ein='xla'),
        attn_default=dict(t1=32, w1=4, t2=32, w2=4, bq=32, bk=32, wa=4, sa=1),
        attn_by_n={},
    ),
}

# Checked with JAX 0.9.1. 12.0 has no table of its own and uses the 9.0 rows.
_MEASURED_COMPUTE_CAPABILITIES = ('8.0', '8.6', '8.9', '9.0', '12.0')
_MIN_MEMORY_GIB = 12
# The two pair channel counts of AlphaFold 3; the kernels need N % 64 == 0.
_SUPPORTED_NUM_CHANNELS = (64, 128)
_NUM_TOKENS_MULTIPLE = 64
_SUPPORTED_NUM_HEAD = 4


@dataclasses.dataclass(frozen=True)
class DevicePolicy:
  """Fused triangle kernel limits for one device.

  Attributes:
    compute_capability: CUDA compute capability, normalised to e.g. '9.0'.
    memory_gib: JAX allocator budget in GiB.
    tile_table: Name of the tile table that the kernels use.
    attention_implementation: Triangle attention implementation that 'auto'
      selects.
    multiplication_max_tokens: Largest pair size N that runs fused triangle
      multiplication.
    attention_max_tokens: Largest pair size N that runs fused triangle
      attention.
    reason: Why the fused kernels are disabled on this device; empty when they
      are enabled.
  """

  compute_capability: str
  memory_gib: float
  tile_table: str = ''
  attention_implementation: str = 'default'
  multiplication_max_tokens: int = 0
  attention_max_tokens: int = 0
  reason: str = ''

  @property
  def enabled(self) -> bool:
    return not self.reason


def device_policy(
    compute_capability: str | float, memory_gib: str | float
) -> DevicePolicy:
  """Returns the fused triangle kernel policy for a device.

  Architectures that were not measured fall back until they are, and so does
  an unknown memory budget.

  Args:
    compute_capability: CUDA compute capability, e.g. '9.0' or '90'.
    memory_gib: JAX allocator budget in GiB.

  Returns:
    The policy for the device. Its reason is set when the kernels are disabled.
  """
  try:
    capability_value = float(compute_capability)
    if capability_value >= 20:
      capability_value /= 10
    capability = f'{capability_value:.1f}'
    memory = float(memory_gib)
  except (TypeError, ValueError):
    return DevicePolicy(str(compute_capability), 0, reason='unknown_device')
  if capability not in _MEASURED_COMPUTE_CAPABILITIES:
    return DevicePolicy(
        capability, memory, reason='unvalidated_compute_capability'
    )
  if not 0 < memory < float('inf'):
    return DevicePolicy(capability, memory, reason='unknown_memory_budget')
  if memory < _MIN_MEMORY_GIB:
    return DevicePolicy(capability, memory, reason='memory_budget_below_12_gib')

  if capability in ('8.6', '8.9'):
    tile_table = 'safe'
  elif capability == '8.0':
    tile_table = '8.0'
  else:
    tile_table = '9.0'
  if tile_table == 'safe':
    attention_implementation = 'pallas_tokamax_core'
  else:
    attention_implementation = 'pallas'

  # Stay within the layer-tested range, and keep attention chunked near the
  # observed capacity of the full model. Measurements can raise these limits.
  multiplication_max_tokens = 5120 if capability in ('9.0', '12.0') else 3584
  if memory < 20:
    multiplication_max_tokens = min(multiplication_max_tokens, 1536)
    attention_max_tokens = 768
  elif memory < 32:
    multiplication_max_tokens = min(multiplication_max_tokens, 2560)
    attention_max_tokens = 1536
  elif memory < 64:
    attention_max_tokens = 2048
  else:
    attention_max_tokens = 3072
  return DevicePolicy(
      compute_capability=capability,
      memory_gib=memory,
      tile_table=tile_table,
      attention_implementation=attention_implementation,
      multiplication_max_tokens=multiplication_max_tokens,
      attention_max_tokens=attention_max_tokens,
  )


def select_implementation(
    operation: Operation,
    requested: str,
    policy: DevicePolicy,
    shape: Sequence[int],
    dtype: npt.DTypeLike,
    mask_shape: Sequence[int],
    *,
    num_head: int = _SUPPORTED_NUM_HEAD,
) -> tuple[str, str]:
  """Selects the implementation that a triangle layer runs.

  The result depends only on the arguments, so the same call can record which
  implementation ran, also when a compiled function is reused.

  Args:
    operation: The triangle module.
    requested: The configured implementation of this operation.
    policy: The device policy, from device_policy.
    shape: Shape of the pair activations, [N, N, C].
    dtype: Data type of the pair activations.
    mask_shape: Shape of the pair mask.
    num_head: Number of attention heads; triangle attention only.

  Returns:
    A tuple (implementation, reason). The implementation is 'default' when the
    layer must run the original module body, and the reason says why;
    otherwise it is the fused implementation to run and the reason is empty.

  Raises:
    ValueError: If the operation or the requested implementation is unknown.
  """
  if operation not in ('triangle_multiplication', 'triangle_attention'):
    raise ValueError(f'Unknown triangle operation: {operation}')
  if requested == 'default':
    return 'default', 'not_requested'
  if not policy.enabled:
    return 'default', policy.reason
  if str(dtype) != 'bfloat16':
    return 'default', 'dtype_not_bfloat16'
  if len(shape) != 3 or shape[0] != shape[1] or shape[0] <= 0:
    return 'default', 'non_square_pair'
  num_tokens, _, num_channels = shape
  if tuple(mask_shape) != (num_tokens, num_tokens):
    return 'default', 'mask_shape'
  if (
      num_channels not in _SUPPORTED_NUM_CHANNELS
      or num_tokens % _NUM_TOKENS_MULTIPLE
  ):
    return 'default', 'unvalidated_shape'

  if operation == 'triangle_multiplication':
    if num_tokens > policy.multiplication_max_tokens:
      return 'default', 'size_limit'
    if requested != 'pallas':
      raise ValueError(
          f'Unknown triangle multiplication implementation: {requested}'
      )
    return 'pallas', ''

  if num_head != _SUPPORTED_NUM_HEAD:
    return 'default', 'unvalidated_heads'
  if num_tokens > policy.attention_max_tokens:
    return 'default', 'size_limit'
  if requested == 'auto':
    implementation = policy.attention_implementation
  else:
    implementation = requested
  if implementation not in ('pallas', 'pallas_tokamax_core'):
    raise ValueError(
        f'Unknown triangle attention implementation: {implementation}'
    )
  return implementation, ''


def triangle_multiplication_tiles(
    policy: DevicePolicy,
) -> dict[str, int | str]:
  """Returns the triangle multiplication kernel tiles for a device policy."""
  return dict(_TILE_TABLES[policy.tile_table]['trimul'])


def triangle_attention_tiles(
    num_tokens: int, policy: DevicePolicy
) -> dict[str, int]:
  """Returns the triangle attention kernel tiles for a pair size N."""
  table = _TILE_TABLES[policy.tile_table]
  return {**table['attn_default'], **table['attn_by_n'].get(num_tokens, {})}
