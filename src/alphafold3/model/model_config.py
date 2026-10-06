# Copyright 2024 DeepMind Technologies Limited
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

"""Global config for the model."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from alphafold3.common import base_config
import tokamax

_Shape2DType: TypeAlias = tuple[int | None, int | None]


class GlobalConfig(base_config.BaseConfig):
  """Global configuration for the AlphaFold3 model."""

  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation: tokamax.DotProductAttentionImplementation = (
      'triton'
  )
  # Inference-only fused Pallas kernels for the triangle modules. 'default'
  # runs the original module body. 'pallas_tokamax_core' wraps the attention
  # core of flash_attention_implementation in the Pallas prologue and epilogue;
  # 'auto' picks a triangle attention implementation for the device. Layers
  # that the kernels do not support run the default implementation.
  triangle_multiplication_implementation: Literal['default', 'pallas'] = (
      'default'
  )
  triangle_attention_implementation: Literal[
      'default', 'pallas', 'pallas_tokamax_core', 'auto'
  ] = 'default'
  # Device that the fused triangle kernels dispatch for: CUDA compute
  # capability (e.g. '9.0') and JAX allocator budget in GiB. Unknown or
  # unmeasured devices run the default implementation.
  fused_triangle_compute_capability: str = ''
  fused_triangle_memory_gib: float = 0.0
