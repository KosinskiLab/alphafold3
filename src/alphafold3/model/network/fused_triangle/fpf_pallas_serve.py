"""Inference dispatch for the vendored bf16 triangle kernels.

The original serve layer is preserved in commit 9aedcb1. This subset has no
opt_core dependency, environment overrides, fp32 routes or import-time GPU work.
Size limits are conservative: per-layer parity does not establish model capacity.
"""
from dataclasses import dataclass

SOURCE_COMMIT = 'f4f62fa6592ae4938d49b1757bea0cfeff9f468e'
POLICY_VERSION = 1
# bf16 rows copied from the upstream serve table; sm_120 used sm_90 in Gate 0.
TILE_TABLES = {
    '8.0': dict(
        trimul=dict(t1=64, w1=4, s1=2, t2=64, w2=4, s2=2, ein='xla'),
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=32, wa=4, sa=3),
        attn_by_n={768: dict(bq=128), 1024: dict(t2=128, w2=8, bk=64),
                   1536: dict(bq=128), 2048: dict(t1=128, bk=64)}),
    '9.0': dict(
        trimul=dict(t1=64, w1=8, s1=2, t2=64, w2=8, s2=2, ein='xla'),
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=64, wa=4, sa=3),
        attn_by_n={1024: dict(bq=128, bk=32, wa=4, sa=3)}),
    'safe': dict(
        trimul=dict(t1=32, w1=4, s1=1, t2=32, w2=4, s2=1, ein='xla'),
        attn_default=dict(t1=32, w1=4, t2=32, w2=4, bq=32, bk=32, wa=4, sa=1),
        attn_by_n={}),
}


@dataclass(frozen=True)
class DevicePolicy:
    compute_capability: str
    memory_gib: float
    tiles: str = ''
    attention: str = 'off'
    trimul_max_n: int = 0
    attention_max_n: int = 0
    reason: str = ''

    @property
    def enabled(self):
        return not self.reason


def card_policy(compute_capability, memory_gib):
    """Choose only measured architectures; use the allocator's memory budget.

    8.0/8.6/8.9/9.0/12.0 were checked with JAX 0.9.1. Future cards fall
    back until measured. An unknown memory budget also falls back.
    """
    try:
        cc = float(compute_capability)
        if cc >= 20:
            cc /= 10
        cc = f'{cc:.1f}'
        memory = float(memory_gib)
    except (TypeError, ValueError):
        return DevicePolicy(str(compute_capability), 0, reason='unknown_device')
    if cc not in ('8.0', '8.6', '8.9', '9.0', '12.0'):
        return DevicePolicy(cc, memory, reason='unvalidated_compute_capability')
    if not 0 < memory < float('inf'):
        return DevicePolicy(cc, memory, reason='unknown_memory_budget')
    if memory < 12:
        return DevicePolicy(cc, memory, reason='memory_budget_below_12_gib')
    tiles = 'safe' if cc in ('8.6', '8.9') else '8.0' if cc == '8.0' else '9.0'
    attention = 'pallas_tokamax_core' if tiles == 'safe' else 'pallas'
    # Stay within the layer-tested range and leave attention chunked near
    # the full model's observed capacity. These limits can be raised by data.
    trimul_max = 5120 if cc in ('9.0', '12.0') else 3584
    if memory < 20:
        trimul_max, attention_max = min(trimul_max, 1536), 768
    elif memory < 32:
        trimul_max, attention_max = min(trimul_max, 2560), 1536
    elif memory < 64:
        attention_max = 2048
    else:
        attention_max = 3072
    return DevicePolicy(cc, memory, tiles, attention, trimul_max, attention_max)


def policy_from_config(config):
    return card_policy(config.fused_triangle_compute_capability,
                       config.fused_triangle_memory_gib)


def select(config, kind, shape, dtype, mask_shape, *, num_head=4):
    """Return (backend, reason) without importing JAX or reading a device.

    This same predicate drives execution and output metadata. No JIT trace
    counters are used, so cache hits retain the same provenance.
    """
    if kind not in ('trimul', 'attention'):
        raise ValueError(f'Unknown triangle operation: {kind}')
    requested = (config.triangle_multiplication_implementation if kind == 'trimul'
                 else config.triangle_attention_implementation)
    if requested == 'default':
        return 'stock', 'off'
    policy = policy_from_config(config)
    if not policy.enabled:
        return 'stock', policy.reason
    if str(dtype) != 'bfloat16':
        return 'stock', 'dtype_not_bfloat16'
    if len(shape) != 3 or shape[0] != shape[1] or shape[0] <= 0:
        return 'stock', 'non_square_pair'
    n, _, c = shape
    if tuple(mask_shape) != (n, n):
        return 'stock', 'mask_shape'
    # Limit v1 to the two channel counts validated in AF3.
    if c not in (64, 128) or n % 64:
        return 'stock', 'unvalidated_shape'
    if kind == 'trimul':
        if n > policy.trimul_max_n:
            return 'stock', 'size_limit'
        if requested != 'pallas':
            raise ValueError(
                f'Unknown triangle multiplication implementation: {requested}')
        return 'pallas', ''
    if num_head != 4:
        return 'stock', 'unvalidated_heads'
    if n > policy.attention_max_n:
        return 'stock', 'size_limit'
    backend = requested
    if backend == 'auto':
        backend = policy.attention
    if backend not in ('pallas', 'pallas_tokamax_core'):
        raise ValueError(f'Unknown triangle attention implementation: {backend}')
    return backend, ''


def trimul_cfg(n, policy):
    del n
    return dict(TILE_TABLES[policy.tiles]['trimul'])


def attn_cfg(n, policy):
    table = TILE_TABLES[policy.tiles]
    return {**table['attn_default'], **table['attn_by_n'].get(n, {})}
