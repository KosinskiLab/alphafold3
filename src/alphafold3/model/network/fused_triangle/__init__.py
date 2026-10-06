"""Optional AF3 inference kernels. Heavy imports happen only after dispatch."""


def triangle_multiplication(act, mask, config, global_config):
    """Return the fused output, or None to keep the original module body."""
    from .fpf_pallas_serve import policy_from_config, select, trimul_cfg
    backend, _ = select(global_config, 'trimul', act.shape, act.dtype, mask.shape)
    if backend == 'stock' or not config.use_glu_kernel or mask.dtype != act.dtype:
        return None
    if config.equation not in ('ikc,jkc->ijc', 'kjc,kic->ijc'):
        return None
    import haiku as hk
    if hk.running_init():
        return None  # Use stock initialisers, parameter creation order and RNG.
    import jax.numpy as jnp
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.jax.fused_triangle import trimul_pallas

    c = act.shape[-1]
    def norm(name):
        with hk.name_scope(name):
            return (hk.get_parameter('scale', (c,), jnp.float32),
                    hk.get_parameter('offset', (c,), jnp.float32))
    s_in, o_in = norm('left_norm_input')
    s_c, o_c = norm('center_norm')
    wp, _ = hm.haiku_linear_get_params(act, num_output=2*c, name='projection')
    wg, _ = hm.haiku_linear_get_params(act, num_output=2*c, name='gate')
    wo, _ = hm.haiku_linear_get_params(act, num_output=c, name='output_projection')
    wgl, _ = hm.haiku_linear_get_params(act, num_output=c, name='gating_linear')
    params = dict(ln_in_scale=s_in, ln_in_offset=o_in, ln_c_scale=s_c,
                  ln_c_offset=o_c, w_proj=wp, w_gate=wg, w_out=wo, w_gl=wgl)
    return trimul_pallas.triangle_multiplication_fused(
        act, mask, params, equation=config.equation,
        cfg=trimul_cfg(act.shape[0], policy_from_config(global_config)))


def grid_self_attention(act, mask, config, global_config, *, transpose):
    """Return DeepMind-convention fused attention, or None for stock."""
    from .fpf_pallas_serve import attn_cfg, policy_from_config, select
    backend, _ = select(global_config, 'attention', act.shape, act.dtype,
                        mask.shape, num_head=config.num_head)
    if backend == 'stock' or mask.dtype != act.dtype:
        return None
    import haiku as hk
    if hk.running_init():
        return None
    import jax.numpy as jnp
    from alphafold3.jax.fused_triangle import triattn_pallas as kernel

    n, _, c = act.shape
    h, d = config.num_head, c // config.num_head
    def param(name, shape, key='weights', dtype=act.dtype):
        with hk.name_scope(name):
            return hk.get_parameter(key, shape, dtype)
    with hk.name_scope('act_norm'):
        norm_params = {
            'scale': hk.get_parameter('scale', (c,), jnp.float32),
            'offset': hk.get_parameter('offset', (c,), jnp.float32),
        }
    params = {
        'act_norm': norm_params,
        'pair_bias_projection': {'weights': param('pair_bias_projection', (c, h))},
        'q_projection': {'weights': param('q_projection', (h, d, c))},
        'k_projection': {'weights': param('k_projection', (h, d, c))},
        'v_projection': {'weights': param('v_projection', (c, h, d))},
        'gating_query': {'weights': param('gating_query', (h*d, c))},
        'output_projection': {'weights': param('output_projection', (h*d, c))},
    }
    kp = kernel.attn_params_from_haiku(params)
    tiles = attn_cfg(n, policy_from_config(global_config))
    if backend == 'pallas':
        return kernel.grid_self_attention_fused(
            act, mask, kp, transpose=transpose,
            ending_bias_transposed=False, cfg=tiles)
    import tokamax
    q, k, v, bias = kernel.attn_prologue(
        act, kp['ln_scale'], kp['ln_offset'], kp['wq_t'], kp['wk_t'],
        kp['wv2'], kp['wb16'], transpose=transpose,
        t=tiles['t1'], num_warps=tiles['w1'])
    bias = jnp.transpose(bias[:, :, :h], (2, 0, 1))
    key_mask = jnp.swapaxes(mask, -1, -2) > 0
    out = tokamax.dot_product_attention(
        q.reshape(n, n, h, d), k.reshape(n, n, h, d), v.reshape(n, n, h, d),
        bias=bias[None], mask=key_mask[:, None, None, :],
        implementation=global_config.flash_attention_implementation)
    return kernel.attn_epilogue(
        out.reshape(n, n, h*d), act, kp['ln_scale'], kp['ln_offset'],
        kp['wg_t'], kp['wo'], transpose=transpose,
        t=tiles['t2'], num_warps=tiles['w2'])
