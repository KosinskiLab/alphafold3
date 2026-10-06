# Vendored: Anthropic's FlashPairformer triangle kernels

- Repository: https://github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0)
- Commit: f4f62fa6592ae4938d49b1757bea0cfeff9f468e (initial public release, 2026-09-17)
- Files, copied unchanged in the commit that adds this note:
  - `common/opt_core/opt_core/kernels/fpf_pallas/trimul_pallas.py`
  - `common/opt_core/opt_core/kernels/fpf_pallas/triattn_pallas.py`
  - `common/opt_core/opt_core/kernels/fpf_pallas/NOTICE`
  - `common/opt_core/opt_core/kernels/fpf_pallas_serve.py`

Later commits on this branch adapt them for this fork; `git log -p -- src/alphafold3/model/network/fused_triangle/` shows every
change against the original files.

The integration keeps both kernel files unchanged. `fpf_pallas_serve.py` is now
a standalone subset of the bf16 tile tables and device/shape dispatch, without
`opt_core` imports or runtime class replacement. `__init__.py` adapts the existing
Haiku parameters; module hooks are guarded by off-by-default `GlobalConfig`
fields. Initialisation always uses the original module body.

Set `fused_triangle_multiplication=True`, `fused_triangle_attention='auto'`,
`fused_triangle_compute_capability` and `fused_triangle_memory_gib` (the JAX
allocator budget) to opt in. AlphaPulldown fills these after a device smoke test.
Unsupported layers return to the original body. The measured architectures are
8.0, 8.6, 8.9, 9.0 and 12.0; 12.0 uses the measured 9.0 tile table. Attention has
a lower size limit than multiplication because it removes row chunking.

`tests/test_fused_triangle.py` checks parameter compatibility, nonzero-output
parity, padding, determinism, lazy imports and fallbacks on CPU with Pallas
interpret mode. GPU regression and full-model validation live in AlphaPulldown's
`exp/kernel-bench-phase0` harness branch, under `af3_integration/`.
