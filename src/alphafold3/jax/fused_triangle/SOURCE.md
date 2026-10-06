# Vendored: Anthropic's FlashPairformer triangle kernels

- Repository: https://github.com/anthropics/uplifting-biomolecular-modeling
- Commit: f4f62fa6592ae4938d49b1757bea0cfeff9f468e (initial public release,
  2026-09-17)
- Copyright 2026 Anthropic, PBC. Licensed under the Apache License, Version 2.0
  (the repository's `LICENSE` and `NOTICE`).

## Vendored unchanged

These files are byte-identical to the kit at that commit, from its
`common/opt_core/opt_core/kernels/fpf_pallas/` directory:

- `trimul_pallas.py`
- `triattn_pallas.py`
- `NOTICE`

`NOTICE` names paths of the kit repository, not of this one:
`transition_pallas.py`, `opt_core/kernels/fpf_pallas`, `fpf_pallas_f32` and the
`common/opt_core/...` licence and notice files are not carried here.

## Copied in part

`dispatch.py` is this fork's dispatch policy. Only its bf16 tile rows are copied
from the kit: the `"8.0"` and `"9.0"` rows of `TILE_TABLES` and `SAFE_TABLE` in
`common/opt_core/opt_core/kernels/fpf_pallas_serve.py`. The rest of that file is
not carried; commit 9aedcb1 of this branch holds it as first vendored.

## Integration

`alphafold3.model.network.fused_triangle` adapts the existing Haiku parameters
of `TriangleMultiplication` and `GridSelfAttention` to the kernels. The module
hooks are guarded by `GlobalConfig` fields that default to the original module
body, and initialisation always runs the original body.

To opt in, set `triangle_multiplication_implementation='pallas'`,
`triangle_attention_implementation='auto'`, `fused_triangle_compute_capability`
and `fused_triangle_memory_gib` (the JAX allocator budget). Unsupported layers
run the original body. The measured architectures are 8.0, 8.6, 8.9, 9.0 and
12.0; 12.0 uses the measured 9.0 tile table. Attention has a lower size limit
than multiplication because it removes row chunking.

`src/alphafold3/model/network/fused_triangle_test.py` checks dispatch,
parameter compatibility, nonzero-output parity, padding, determinism, lazy
imports and fallbacks on CPU with Pallas interpret mode.
