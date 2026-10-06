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
