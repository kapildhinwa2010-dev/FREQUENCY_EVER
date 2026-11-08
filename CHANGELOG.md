# Changelog

## [0.1.0] - 2026-11-08

### Added
- 9-step wave-native architecture: freq bitnet (ternary weights as frequencies),
  tokenizer-free converter, MambaByte frontend, REAL NLSE soliton transfer (SSFM),
  Mamba-2 FFT engine, frequency backprop, training loop, Loop Thinking (ACT), unified model.
- Bounded smoke tests (tests/test_smoke.py) - converter roundtrip, soliton zero-loss
  physics, 0D-3D grad safety, soliton shape proof, unified forward+generate.
- Self-diagnosing CI (run-all-steps.yml): pytest gate + 9-step demos, logs repo me commit.
- Packaging (pyproject.toml), MIT License, CITATION.cff, CodeQL, Dependabot, templates.

### Fixed
- SolitonChannel.encode: broadcast-multiply + invalid reshape (T>1 pe har input crash) ->
  complex matmul (CI-verified).
- Complex dtype mismatch (A @ template) - CI smoke test ne pakda.
- Decode global-phase correction ab actual propagation distance use karta hai.
- Scalar/1D parameter gradients (soliton_grad_pass) 0D-3D safe.
- step9 demo ab root README.md clobber nahi karta (artifacts/ me likhta hai).
- Workflow install step requirements ki comment lines filter karta hai.
