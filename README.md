# qecdecoder

A graph neural network decoder for quantum error correction, benchmarked
against the classical Minimum Weight Perfect Matching (MWPM) baseline.

This is an ongoing research project. The goal is to train a GNN decoder for
the surface code, rigorously compare it against MWPM (via
[PyMatching](https://github.com/oscarhiggott/PyMatching)) on logical error
rate and decode latency, and publish both the code and a benchmark
report/preprint.

## Status

**Phase 1 complete**: simulation + baseline pipeline validated. The
end-to-end `simulate -> decode -> benchmark` pipeline is tested against a
closed-form theoretical logical error rate for the repetition code (where
MWPM decoding is provably optimal and reduces to majority vote). This gives
confidence the pipeline is correct before moving to the surface code, where
no simple closed form exists.

**Phase 2 complete**: MWPM baseline on the rotated surface code (d=3, d=5)
under code-capacity-style depolarizing noise. Logical error rate vs.
physical error rate, with Wilson-interval error bars:

![MWPM baseline: logical error rate vs physical error rate for d=3 and d=5](experiments/results/phase2_mwpm_sweep.png)

The d=3/d=5 curves cross at p ~ 0.14 -- below that, the larger code
outperforms the smaller one (as expected below threshold); above it, the
ordering flips. This crossing-point estimate is a quick visual-threshold
check, not a rigorous finite-size-scaling fit (more distances/shots needed
for that -- future work).

Reproduce: `uv run qecdecoder sweep experiments/configs/phase2_mwpm_sweep.yaml --name phase2_mwpm_sweep`

**Next**: Phase 3, training a GNN decoder on the decoding graph derived
from Stim's `DetectorErrorModel` and benchmarking it against this MWPM
baseline.

See `.claude/plans/` (or ask) for the full roadmap.

## Stack

- [Stim](https://github.com/quantumlib/Stim) — circuit simulation & sampling
- [PyMatching](https://github.com/oscarhiggott/PyMatching) — MWPM baseline
- PyTorch + PyTorch Geometric — GNN decoder (CPU-first, GPU-ready)
- `uv` for dependency management

## Setup

```bash
uv sync
uv run pytest
```

## Layout

```
src/qecdecoder/
  codes.py       # Stim circuit builders (repetition code, rotated surface code)
  noise.py        # noise-model helpers (code-capacity depolarizing noise)
  simulate.py      # syndrome + logical-observable sampling
  baseline.py       # PyMatching MWPM wrapper
  benchmark.py       # empirical/theoretical logical error rate, CIs, threshold estimate
  sweep.py            # sweep MWPM over (distance, physical_error_rate)
  cli.py                # `qecdecoder sweep <config.yaml>` -> CSV + plot
tests/             # one test module per src module, plus integration tests
experiments/configs/  # per-experiment YAML configs (reproducibility)
experiments/results/  # experiment outputs (gitignored except README figures)
paper/               # preprint/benchmark-report draft
```
