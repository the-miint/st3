# synthetic_small reference fixtures — provenance

Ground-truth mixing-proportion estimates for the SourceTracker collapsed-Gibbs
source-attribution algorithm on a small, fully deterministic dataset with known
mixing structure, generated once and committed so the test suite is reproducible
without any external runtime. The expected matrices are the numeric output of
independent reference implementations of the algorithm, recorded solely for
statistical-equivalence testing of st3.

## Design (why this dataset exists)
Three source environments with disjoint, distinctive feature profiles and two
samples each, plus three sinks with known composition:
- `sink_pureA` — pure environment A.
- `sink_AB` — an even mixture of A and B.
- `sink_ABC` — an even mixture of A, B, and C.

Because every environment has two members, no environment vanishes under
leave-one-out (contrast with `tiny_test`). This makes the expected attribution
obvious: sinks load onto their constituent environments, and every held-out
source sample re-identifies its own environment — a self-consistency check.

## Inputs (this directory)
- `table.tsv` — 6 features × 9 samples, integer counts.
- `metadata.tsv` — 9 samples: 6 sources (environments: envA, envB, envC) and 3 sinks.

sha256:
```
14e575f13620e727eacdde9d06382e40730d0b679cd0d12f5cad6e442c4b2d7f  table.tsv
7b8ce09abc38e77fbbb9230a3a60f12af2b6f265a061fb27bebbc033807c8249  metadata.tsv
```

## Expected outputs
| file | role | collapse | shape | columns |
|------|------|----------|-------|---------|
| `expected_sink_sum.tsv`    | primary   | sum  | 3 sinks × 4          | envA, envB, envC, Unknown |
| `expected_sink_sum_sd.tsv` | primary   | sum  | 3 sinks × 4          | (std deviations of the above) |
| `expected_loo_sum.tsv`     | primary   | sum  | 6 source samples × 4 | envA, envB, envC, Unknown |
| `expected_sink_mean.tsv`   | secondary | mean | 3 sinks × 4          | envA, envB, envC, Unknown |

Columns are the sorted source environments followed by `Unknown`. Proportion
rows are renormalized to sum to 1.

## Parameters (pinned; identical for every matrix)
```
alpha1            = 0.001
alpha2            = 0.1
beta              = 10
restarts          = 100
draws_per_restart = 10        # 1000 total draws
burnin            = 100
delay             = 1
rarefaction       = OFF
```

## Collapse modes
- The **primary** matrices use **sum** collapse (source samples within an
  environment are summed).
- The **secondary** matrix uses **mean** collapse (per-environment mean, floored
  to integer counts before sampling).

The two modes are distinct configurations, not expected to be identical; on
these inputs they attribute sinks in the same ballpark (max difference ≈ 0.006
per proportion). Equivalence testing compares st3 against each matrix in its own
collapse mode, not the two modes against each other.

## Reproducibility
RNG seed = 42.

## Tolerance
Intended absolute tolerance for later equivalence comparison of st3 output
against these matrices: **0.02** per mixing proportion.
