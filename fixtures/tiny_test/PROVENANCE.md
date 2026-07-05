# tiny_test reference fixtures — provenance

Ground-truth mixing-proportion estimates for the SourceTracker collapsed-Gibbs
source-attribution algorithm, generated once from the inputs in this directory
and committed so the test suite is reproducible without any external runtime.
The expected matrices are the numeric output of independent reference
implementations of the algorithm, recorded solely for statistical-equivalence
testing of st3.

## Inputs (BSD-3; this directory)
- `table.tsv` — 20 features × 10 samples, integer counts.
- `metadata.tsv` — 10 samples: 5 sources (environments: drainwater, seawater,
  sewage) and 5 sinks.

sha256:
```
caea4cbcab68cba892e6deea3fdbad11f99842193287957f3ab6fc3b5162b10a  table.tsv
a840ccb776878819dc062af4c9a708e01ff09838a9b33c2b53ec89fc6f161cfc  metadata.tsv
```

## Expected outputs
| file | role | collapse | shape | columns |
|------|------|----------|-------|---------|
| `expected_sink_sum.tsv`        | primary   | sum  | 5 sinks × 4          | drainwater, seawater, sewage, Unknown |
| `expected_sink_sum_sd.tsv`     | primary   | sum  | 5 sinks × 4          | (std deviations of the above) |
| `expected_loo_sum.tsv`         | primary   | sum  | 5 source samples × 4 | drainwater, seawater, sewage, Unknown |
| `expected_sink_mean.tsv`       | secondary | mean | 5 sinks × 4          | drainwater, seawater, sewage, Unknown |
| `expected_contingency_sum.tsv` | primary   | sum  | COO (nonzeros)       | sink_id, source, feature, mean_count |

Columns of the proportion matrices are the sorted source environments followed
by `Unknown`; proportion rows are renormalized to sum to 1.

`expected_contingency_sum.tsv` is the per-sink source × taxon assignment table in
long COO form: `mean_count` is the mean number of the sink's sequences of that
feature attributed to that source, averaged over draws, so each sink's cells sum
to its sequencing depth. Only nonzero cells are listed; `source` is a source
environment or `Unknown`.

In leave-one-out, the single drainwater
source sample (`s7`) leaves an empty drainwater class when held out, so that
row's drainwater column is 0 and its mass is distributed over the remaining
classes.

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
Rarefaction is disabled: the tiny counts make a real subsampling depth
meaningless, and turning it off removes a source of randomness so the reference
means are stable. 1000 draws let the posterior means converge toward an
implementation-independent value.

## Collapse modes
- The **primary** matrices use **sum** collapse (source samples within an
  environment are summed).
- The **secondary** matrix uses **mean** collapse (per-environment mean, floored
  to integer counts before sampling).

The two modes are distinct configurations, not expected to be identical; on
these inputs they attribute sinks in the same ballpark (max difference ≈ 0.019
per proportion). Equivalence testing compares st3 against each matrix in its own
collapse mode, not the two modes against each other.

## Reproducibility
RNG seed = 42.

## Tolerance
Intended absolute tolerance for later equivalence comparison of st3 output
against these matrices: **0.02** per mixing proportion.
