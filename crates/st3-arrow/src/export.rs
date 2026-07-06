// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Export of a core [`SourceMixing`] result into Arrow batches.
//!
//! Two output kinds, matching design §6:
//! - **dense** mixing means and standard deviations, each one [`RecordBatch`]:
//!   a `sink_id` (Utf8) column plus one `Float64` column per environment (named
//!   by [`SourceMixing::env_names`], `Unknown` last), one row per sink;
//! - **sparse** per-sink feature assignments, streamed as a
//!   [`ContingencyReader`] — one flat-COO [`RecordBatch`] per sink over the
//!   uniform schema `[sink, source, feature, value]` (decision D13), so a
//!   consumer's peak memory is bounded by one sink's nonzeros.
//!
//! The exporters are result-type-agnostic: any [`SourceMixing`] — from
//! [`st3_core::predict_sinks`] or `predict_loo` — flows through unchanged. The
//! reader owns its data (`'static`), so a later milestone can hand it to an
//! `FFI_ArrowArrayStream` without borrowing the result.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, RecordBatch, RecordBatchReader, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::error::ArrowError;
use st3_core::{CooTally, SourceMixing};

use crate::error::Result;

/// Column name of the sink identifier in the dense means/stds batches.
const SINK_ID: &str = "sink_id";

/// Flat-COO contingency schema column names (decision D13).
const C_SINK: &str = "sink";
const C_SOURCE: &str = "source";
const C_FEATURE: &str = "feature";
const C_VALUE: &str = "value";

/// The dense mixing-**mean** batch: `sink_id` + one `Float64` column per env.
///
/// # Examples
/// ```
/// use st3_core::{
///     CollapseMethod, CountTable, GibbsParams, Role, SampleContext, predict_sinks,
/// };
/// use st3_arrow::means_batch;
///
/// let table = CountTable::from_coo(
///     vec!["f0".into(), "f1".into()],
///     vec!["a".into(), "b".into(), "sink".into()],
///     &[0, 1, 0, 1],
///     &[0, 1, 2, 2],
///     &[100.0, 100.0, 90.0, 10.0],
/// )?;
/// let ctx = SampleContext::new(
///     vec![Role::Source, Role::Source, Role::Sink],
///     vec![Some("envA".into()), Some("envB".into()), None],
/// )?;
/// let params = GibbsParams {
///     restarts: 4,
///     draws_per_restart: 2,
///     burnin: 5,
///     collapse: CollapseMethod::Sum,
///     ..GibbsParams::default()
/// };
/// let mixing = predict_sinks(&table, &ctx, &params, 42, 1)?;
///
/// // A dense batch: a `sink_id` column plus one Float64 column per environment.
/// let batch = means_batch(&mixing)?;
/// assert_eq!(batch.num_rows(), 1); // one sink
/// assert_eq!(batch.num_columns(), 1 + 3); // sink_id + envA + envB + Unknown
/// assert_eq!(batch.schema().field(0).name(), "sink_id");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// [`Error::Arrow`](crate::Error::Arrow) if the batch fails Arrow's own schema/
/// length validation (not expected for well-formed results).
pub fn means_batch(mixing: &SourceMixing) -> Result<RecordBatch> {
    dense_batch(mixing, mixing.means())
}

/// The dense mixing-**standard-deviation** batch, same shape as [`means_batch`].
///
/// # Errors
/// As [`means_batch`].
pub fn stds_batch(mixing: &SourceMixing) -> Result<RecordBatch> {
    dense_batch(mixing, mixing.stds())
}

/// Build a dense sink × env batch from a sink-major flat matrix (`means`/`stds`).
fn dense_batch(mixing: &SourceMixing, flat: &[f64]) -> Result<RecordBatch> {
    let n = mixing.n_sinks();
    let v = mixing.n_envs();
    let env_names = mixing.env_names();

    let mut fields = Vec::with_capacity(v + 1);
    fields.push(Field::new(SINK_ID, DataType::Utf8, false));
    for name in env_names {
        fields.push(Field::new(name, DataType::Float64, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(v + 1);
    columns.push(Arc::new(StringArray::from_iter_values(
        mixing.sink_ids().iter().map(|s| s.as_str()),
    )));
    for k in 0..v {
        let col: Vec<f64> = (0..n).map(|i| flat[i * v + k]).collect();
        columns.push(Arc::new(Float64Array::from(col)));
    }

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// The shared flat-COO contingency schema `[sink, source, feature, value]`.
fn contingency_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(C_SINK, DataType::Int32, false),
        Field::new(C_SOURCE, DataType::Int32, false),
        Field::new(C_FEATURE, DataType::Int32, false),
        Field::new(C_VALUE, DataType::Float64, false),
    ]))
}

/// A streaming reader over a result's per-sink contingency tables.
///
/// Yields one [`RecordBatch`] per sink over the uniform schema
/// `[sink, source, feature, value]` (all `Int32` indices plus a `Float64`
/// value). `sink` is the sink's position in [`SourceMixing::sink_ids`],
/// `source` an index into the environment order (`Unknown` last), `feature` a
/// taxon index; `value` is the mean count attributed over draws. Because it owns
/// its data, the reader is `'static`.
pub struct ContingencyReader {
    tallies: Vec<CooTally>,
    sink_ids: Vec<String>,
    schema: SchemaRef,
    next: usize,
}

impl ContingencyReader {
    fn new(tallies: Vec<CooTally>, sink_ids: Vec<String>) -> Self {
        debug_assert_eq!(
            tallies.len(),
            sink_ids.len(),
            "one tally per sink id is required"
        );
        Self {
            tallies,
            sink_ids,
            schema: contingency_schema(),
            next: 0,
        }
    }

    /// The sink identifiers, in the same order as the emitted `sink` indices.
    ///
    /// The batches carry integer `sink` indices (for a compact uniform schema);
    /// this maps index `i` back to its id.
    pub fn sink_ids(&self) -> &[String] {
        &self.sink_ids
    }
}

impl Iterator for ContingencyReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.tallies.len() {
            return None;
        }
        let i = self.next;
        self.next += 1;

        let tally = &self.tallies[i];
        let mut sink = Vec::new();
        let mut source = Vec::new();
        let mut feature = Vec::new();
        let mut value = Vec::new();
        for (src, feat, val) in tally.triples() {
            sink.push(i as i32);
            source.push(src as i32);
            feature.push(feat as i32);
            value.push(val);
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(sink)),
            Arc::new(Int32Array::from(source)),
            Arc::new(Int32Array::from(feature)),
            Arc::new(Float64Array::from(value)),
        ];
        Some(RecordBatch::try_new(self.schema.clone(), columns))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tallies.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl RecordBatchReader for ContingencyReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// A streaming reader over the per-sink contingency, or `None` when the result
/// carries no contingency (the common `contingency = false` path).
pub fn contingency_reader(mixing: &SourceMixing) -> Option<ContingencyReader> {
    let tallies = mixing.contingency()?;
    Some(ContingencyReader::new(
        tallies.to_vec(),
        mixing.sink_ids().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use st3_core::{
        CollapseMethod, CountTable, GibbsParams, Role, SampleContext, SourceMixing, predict_sinks,
    };

    /// Light, deterministic sampler params (fast; behaviour is seed-fixed).
    fn light_params(contingency: bool) -> GibbsParams {
        GibbsParams {
            restarts: 4,
            draws_per_restart: 2,
            burnin: 5,
            delay: 1,
            collapse: CollapseMethod::Sum,
            contingency,
            ..GibbsParams::default()
        }
    }

    /// A tiny two-env, two-sink table + context for driving `predict_sinks`.
    fn small_result(contingency: bool) -> SourceMixing {
        // 3 features; sources s0(envA), s1(envB); sinks s2, s3.
        let table = CountTable::from_coo(
            vec!["f0".into(), "f1".into(), "f2".into()],
            vec!["s0".into(), "s1".into(), "s2".into(), "s3".into()],
            &[0, 1, 1, 2, 0, 2],
            &[0, 0, 1, 1, 2, 3],
            &[8.0, 2.0, 9.0, 1.0, 5.0, 6.0],
        )
        .unwrap();
        let ctx = SampleContext::new(
            vec![Role::Source, Role::Source, Role::Sink, Role::Sink],
            vec![Some("envA".into()), Some("envB".into()), None, None],
        )
        .unwrap();
        predict_sinks(&table, &ctx, &light_params(contingency), 42, 1).unwrap()
    }

    #[test]
    fn means_batch_shape_names_and_cells() {
        let mixing = small_result(false);
        let batch = means_batch(&mixing).unwrap();

        assert_eq!(batch.num_rows(), mixing.n_sinks());
        assert_eq!(batch.num_columns(), 1 + mixing.n_envs());

        // Column names: sink_id then the env names (Unknown last).
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        let mut expected = vec![SINK_ID];
        expected.extend(mixing.env_names().iter().map(|s| s.as_str()));
        assert_eq!(names, expected);

        // sink_id column equals the result's sink ids.
        let ids = batch
            .column_by_name(SINK_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for (i, id) in mixing.sink_ids().iter().enumerate() {
            assert_eq!(ids.value(i), id);
        }

        // Each env cell equals mean_row(i)[k].
        for (k, env) in mixing.env_names().iter().enumerate() {
            let col = batch
                .column_by_name(env)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..mixing.n_sinks() {
                assert_eq!(col.value(i), mixing.mean_row(i)[k]);
            }
        }
    }

    #[test]
    fn stds_batch_cells_match_std_rows() {
        let mixing = small_result(false);
        let batch = stds_batch(&mixing).unwrap();
        assert_eq!(batch.num_rows(), mixing.n_sinks());
        for (k, env) in mixing.env_names().iter().enumerate() {
            let col = batch
                .column_by_name(env)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..mixing.n_sinks() {
                assert_eq!(col.value(i), mixing.std_row(i)[k]);
            }
        }
    }

    #[test]
    fn contingency_reader_none_when_absent() {
        let mixing = small_result(false);
        assert!(mixing.contingency().is_none());
        assert!(contingency_reader(&mixing).is_none());
    }

    #[test]
    fn contingency_reader_reproduces_triples_per_sink() {
        let mixing = small_result(true);
        let reader = contingency_reader(&mixing).expect("contingency requested");

        // Reader schema is the flat 4-column form.
        let reader_schema = reader.schema();
        let names: Vec<&str> = reader_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec![C_SINK, C_SOURCE, C_FEATURE, C_VALUE]);
        assert_eq!(reader.sink_ids(), mixing.sink_ids());

        let tallies = mixing.contingency().unwrap().to_vec();
        let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), mixing.n_sinks());

        let mut reader_total = 0.0f64;
        let mut triples_total = 0.0f64;
        for (i, batch) in batches.iter().enumerate() {
            let sink = col_i32(batch, C_SINK);
            let source = col_i32(batch, C_SOURCE);
            let feature = col_i32(batch, C_FEATURE);
            let value = col_f64(batch, C_VALUE);

            let expected: Vec<(u32, u32, f64)> = tallies[i].triples().collect();
            assert_eq!(batch.num_rows(), expected.len());
            for (row, (src, feat, val)) in expected.iter().enumerate() {
                assert_eq!(sink.value(row), i as i32);
                assert_eq!(source.value(row), *src as i32);
                assert_eq!(feature.value(row), *feat as i32);
                assert_eq!(value.value(row), *val);
                reader_total += value.value(row);
                triples_total += *val;
            }
        }
        assert!((reader_total - triples_total).abs() < 1e-12);
        assert!(reader_total > 0.0, "a non-empty result must tally mass");
    }

    fn col_i32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int32Array {
        batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
    }

    fn col_f64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float64Array {
        batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
    }
}
