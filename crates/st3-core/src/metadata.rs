// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Per-sample roles and environments, and the source/sink split.
//!
//! A [`SampleContext`] is index-aligned to a [`CountTable`]'s sample axis and
//! records each sample's role and environment. The split it exposes is a
//! *view* — lists of indices into the shared table, not a copy of counts — so
//! leave-one-out (a later milestone) can drop one source index and re-collapse
//! without re-materializing anything.

use crate::error::{Error, Result};
use crate::table::CountTable;

/// A sample's role in source attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A training sample contributing to a source environment.
    Source,
    /// A sample whose source composition is estimated.
    Sink,
}

/// Per-sample role and environment, index-aligned to a table's sample axis.
///
/// Sources must carry a non-empty environment; sinks may carry one, but it is
/// unused by collapse. The source and sink index lists are precomputed in
/// sample order (ascending).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleContext {
    roles: Vec<Role>,
    envs: Vec<Option<String>>,
    source_indices: Vec<u32>,
    sink_indices: Vec<u32>,
}

impl SampleContext {
    /// Build from index-aligned role and environment vectors.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `roles` and `envs` differ in length;
    /// [`Error::MissingEnv`] if a source has no (non-empty) environment;
    /// [`Error::NoSources`] if no sample is a source.
    pub fn new(roles: Vec<Role>, envs: Vec<Option<String>>) -> Result<Self> {
        if roles.len() != envs.len() {
            return Err(Error::LengthMismatch {
                what: "envs",
                expected: roles.len(),
                actual: envs.len(),
            });
        }

        let mut source_indices = Vec::new();
        let mut sink_indices = Vec::new();
        for (i, role) in roles.iter().enumerate() {
            match role {
                Role::Source => {
                    let has_env = matches!(envs[i].as_deref(), Some(s) if !s.is_empty());
                    if !has_env {
                        return Err(Error::MissingEnv { sample_index: i });
                    }
                    source_indices.push(i as u32);
                }
                Role::Sink => sink_indices.push(i as u32),
            }
        }
        if source_indices.is_empty() {
            return Err(Error::NoSources);
        }

        Ok(Self {
            roles,
            envs,
            source_indices,
            sink_indices,
        })
    }

    /// As [`SampleContext::new`], additionally requiring the lengths to match
    /// `table.n_samples()`.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `roles` is not the sample count, plus every
    /// error of [`SampleContext::new`].
    pub fn for_table(
        table: &CountTable,
        roles: Vec<Role>,
        envs: Vec<Option<String>>,
    ) -> Result<Self> {
        let n = table.n_samples();
        if roles.len() != n {
            return Err(Error::LengthMismatch {
                what: "roles",
                expected: n,
                actual: roles.len(),
            });
        }
        Self::new(roles, envs)
    }

    /// Sample indices of the sources, ascending.
    pub fn source_indices(&self) -> &[u32] {
        &self.source_indices
    }

    /// Sample indices of the sinks, ascending.
    pub fn sink_indices(&self) -> &[u32] {
        &self.sink_indices
    }

    /// Environment label of each source, parallel to [`Self::source_indices`].
    pub fn source_envs(&self) -> Vec<&str> {
        self.source_indices
            .iter()
            .map(|&i| self.envs[i as usize].as_deref().unwrap_or(""))
            .collect()
    }

    /// Role of sample `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of range.
    pub fn role(&self, i: usize) -> Role {
        self.roles[i]
    }

    /// Environment of sample `i`, if any.
    ///
    /// # Panics
    /// Panics if `i` is out of range.
    pub fn env(&self, i: usize) -> Option<&str> {
        self.envs[i].as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(s: &str) -> Option<String> {
        Some(s.to_owned())
    }

    #[test]
    fn context_splits_roles() {
        let ctx = SampleContext::new(
            vec![Role::Source, Role::Sink, Role::Source],
            vec![env("a"), None, env("b")],
        )
        .unwrap();
        assert_eq!(ctx.source_indices(), &[0, 2]);
        assert_eq!(ctx.sink_indices(), &[1]);
    }

    #[test]
    fn context_source_envs_parallel() {
        let ctx = SampleContext::new(
            vec![Role::Source, Role::Sink, Role::Source],
            vec![env("a"), None, env("b")],
        )
        .unwrap();
        assert_eq!(ctx.source_envs(), vec!["a", "b"]);
    }

    #[test]
    fn err_no_sources() {
        let e = SampleContext::new(vec![Role::Sink, Role::Sink], vec![None, None]).unwrap_err();
        assert_eq!(e, Error::NoSources);
    }

    #[test]
    fn err_source_missing_env() {
        let e = SampleContext::new(vec![Role::Source], vec![None]).unwrap_err();
        assert_eq!(e, Error::MissingEnv { sample_index: 0 });
        // An empty string is also "missing".
        let e = SampleContext::new(vec![Role::Source], vec![env("")]).unwrap_err();
        assert_eq!(e, Error::MissingEnv { sample_index: 0 });
    }

    #[test]
    fn err_roles_len_mismatch() {
        let t = CountTable::from_coo(
            vec!["f0".into()],
            vec!["s0".into(), "s1".into()],
            &[0],
            &[0],
            &[1.0],
        )
        .unwrap();
        let e = SampleContext::for_table(&t, vec![Role::Source], vec![env("a")]).unwrap_err();
        assert!(matches!(e, Error::LengthMismatch { .. }));
    }

    #[test]
    fn sink_env_optional_ok() {
        let ctx = SampleContext::new(vec![Role::Source, Role::Sink], vec![env("a"), None]).unwrap();
        assert_eq!(ctx.source_indices(), &[0]);
        assert_eq!(ctx.env(1), None);
    }
}
