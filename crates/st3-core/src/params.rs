// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Gibbs sampler configuration.
//!
//! Plain configuration data with eager validation. There is no ABI or
//! versioning here — that belongs to the C boundary (a later milestone). The
//! defaults follow the Python `sourcetracker2` library defaults, including a
//! `Mean` collapse; experiments (and the committed fixtures) override the
//! sampling counts explicitly.

use crate::collapse::CollapseMethod;
use crate::error::{Error, Result};

/// Configuration for the collapsed-Gibbs source-attribution sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct GibbsParams {
    /// Dirichlet prior counts for each feature in the source environments.
    pub alpha1: f64,
    /// Dirichlet prior counts for each feature in the Unknown environment.
    pub alpha2: f64,
    /// Prior counts of sink sequences in each environment.
    pub beta: f64,
    /// Number of independent Markov chains grown per sink.
    pub restarts: u32,
    /// Number of retained draws per restart.
    pub draws_per_restart: u32,
    /// Passes made before the first draw is taken.
    pub burnin: u32,
    /// Passes between retained draws (thinning gap).
    pub delay: u32,
    /// How source samples are aggregated per environment.
    pub collapse: CollapseMethod,
    /// If set, emit the per-sink feature-assignment (source × taxon) table.
    pub contingency: bool,
}

impl Default for GibbsParams {
    fn default() -> Self {
        Self {
            alpha1: 0.001,
            alpha2: 0.1,
            beta: 10.0,
            restarts: 10,
            draws_per_restart: 1,
            burnin: 100,
            delay: 1,
            collapse: CollapseMethod::Mean,
            contingency: false,
        }
    }
}

impl GibbsParams {
    /// Validate the parameters.
    ///
    /// # Errors
    /// [`Error::InvalidParam`] if any of `alpha1`, `alpha2`, `beta` is negative
    /// or non-finite, or if any of `restarts`, `draws_per_restart`, `burnin`,
    /// `delay` is zero.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("alpha1", self.alpha1),
            ("alpha2", self.alpha2),
            ("beta", self.beta),
        ] {
            if !value.is_finite() {
                return Err(Error::InvalidParam {
                    name,
                    reason: "must be finite",
                });
            }
            if value < 0.0 {
                return Err(Error::InvalidParam {
                    name,
                    reason: "must be non-negative",
                });
            }
        }
        for (name, value) in [
            ("restarts", self.restarts),
            ("draws_per_restart", self.draws_per_restart),
            ("burnin", self.burnin),
            ("delay", self.delay),
        ] {
            if value == 0 {
                return Err(Error::InvalidParam {
                    name,
                    reason: "must be at least 1",
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_default_validates() {
        assert!(GibbsParams::default().validate().is_ok());
        assert_eq!(GibbsParams::default().collapse, CollapseMethod::Mean);
    }

    #[test]
    fn params_reject_negative_alpha() {
        let p = GibbsParams {
            alpha1: -1.0,
            ..GibbsParams::default()
        };
        assert!(matches!(
            p.validate(),
            Err(Error::InvalidParam { name: "alpha1", .. })
        ));
    }

    #[test]
    fn params_reject_nonfinite_beta() {
        let p = GibbsParams {
            beta: f64::NAN,
            ..GibbsParams::default()
        };
        assert!(matches!(
            p.validate(),
            Err(Error::InvalidParam { name: "beta", .. })
        ));
    }

    #[test]
    fn params_reject_zero_counters() {
        for mutate in [
            |p: &mut GibbsParams| p.restarts = 0,
            |p: &mut GibbsParams| p.draws_per_restart = 0,
            |p: &mut GibbsParams| p.burnin = 0,
            |p: &mut GibbsParams| p.delay = 0,
        ] {
            let mut p = GibbsParams::default();
            mutate(&mut p);
            assert!(matches!(p.validate(), Err(Error::InvalidParam { .. })));
        }
    }
}
