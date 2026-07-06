// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The versioned, `#[repr(C)]` run configuration and its mapping onto core types.
//!
//! [`St3Config`] carries a Vulkan-style `struct_version` + `struct_size` prefix
//! (design §13) so a future estimator or tuning block can extend it without
//! breaking a compiled caller: the library reads only the fields the declared
//! version guarantees. [`St3Config::to_core`] validates the prefix and the run
//! selectors, then produces the core [`GibbsParams`] and [`RarefyConfig`] plus
//! the seed, job count, and mode.
//!
//! Every field has a type with no invalid bit patterns (`u32`, `u64`, `i32`,
//! `u8`, `f64`), so reading a fully-initialized `St3Config` from C is always
//! sound regardless of the field values. That drives two deliberate choices:
//! - The multi-valued selectors `collapse` and `estimator` are stored as `i32`,
//!   not as the [`St3Collapse`]/[`St3EstimatorKind`] enums. A `#[repr(C)]` enum
//!   has invalid discriminants, so forming a reference to one holding, say, `2`
//!   would be undefined behavior; an `i32` has none. [`to_core`](St3Config::to_core)
//!   then rejects an unrecognized selector with [`St3Status::ErrInvalidInput`] and
//!   a message — the realistic caller mistake (an out-of-range choice) becomes a
//!   clean error rather than UB. The enums are still exported so C callers have
//!   named constants (`ST3_COLLAPSE_*`, `ST3_EST_*`) to assign.
//! - The boolean flags are stored as `u8` (0 = false, nonzero = true) for the
//!   same reason: `bool` is only valid as `0`/`1`, so an initialized-but-other
//!   byte would be UB, whereas any `u8` is valid.

use std::mem::size_of;

use st3_core::{CollapseMethod, GibbsParams, RarefyConfig};

use crate::last_error::set_last_error;
use crate::status::St3Status;

/// The only configuration layout version understood by this ABI. Callers set
/// the `St3Config.struct_version` field to this value.
pub const ST3_CONFIG_V1: u32 = 1;

/// The version + size prefix every `St3Config` layout version shares (the first
/// two fields, at offsets 0 and 4). Reading only this — rather than the whole
/// struct — lets the library reject a differently-sized (other-version) caller
/// allocation before any out-of-bounds read.
#[repr(C)]
#[derive(Clone, Copy)]
struct St3ConfigPrefix {
    struct_version: u32,
    struct_size: u32,
}

/// Validate a config pointer's version/size prefix without dereferencing the
/// full struct.
///
/// This realizes the Vulkan-style prefix contract (design §13): a caller
/// compiled against a different layout version sets `struct_size` to *its* size,
/// which will not match this ABI's size, so it is rejected here — before
/// [`st3_run`](crate::st3_run) forms a full `&St3Config` (which would over-read a
/// smaller allocation). Only the 8-byte prefix is read, the minimum any versioned
/// config guarantees.
///
/// # Safety
/// `config` must be non-null, aligned for [`St3Config`], and point to at least a
/// readable [`St3ConfigPrefix`] (8 bytes).
pub(crate) unsafe fn validate_prefix(config: *const St3Config) -> Result<(), St3Status> {
    // SAFETY: the caller guarantees at least 8 readable, aligned bytes;
    // `St3ConfigPrefix` mirrors `St3Config`'s first two fields.
    let prefix = unsafe { std::ptr::read(config.cast::<St3ConfigPrefix>()) };
    if prefix.struct_version != ST3_CONFIG_V1 {
        set_last_error(format!(
            "unsupported config struct_version {} (expected {ST3_CONFIG_V1})",
            prefix.struct_version
        ));
        return Err(St3Status::ErrInvalidInput);
    }
    let want = size_of::<St3Config>();
    if prefix.struct_size as usize != want {
        set_last_error(format!(
            "config struct_size {} does not match this ABI's size {want}",
            prefix.struct_size
        ));
        return Err(St3Status::ErrInvalidInput);
    }
    Ok(())
}

/// How source samples are aggregated per environment. Named constants for the
/// `St3Config.collapse` field.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum St3Collapse {
    /// Per-taxon mean over the environment's source samples (floored to integer
    /// counts). The default.
    Mean = 0,
    /// Per-taxon sum over the environment's source samples.
    Sum = 1,
}

/// Which per-sink estimator to run. Named constant for the
/// `St3Config.estimator` field. Only the collapsed-Gibbs sampler ships in v1.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum St3EstimatorKind {
    /// The collapsed-Gibbs source-attribution sampler.
    GibbsCollapsed = 0,
}

/// Versioned run configuration passed to `st3_run`.
///
/// The layout is frozen for a given `struct_version`; new fields are only ever
/// appended under a higher version. `collapse` and `estimator` each hold a
/// `St3Collapse` / `St3EstimatorKind` value (as an `int32_t`); the boolean flags
/// are `uint8_t` (0 = false, nonzero = true). A rarefaction depth of `0`
/// disables rarefaction for that role.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct St3Config {
    /// ABI layout version; must equal `ST3_CONFIG_V1`.
    pub struct_version: u32,
    /// `size_of` the config struct; must equal the library's own size.
    pub struct_size: u32,
    /// Run seed. Output is a function of the seed alone, not the job count.
    pub seed: u64,
    /// Worker count: `0` uses all logical cores, `1` runs serially, `n` uses `n`.
    pub jobs: i32,
    /// Target depth for source samples; `0` disables source rarefaction.
    pub source_rarefaction_depth: i32,
    /// Target depth for sink samples; `0` disables sink rarefaction.
    pub sink_rarefaction_depth: i32,
    /// Rarefy with replacement (multinomial) rather than without (reservoir).
    /// `0` = false, nonzero = true.
    pub with_replacement: u8,
    /// Collapse method: an `St3Collapse` value.
    pub collapse: i32,
    /// Run leave-one-out source prediction rather than sink prediction.
    /// `0` = false, nonzero = true.
    pub loo: u8,
    /// Emit the per-sink source × taxon assignment tally. `0` = false, nonzero =
    /// true.
    pub contingency: u8,
    /// Estimator kind: an `St3EstimatorKind` value.
    pub estimator: i32,
    /// Dirichlet prior count per feature in the source environments.
    pub alpha1: f64,
    /// Dirichlet prior count per feature in the Unknown environment.
    pub alpha2: f64,
    /// Prior count of sink sequences in each environment.
    pub beta: f64,
    /// Independent Markov chains grown per sink.
    pub restarts: u32,
    /// Retained draws per restart.
    pub draws_per_restart: u32,
    /// Passes made before the first retained draw.
    pub burnin: u32,
    /// Passes between retained draws (thinning gap).
    pub delay: u32,
}

/// The validated core inputs derived from an [`St3Config`].
#[derive(Debug)]
pub(crate) struct RunPlan {
    /// The sampler configuration.
    pub params: GibbsParams,
    /// The rarefaction configuration (depths may be `None`).
    pub rarefy: RarefyConfig,
    /// The run seed.
    pub seed: u64,
    /// The worker count (`0` = all cores).
    pub jobs: usize,
    /// Whether to run leave-one-out instead of sink prediction.
    pub loo: bool,
}

impl RunPlan {
    /// Whether any rarefaction was requested (either role has a positive depth).
    pub(crate) fn rarefaction_requested(&self) -> bool {
        self.rarefy.source_depth.is_some() || self.rarefy.sink_depth.is_some()
    }
}

impl St3Config {
    /// Validate the configuration and lower it to core inputs.
    ///
    /// Rejects a config whose `struct_version` / `struct_size` does not match
    /// this ABI, whose `collapse` / `estimator` is not a recognized discriminant,
    /// or whose `jobs` / rarefaction depth is negative; it also runs
    /// [`GibbsParams::validate`]. On any failure it records a descriptive
    /// last-error and returns the matching [`St3Status`].
    pub(crate) fn to_core(self) -> Result<RunPlan, St3Status> {
        if self.struct_version != ST3_CONFIG_V1 {
            set_last_error(format!(
                "unsupported config struct_version {} (expected {ST3_CONFIG_V1})",
                self.struct_version
            ));
            return Err(St3Status::ErrInvalidInput);
        }
        let want = size_of::<St3Config>();
        if self.struct_size as usize != want {
            set_last_error(format!(
                "config struct_size {} does not match this ABI's size {want}",
                self.struct_size
            ));
            return Err(St3Status::ErrInvalidInput);
        }

        let collapse = collapse_from_i32(self.collapse)?;
        estimator_check(self.estimator)?;

        if self.jobs < 0 {
            set_last_error(format!("jobs must be >= 0, got {}", self.jobs));
            return Err(St3Status::ErrInvalidInput);
        }

        let params = GibbsParams {
            alpha1: self.alpha1,
            alpha2: self.alpha2,
            beta: self.beta,
            restarts: self.restarts,
            draws_per_restart: self.draws_per_restart,
            burnin: self.burnin,
            delay: self.delay,
            collapse,
            contingency: self.contingency != 0,
        };
        params.validate().map_err(|e| {
            set_last_error(&e);
            crate::status::status_of_core(&e)
        })?;

        let rarefy = RarefyConfig {
            source_depth: depth_from_i32(
                self.source_rarefaction_depth,
                "source_rarefaction_depth",
            )?,
            sink_depth: depth_from_i32(self.sink_rarefaction_depth, "sink_rarefaction_depth")?,
            with_replacement: self.with_replacement != 0,
        };

        Ok(RunPlan {
            params,
            rarefy,
            seed: self.seed,
            jobs: self.jobs as usize,
            loo: self.loo != 0,
        })
    }
}

/// Map a `collapse` discriminant onto [`CollapseMethod`].
fn collapse_from_i32(v: i32) -> Result<CollapseMethod, St3Status> {
    if v == St3Collapse::Mean as i32 {
        Ok(CollapseMethod::Mean)
    } else if v == St3Collapse::Sum as i32 {
        Ok(CollapseMethod::Sum)
    } else {
        set_last_error(format!(
            "unknown collapse selector {v} (expected {} for MEAN or {} for SUM)",
            St3Collapse::Mean as i32,
            St3Collapse::Sum as i32
        ));
        Err(St3Status::ErrInvalidInput)
    }
}

/// Verify the `estimator` discriminant names a supported estimator.
fn estimator_check(v: i32) -> Result<(), St3Status> {
    if v == St3EstimatorKind::GibbsCollapsed as i32 {
        Ok(())
    } else {
        set_last_error(format!(
            "unsupported estimator selector {v} (only {} = GIBBS_COLLAPSED is available)",
            St3EstimatorKind::GibbsCollapsed as i32
        ));
        Err(St3Status::ErrInvalidInput)
    }
}

/// Map a rarefaction depth field onto `Option<u32>` (`0` disables; negative is
/// rejected).
fn depth_from_i32(v: i32, field: &str) -> Result<Option<u32>, St3Status> {
    if v < 0 {
        set_last_error(format!("{field} must be >= 0, got {v}"));
        Err(St3Status::ErrInvalidInput)
    } else if v == 0 {
        Ok(None)
    } else {
        Ok(Some(v as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// A valid v1 config with the committed reference parameters.
    fn valid_config() -> St3Config {
        St3Config {
            struct_version: ST3_CONFIG_V1,
            struct_size: size_of::<St3Config>() as u32,
            seed: 42,
            jobs: 1,
            source_rarefaction_depth: 0,
            sink_rarefaction_depth: 0,
            with_replacement: 0,
            collapse: St3Collapse::Sum as i32,
            loo: 0,
            contingency: 0,
            estimator: St3EstimatorKind::GibbsCollapsed as i32,
            alpha1: 0.001,
            alpha2: 0.1,
            beta: 10.0,
            restarts: 100,
            draws_per_restart: 10,
            burnin: 100,
            delay: 1,
        }
    }

    #[test]
    fn config_size_is_pinned() {
        // ABI golden: any change to the struct layout must be deliberate.
        assert_eq!(size_of::<St3Config>(), 88);
    }

    #[test]
    fn config_field_offsets_are_pinned() {
        assert_eq!(offset_of!(St3Config, struct_version), 0);
        assert_eq!(offset_of!(St3Config, struct_size), 4);
        assert_eq!(offset_of!(St3Config, seed), 8);
        assert_eq!(offset_of!(St3Config, jobs), 16);
        assert_eq!(offset_of!(St3Config, source_rarefaction_depth), 20);
        assert_eq!(offset_of!(St3Config, sink_rarefaction_depth), 24);
        assert_eq!(offset_of!(St3Config, with_replacement), 28);
        assert_eq!(offset_of!(St3Config, collapse), 32);
        assert_eq!(offset_of!(St3Config, loo), 36);
        assert_eq!(offset_of!(St3Config, contingency), 37);
        assert_eq!(offset_of!(St3Config, estimator), 40);
        assert_eq!(offset_of!(St3Config, alpha1), 48);
        assert_eq!(offset_of!(St3Config, alpha2), 56);
        assert_eq!(offset_of!(St3Config, beta), 64);
        assert_eq!(offset_of!(St3Config, restarts), 72);
        assert_eq!(offset_of!(St3Config, draws_per_restart), 76);
        assert_eq!(offset_of!(St3Config, burnin), 80);
        assert_eq!(offset_of!(St3Config, delay), 84);
    }

    #[test]
    fn enum_discriminants_are_pinned() {
        assert_eq!(St3Collapse::Mean as i32, 0);
        assert_eq!(St3Collapse::Sum as i32, 1);
        assert_eq!(St3EstimatorKind::GibbsCollapsed as i32, 0);
        assert_eq!(St3Status::Ok as i32, 0);
    }

    #[test]
    fn to_core_maps_a_valid_config() {
        let plan = valid_config().to_core().expect("valid config");
        assert_eq!(plan.seed, 42);
        assert_eq!(plan.jobs, 1);
        assert!(!plan.loo);
        assert!(!plan.rarefaction_requested());
        assert_eq!(plan.params.collapse, CollapseMethod::Sum);
        assert_eq!(plan.params.alpha1, 0.001);
        assert_eq!(plan.params.beta, 10.0);
        assert_eq!(plan.params.restarts, 100);
        assert_eq!(plan.rarefy.source_depth, None);
        assert_eq!(plan.rarefy.sink_depth, None);
    }

    #[test]
    fn to_core_maps_rarefaction_and_mean_collapse() {
        let cfg = St3Config {
            collapse: St3Collapse::Mean as i32,
            source_rarefaction_depth: 1000,
            sink_rarefaction_depth: 500,
            with_replacement: 1,
            loo: 1,
            ..valid_config()
        };
        let plan = cfg.to_core().expect("valid config");
        assert_eq!(plan.params.collapse, CollapseMethod::Mean);
        assert!(plan.loo);
        assert!(plan.rarefaction_requested());
        assert_eq!(plan.rarefy.source_depth, Some(1000));
        assert_eq!(plan.rarefy.sink_depth, Some(500));
        assert!(plan.rarefy.with_replacement);
    }

    #[test]
    fn validate_prefix_accepts_valid_and_rejects_mismatch() {
        let good = valid_config();
        // SAFETY: `&good` points to a full, initialized St3Config.
        assert!(unsafe { validate_prefix(&good) }.is_ok());

        let bad_version = St3Config {
            struct_version: 999,
            ..valid_config()
        };
        // SAFETY: as above.
        assert_eq!(
            unsafe { validate_prefix(&bad_version) }.unwrap_err(),
            St3Status::ErrInvalidInput
        );

        let bad_size = St3Config {
            struct_size: 8,
            ..valid_config()
        };
        // SAFETY: as above.
        assert_eq!(
            unsafe { validate_prefix(&bad_size) }.unwrap_err(),
            St3Status::ErrInvalidInput
        );
    }

    #[test]
    fn to_core_rejects_wrong_version() {
        let cfg = St3Config {
            struct_version: 999,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_wrong_size() {
        let cfg = St3Config {
            struct_size: 1,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_unknown_collapse() {
        let cfg = St3Config {
            collapse: 7,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_unknown_estimator() {
        let cfg = St3Config {
            estimator: 1,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_negative_jobs() {
        let cfg = St3Config {
            jobs: -1,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_negative_depth() {
        let cfg = St3Config {
            source_rarefaction_depth: -5,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn to_core_rejects_invalid_params() {
        // Zero restarts fails GibbsParams::validate -> ErrInvalidInput.
        let cfg = St3Config {
            restarts: 0,
            ..valid_config()
        };
        assert_eq!(cfg.to_core().unwrap_err(), St3Status::ErrInvalidInput);
    }
}
