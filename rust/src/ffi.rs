//! Raw declarations of include/sdca/c_api.h (ABI version 6): its structs, with
//! `Default` as their `SDCA_*_INIT` initializers, and its entry points.
use std::ffi::{c_char, c_double, c_int, c_void};
use std::mem::size_of;
use std::ptr;

/// SDCA_ABI_VERSION.
pub const ABI_VERSION: c_int = 6;

/// SDCA_INVALID_ARGUMENT of enum sdca_result.
pub const INVALID_ARGUMENT: c_int = -2;

/// SDCA_HISTORY_FIELDS: the columns of a history row.
pub const HISTORY_FIELDS: usize = 10;

/// enum sdca_precision.
pub const PRECISION_DEFAULT: c_int = 0;
pub const PRECISION_FLOAT32: c_int = 1;
pub const PRECISION_FLOAT64: c_int = 2;

/// enum sdca_target_kind.
pub const TARGET_LABELS: c_int = 0;
pub const TARGET_LABEL_SETS: c_int = 1;
pub const TARGET_PROBABILITIES: c_int = 3;
pub const TARGET_LOGITS: c_int = 4;

/// enum sdca_input_kind.
pub const INPUT_FEATURES: c_int = 0;
pub const INPUT_KERNEL: c_int = 1;

/// struct sdca_loss.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Loss {
    pub size: usize,
    pub kind: c_int,
    pub k: usize,
    pub gamma: c_double,
    pub alpha: c_double,
    pub temperature: c_double,
    pub projection_precision: c_int,
}

impl Default for Loss {
    /// SDCA_LOSS_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            kind: 0,
            k: 0,
            gamma: 0.0,
            alpha: 1.0,
            temperature: 1.0,
            projection_precision: 0,
        }
    }
}

/// struct sdca_execution.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Execution {
    pub size: usize,
    pub precision: c_int,
    pub num_threads: usize,
}

impl Default for Execution {
    /// SDCA_EXECUTION_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            precision: PRECISION_DEFAULT,
            num_threads: 0,
        }
    }
}

/// struct sdca_targets.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Targets {
    pub size: usize,
    pub kind: c_int,
    pub labels: *const i64,
    pub offsets: *const i64,
    pub mask: *const u8,
    pub probabilities: *const c_double,
    pub logits: *const c_void,
}

impl Default for Targets {
    /// SDCA_TARGETS_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            kind: TARGET_LABELS,
            labels: ptr::null(),
            offsets: ptr::null(),
            mask: ptr::null(),
            probabilities: ptr::null(),
            logits: ptr::null(),
        }
    }
}

/// struct sdca_input.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub size: usize,
    pub kind: c_int,
    pub dtype: c_int,
    pub n: usize,
    pub d: usize,
    pub x: *const c_void,
}

impl Default for Input {
    /// SDCA_INPUT_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            kind: INPUT_FEATURES,
            dtype: 1,
            n: 0,
            d: 0,
            x: ptr::null(),
        }
    }
}

/// struct sdca_fit_options.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub size: usize,
    pub c: c_double,
    pub tol: c_double,
    pub max_epoch: usize,
    pub eval_epoch: usize,
    pub eval_on_start: c_int,
    pub recompute_primal: c_int,
    pub shrinking: c_int,
    pub seed: u64,
}

impl Default for FitOptions {
    /// SDCA_FIT_OPTIONS_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            c: 1.0,
            tol: 1e-3,
            max_epoch: 1000,
            eval_epoch: 10,
            eval_on_start: 0,
            recompute_primal: 1,
            shrinking: 0,
            seed: 0,
        }
    }
}

/// struct sdca_fit_result.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FitResult {
    pub size: usize,
    pub status: c_int,
    pub epochs: usize,
    pub history_rows: usize,
}

impl Default for FitResult {
    /// SDCA_FIT_RESULT_INIT.
    fn default() -> Self {
        Self {
            size: size_of::<Self>(),
            status: 2,
            epochs: 0,
            history_rows: 0,
        }
    }
}

extern "C" {
    pub fn sdca_loss_evaluate(
        loss: *const Loss,
        execution: *const Execution,
        dtype: c_int,
        n: usize,
        num_classes: usize,
        scores: *const c_void,
        targets: *const Targets,
        values: *mut c_double,
        gradients: *mut c_void,
    ) -> c_int;
    pub fn sdca_dual_update(
        loss: *const Loss,
        c: c_double,
        execution: *const Execution,
        dtype: c_int,
        n: usize,
        num_classes: usize,
        scores: *const c_void,
        targets: *const Targets,
        previous: *const c_void,
        norm2: *const c_double,
        out: *mut c_void,
    ) -> c_int;
    pub fn sdca_history_rows(options: *const FitOptions) -> usize;
    pub fn sdca_max_extent() -> usize;
    pub fn sdca_model_fit(
        loss: *const Loss,
        options: *const FitOptions,
        execution: *const Execution,
        input: *const Input,
        num_classes: usize,
        targets: *const Targets,
        center: *const c_void,
        initial_dual: *const c_void,
        weights: *mut c_void,
        dual: *mut c_void,
        history: *mut c_double,
        history_rows: usize,
        result: *mut FitResult,
    ) -> c_int;
    pub fn sdca_abi_version() -> c_int;
    pub fn sdca_version() -> *const c_char;
    pub fn sdca_last_error() -> *const c_char;
}
