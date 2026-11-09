//! Raw declarations of include/sdca/c_api.h (the entry points the crate calls).
use std::ffi::{c_char, c_double, c_int, c_void};

/// struct sdca_options.
#[repr(C)]
pub struct Options {
    pub size: usize,
    pub precision: c_int,
    pub loss: c_int,
    pub c: c_double,
    pub k: usize,
    pub gamma: c_double,
    pub alpha: c_double,
    pub projection_precision: c_int,
    pub num_threads: usize,
    pub max_epoch: usize,
    pub eval_epoch: usize,
    pub tol: c_double,
    pub eval_on_start: c_int,
    pub recompute_primal: c_int,
    pub kernel: c_int,
    pub warm: c_int,
    pub shrinking: c_int,
}

impl Default for Options {
    /// SDCA_OPTIONS_DEFAULT.
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            precision: 2,
            loss: 0,
            c: 1.0,
            k: 1,
            gamma: 0.0,
            alpha: 1.0,
            projection_precision: 0,
            num_threads: usize::MAX,
            max_epoch: 1000,
            eval_epoch: 10,
            tol: 1e-3,
            eval_on_start: 0,
            recompute_primal: 1,
            kernel: 0,
            warm: 0,
            shrinking: 0,
        }
    }
}

/// struct sdca_fit_result.
#[repr(C)]
pub struct FitResult {
    pub size: usize,
    pub status: c_int,
    pub epochs: usize,
    pub history_used: usize,
}

impl Default for FitResult {
    /// SDCA_FIT_RESULT_INIT.
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            status: 0,
            epochs: 0,
            history_used: 0,
        }
    }
}

extern "C" {
    pub fn sdca_abi_version() -> c_int;
    pub fn sdca_version() -> *const c_char;
    pub fn sdca_last_error() -> *const c_char;
    pub fn sdca_history_fields() -> usize;
    pub fn sdca_loss_gradient(
        options: *const Options,
        n: usize,
        k: usize,
        z: *const c_void,
        positive: *const u8,
        values: *mut c_double,
        gradients: *mut c_void,
    ) -> c_int;
    pub fn sdca_update_objective(
        options: *const Options,
        n: usize,
        k: usize,
        z: *const c_void,
        labels: *const i64,
        offsets: *const i64,
        previous: *const c_void,
        norm2: c_double,
        out: *mut c_void,
    ) -> c_int;
    pub fn sdca_fit(
        options: *const Options,
        n: usize,
        d: usize,
        k: usize,
        x: *const c_void,
        labels: *const i64,
        offsets: *const i64,
        center: *const c_void,
        w: *mut c_void,
        a: *mut c_void,
        history: *mut c_double,
        history_rows: usize,
        result: *mut FitResult,
    ) -> c_int;
    pub fn sdca_update_distillation(
        options: *const Options,
        n: usize,
        k: usize,
        z: *const c_void,
        targets: *const c_double,
        previous: *const c_void,
        norm2: c_double,
        out: *mut c_void,
    ) -> c_int;
    pub fn sdca_fit_distillation(
        options: *const Options,
        n: usize,
        d: usize,
        k: usize,
        x: *const c_void,
        targets: *const c_double,
        center: *const c_void,
        w: *mut c_void,
        a: *mut c_void,
        history: *mut c_double,
        history_rows: usize,
        result: *mut FitResult,
    ) -> c_int;
    pub fn sdca_distillation_forward(
        options: *const Options,
        dtype: c_int,
        n: usize,
        k: usize,
        student: *const c_void,
        teacher: *const c_void,
        temperature: c_double,
        values: *mut c_double,
        stats: *mut c_double,
    ) -> c_int;
    pub fn sdca_distillation_backward(
        options: *const Options,
        dtype: c_int,
        n: usize,
        k: usize,
        student: *const c_void,
        teacher: *const c_void,
        temperature: c_double,
        stats: *const c_double,
        scale: *const c_double,
        scale_stride: usize,
        gradients: *mut c_void,
    ) -> c_int;
    pub fn sdca_proximal_distillation_rows(
        options: *const Options,
        dtype: c_int,
        n: usize,
        k: usize,
        student: *const c_void,
        teacher: *const c_void,
        temperature: c_double,
        alpha: c_double,
        values: *mut c_double,
        gradients: *mut c_void,
    ) -> c_int;
}
