//! The calls into the C ABI and the checks they share.
use std::ffi::{c_int, c_void, CStr};
use std::sync::OnceLock;

use crate::error::{invalid, Error, Result};
use crate::ffi;
use crate::names::Precision;
use crate::settings::Settings;

/// Whether the native library has the wrapper's C ABI (which fixes the history layout).
fn check_abi(abi: c_int) -> Result<()> {
    if abi != ffi::ABI_VERSION {
        return Err(Error::Native(format!(
            "Library and wrapper disagree on the C ABI: version {abi}, not {}",
            ffi::ABI_VERSION
        )));
    }
    Ok(())
}

/// The ABI handshake, done once, before any computing call.
pub(crate) fn handshake() -> Result<()> {
    static HANDSHAKE: OnceLock<Result<()>> = OnceLock::new();
    HANDSHAKE
        // SAFETY: a plain query of the native library.
        .get_or_init(|| check_abi(unsafe { ffi::sdca_abi_version() }))
        .clone()
}

/// The message of this thread's last failed native call.
fn last_error() -> String {
    // SAFETY: sdca_last_error returns a NUL-terminated thread-local string.
    let message = unsafe { CStr::from_ptr(ffi::sdca_last_error()) };
    message.to_string_lossy().into_owned()
}

/// The result of a computing call, with the native message of a failure.
pub(crate) fn call(code: c_int) -> Result<()> {
    match code {
        0 => Ok(()),
        ffi::INVALID_ARGUMENT => Err(Error::InvalidArgument(last_error())),
        _ => Err(Error::Native(last_error())),
    }
}

/// The largest size the native BLAS takes (`sdca_max_extent`).
pub(crate) fn max_extent() -> usize {
    // SAFETY: sdca_max_extent takes no arguments and only reads a constant.
    unsafe { ffi::sdca_max_extent() }
}

/// The history rows a fit with `options` may write; invalid options are an
/// invalid argument with the native message.
pub(crate) fn history_rows(options: &ffi::FitOptions) -> Result<usize> {
    const TOO_LARGE: &str = "max_epoch / eval_epoch is too large for the history buffer";
    // SAFETY: options is a valid struct sdca_fit_options.
    let rows = unsafe { ffi::sdca_history_rows(options) };
    if rows == 0 {
        let message = last_error();
        // An empty message: max_epoch / eval_epoch + 2 wrapped around to 0.
        return invalid(if message.is_empty() {
            TOO_LARGE.into()
        } else {
            message
        });
    }
    // max_epoch / eval_epoch + 2 wraps around to 1 for max_epoch = usize::MAX.
    if rows < 2 || rows.checked_mul(ffi::HISTORY_FIELDS).is_none() {
        return invalid(TOO_LARGE);
    }
    Ok(rows)
}

/// The C ABI execution settings of `settings`.
pub(crate) fn execution(settings: &Settings) -> ffi::Execution {
    ffi::Execution {
        precision: match settings.precision {
            None => ffi::PRECISION_DEFAULT,
            Some(Precision::Float32) => ffi::PRECISION_FLOAT32,
            Some(Precision::Float64) => ffi::PRECISION_FLOAT64,
        },
        num_threads: settings.threads.code(),
        check_rows: c_int::from(settings.check_rows),
        ..Default::default()
    }
}

pub(crate) fn void<T>(values: &[T]) -> *const c_void {
    values.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handshake_rejects_another_abi() {
        assert_eq!(check_abi(ffi::ABI_VERSION), Ok(()));
        // ABI 5 builds take struct sdca_options.
        for abi in [5, ffi::ABI_VERSION + 1] {
            assert!(check_abi(abi).unwrap_err().to_string().contains("C ABI"));
        }
    }

    #[test]
    fn the_initializers_match_the_header() {
        assert_eq!(handshake(), Ok(()));
        let options = ffi::FitOptions::default();
        // max_epoch / eval_epoch + 2 of SDCA_FIT_OPTIONS_INIT.
        assert_eq!(history_rows(&options), Ok(102));
        let end = ffi::FitOptions {
            eval_epoch: 0,
            ..options
        };
        assert_eq!(history_rows(&end), Ok(2));
        let bad = ffi::FitOptions { c: 0.0, ..options };
        assert!(
            matches!(history_rows(&bad), Err(Error::InvalidArgument(m)) if m.contains("C must"))
        );
        for max_epoch in [usize::MAX, usize::MAX - 1] {
            let wraps = ffi::FitOptions {
                max_epoch,
                eval_epoch: 1,
                ..options
            };
            let error = history_rows(&wraps).unwrap_err().to_string();
            assert!(error.contains("too large"), "{error}");
        }
    }
}
