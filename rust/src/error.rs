use std::fmt;

/// An invalid argument (Python: ValueError) or a native failure (RuntimeError).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// The inputs or options are invalid; nothing was computed.
    InvalidArgument(String),
    /// The native library reported a failure.
    Native(String),
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(message) | Error::Native(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::InvalidArgument(message.into()))
}
