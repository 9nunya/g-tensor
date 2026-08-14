use std::fmt;

use crate::device::Device;
use crate::dtype::Dtype;

/// Class of a recoverable library error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Shape,
    Dtype,
    Device,
    NotUnique,
    Overlap,
    Unsupported,
    BackendPrecheck,
    Index,
    Domain,
}

/// Fallible op error. Display is `"kind error in op: message"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub op: &'static str,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, op: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            op,
            message: message.into(),
        }
    }

    pub fn shape(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Shape, op, message)
    }

    pub fn dtype(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Dtype, op, message)
    }

    pub fn device(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Device, op, message)
    }

    pub fn index(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Index, op, message)
    }

    pub fn unsupported(
        op: &'static str,
        dtype: Dtype,
        device: Device,
        hint: impl Into<String>,
    ) -> Self {
        Self::new(
            ErrorKind::Unsupported,
            op,
            format!("dtype={dtype:?} device={device:?}: {}", hint.into()),
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} error in {}: {}",
            kind_name(self.kind),
            self.op,
            self.message
        )
    }
}

impl std::error::Error for Error {}

fn kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Shape => "shape",
        ErrorKind::Dtype => "dtype",
        ErrorKind::Device => "device",
        ErrorKind::NotUnique => "not-unique",
        ErrorKind::Overlap => "overlap",
        ErrorKind::Unsupported => "unsupported",
        ErrorKind::BackendPrecheck => "backend-precheck",
        ErrorKind::Index => "index",
        ErrorKind::Domain => "domain",
    }
}

/// [`std::result::Result`] alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
