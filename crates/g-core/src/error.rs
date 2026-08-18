use std::fmt;

use crate::device::Device;
use crate::dtype::Dtype;

/// Class of a recoverable library error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Bad shape, stride, or broadcasting.
    Shape,
    /// Wrong element type for the op.
    Dtype,
    /// Wrong or unavailable device.
    Device,
    /// In-place op on a shared tensor.
    NotUnique,
    /// Overlapping views in a scatter/copy.
    Overlap,
    /// Known-but-not-implemented combination.
    Unsupported,
    /// A kernel refused to run and the caller should fall back.
    BackendPrecheck,
    /// Index out of range.
    Index,
    /// Domain error (NaN from sqrt/log of negative input, etc.).
    Domain,
}

/// Fallible op error. Display is `"kind error in op: message"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Error class.
    pub kind: ErrorKind,
    /// Op that produced the error (for messages and logging).
    pub op: &'static str,
    /// Human-readable detail.
    pub message: String,
}

impl Error {
    /// Build an error of any [`ErrorKind`].
    pub fn new(kind: ErrorKind, op: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            op,
            message: message.into(),
        }
    }

    /// Shorthand for an [`ErrorKind::Shape`] error.
    pub fn shape(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Shape, op, message)
    }

    /// Shorthand for an [`ErrorKind::Dtype`] error.
    pub fn dtype(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Dtype, op, message)
    }

    /// Shorthand for an [`ErrorKind::Device`] error.
    pub fn device(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Device, op, message)
    }

    /// Shorthand for an [`ErrorKind::Index`] error.
    pub fn index(op: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Index, op, message)
    }

    /// Build an [`ErrorKind::Unsupported`] error with dtype/device context.
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
