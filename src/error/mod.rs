//! Bridge errors with severity classification.

use std::fmt::{Display, Formatter};

/// Severity level attached to an [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Recoverable,
    Fatal,
}

/// Specific failure category for an [`Error`].
#[derive(Debug, Clone)]
pub enum ErrorKind {
    Unsupported(&'static str),
    NoCompatibleDevice,
    SurfaceNotAttached,
    SurfaceOutdated,
    SurfaceLost,
    SurfaceTimeout,
    SurfaceOccluded,
    ResourceNotFound(&'static str),
    InvalidDescriptor(&'static str),
    TextureUploadFailed(&'static str),
    PipelineCreationFailed(&'static str),
    Backend(&'static str),
}

/// Bridge error with a [`ErrorKind`] and [`Severity`].
#[derive(Debug, Clone)]
pub struct Error {
    /// Failure category.
    pub kind: ErrorKind,
    /// Reported severity.
    pub severity: Severity,
}

impl Error {
    /// Creates an error with the given `kind` and `severity`.
    pub const fn new(kind: ErrorKind, severity: Severity) -> Self {
        Self { kind, severity }
    }

    /// Creates an error with [`Severity::Fatal`].
    pub const fn fatal(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Fatal)
    }

    /// Creates an error with [`Severity::Recoverable`].
    pub const fn recoverable(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Recoverable)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:?}", self.severity, self.kind)
    }
}

impl std::error::Error for Error {}
