/// Element type of a [`crate::Tensor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit float (default compute type).
    F32,
    /// 64-bit float (reference / tiny checks).
    F64,
    /// 64-bit signed integer (indices).
    I64,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size_of(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F64 => 8,
            Dtype::I64 => 8,
        }
    }

    /// Whether this is a floating type.
    pub fn is_float(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F64)
    }

    /// Short name (`"f32"`, `"f64"`, `"i64"`).
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::I64 => "i64",
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
