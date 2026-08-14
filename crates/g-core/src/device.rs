/// Where tensor storage lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host / CPU buffers (the v1 default).
    Cpu,
    /// GPU storage. Not a default placement; the `gpu` feature is a stub plus large GEMM.
    Gpu,
}

impl Device {
    /// `"cpu"` or `"gpu"`.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
