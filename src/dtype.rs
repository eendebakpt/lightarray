/// Data type representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float64,
    Int64,
}

impl DType {
    pub fn as_str(&self) -> &str {
        match self {
            DType::Float64 => "float64",
            DType::Int64 => "int64",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
