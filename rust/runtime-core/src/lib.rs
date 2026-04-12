use std::fmt::{Display, Formatter};

use candle_core::{Device, DeviceLocation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Hip,
    Cuda,
    Metal,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Hip => "hip",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

impl Display for BackendKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSpec {
    pub backend: BackendKind,
    pub family: String,
}

impl TargetSpec {
    pub fn detect(device: &Device) -> Self {
        if let Ok(override_family) = std::env::var("DOTCACHE_MODEL_PACKAGE_FAMILY") {
            return Self {
                backend: backend_kind_for_device(device),
                family: override_family,
            };
        }

        match device.location() {
            DeviceLocation::Cpu => Self {
                backend: BackendKind::Cpu,
                family: "host".to_string(),
            },
            DeviceLocation::Hip { .. } => Self {
                backend: BackendKind::Hip,
                family: detect_hip_family().unwrap_or_else(|| "hip-generic".to_string()),
            },
            DeviceLocation::Cuda { .. } => Self {
                backend: BackendKind::Cuda,
                family: detect_cuda_family().unwrap_or_else(|| "cuda-generic".to_string()),
            },
            DeviceLocation::Metal { .. } => Self {
                backend: BackendKind::Metal,
                family: "metal-generic".to_string(),
            },
        }
    }
}

pub fn backend_kind_for_device(device: &Device) -> BackendKind {
    match device.location() {
        DeviceLocation::Cpu => BackendKind::Cpu,
        DeviceLocation::Hip { .. } => BackendKind::Hip,
        DeviceLocation::Cuda { .. } => BackendKind::Cuda,
        DeviceLocation::Metal { .. } => BackendKind::Metal,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferMutability {
    Mutable,
    Immutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ScalarType {
    U8,
    U32,
    I16,
    I32,
    I64,
    BF16,
    F16,
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferViewDesc {
    pub scalar_type: ScalarType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub mutability: BufferMutability,
}

#[derive(Debug, Clone)]
pub struct ImmutableBufferView<'a> {
    pub target: TargetSpec,
    pub desc: BufferViewDesc,
    pub bytes: &'a [u8],
}

impl<'a> ImmutableBufferView<'a> {
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

fn detect_hip_family() -> Option<String> {
    let output = std::process::Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|token| token.starts_with("gfx"))
        .map(ToOwned::to_owned)
}

fn detect_cuda_family() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut seen_header = false;
    for token in stdout.split_whitespace() {
        if token == "CUDA" {
            seen_header = true;
            continue;
        }
        if seen_header {
            let digits = token
                .chars()
                .filter(|ch| ch.is_ascii_digit())
                .collect::<String>();
            if digits.len() >= 2 {
                return Some(format!("sm{digits}"));
            }
        }
    }
    None
}
