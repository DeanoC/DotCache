#[cfg(feature = "qwen35-minimal")]
mod backends;
#[cfg(feature = "hf")]
mod hf;
#[cfg(feature = "qwen35-minimal")]
#[path = "qwen35_minimal/mod.rs"]
mod qwen35_minimal_impl;

use std::fmt::{Display, Formatter};

pub use dotcache_model_store::{
    adapters::candle::CandleWeightProvider, ImmutableWeightHandle, PreparedDType,
    PreparedPackageProfile, PreparedPackageSummary, PreparedQwen35DirectMetadata,
    PreparedTensorEncoding, PreparedTensorEntry, WeightLoadStats, WeightProvider, WeightView,
};
pub use dotcache_runtime_core::{
    backend_kind_for_device, BackendKind, BufferMutability, BufferViewDesc, ImmutableBufferView,
    ScalarType, TargetSpec,
};

#[cfg(feature = "hf")]
pub use hf::{HfHubModelSource, HfModelArtifacts, HfModelWeightIndex};
#[cfg(feature = "qwen35-minimal")]
pub use qwen35_minimal_impl::{
    MinimalQwen35Config, MinimalQwen35DirectRuntime, MinimalQwen35DirectRuntimeProfile,
    MinimalQwen35DecoderLayerTrace, MinimalQwen35KvCache, MinimalQwen35LinearAttentionBenchResult,
    MinimalQwen35LinearAttentionLayerSpec, MinimalQwen35LinearAttentionTrace,
    MinimalQwen35LoadMode, MinimalQwen35LoadTrace, MinimalQwen35NativeCacheState,
    MinimalQwen35NativeFullAttentionCacheState, MinimalQwen35NativeLayerCacheState,
    MinimalQwen35NativeLinearAttentionCacheState, MinimalQwen35Runner,
    MinimalQwen35RuntimeProfile, MinimalQwen35StateBuffer, MinimalQwen35TextConfig,
    MinimalQwen35Weights, ModelForCausalLM,
};
#[cfg(feature = "qwen35-minimal")]
pub type Qwen35Runtime = MinimalQwen35Runner;
#[cfg(feature = "qwen35-minimal")]
pub type Qwen35Weights = MinimalQwen35Weights;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    MissingAsset {
        model_id: String,
        filename: String,
    },
    External {
        context: &'static str,
        message: String,
    },
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAsset { model_id, filename } => {
                write!(f, "missing asset `{filename}` for model `{model_id}`")
            }
            Self::External { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub type Result<T> = std::result::Result<T, RuntimeError>;

impl From<candle_core::Error> for RuntimeError {
    fn from(err: candle_core::Error) -> Self {
        Self::External {
            context: "candle",
            message: err.to_string(),
        }
    }
}

impl From<dotcache_model_store::ModelStoreError> for RuntimeError {
    fn from(err: dotcache_model_store::ModelStoreError) -> Self {
        Self::External {
            context: "model-store",
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(err: std::io::Error) -> Self {
        Self::External {
            context: "io",
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "qwen35-minimal")]
impl From<serde_json::Error> for RuntimeError {
    fn from(err: serde_json::Error) -> Self {
        Self::External {
            context: "serde-json",
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "hf")]
impl From<hf_hub::api::sync::ApiError> for RuntimeError {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        Self::External {
            context: "hf-hub",
            message: err.to_string(),
        }
    }
}

pub mod model_package {
    pub use dotcache_model_store::{
        BackendKind, CandleWeightProvider, ImmutableWeightHandle, ModelFamilyConverter,
        ModelPackageProfile, ModelStoreError, PackageKey, PreparedDType, PreparedPackageProfile,
        PreparedPackageSummary, PreparedQwen35DirectMetadata, PreparedTensorEncoding,
        PreparedTensorEntry, TargetSpec, WeightLoadStats, WeightProvider, WeightView,
    };
    pub use dotcache_model_store::{
        ModelTarget, PreparedModelManifest as ModelManifest, PreparedModelPackage as ModelPackage,
        PreparedTensorLayout as TensorLayout,
    };
}

pub use model_package::ModelPackage;

#[cfg(feature = "qwen35-minimal")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35BackendDescriptor {
    pub target: TargetSpec,
    pub optimized: bool,
}

#[cfg(feature = "qwen35-minimal")]
pub trait Qwen35BackendOps {
    fn descriptor(&self) -> &Qwen35BackendDescriptor;
}

#[cfg(feature = "qwen35-minimal")]
#[derive(Debug, Clone)]
pub struct Qwen35Backend {
    descriptor: Qwen35BackendDescriptor,
}

#[cfg(feature = "qwen35-minimal")]
impl Qwen35Backend {
    pub fn for_device(device: &candle_core::Device) -> Self {
        let target = TargetSpec::detect(device);
        match target.backend {
            BackendKind::Cpu => backends::cpu::backend(target),
            BackendKind::Hip => backends::hip::backend(target),
            BackendKind::Cuda => backends::cuda::backend(target),
            BackendKind::Metal => backends::metal::backend(target),
        }
    }
}

#[cfg(feature = "qwen35-minimal")]
impl Qwen35BackendOps for Qwen35Backend {
    fn descriptor(&self) -> &Qwen35BackendDescriptor {
        &self.descriptor
    }
}
