pub mod backend;
pub mod cache;
#[cfg(feature = "candle")]
pub mod candle_model;
pub mod decode;
#[cfg(feature = "hf")]
pub mod hf;
#[cfg(feature = "candle")]
mod instrumented_llama;
#[cfg(feature = "candle")]
mod instrumented_qwen2;
#[cfg(feature = "candle")]
mod instrumented_qwen35;
pub mod model;
pub mod page;
pub mod page_mode;
#[cfg(feature = "candle")]
pub mod megakernel_control;
#[cfg(feature = "candle")]
pub mod policy;
#[cfg(feature = "qwen35-minimal")]
pub mod qwen35_fast;
#[cfg(feature = "qwen35-minimal")]
pub mod qwen35_minimal;
pub mod session;
#[cfg(feature = "candle")]
pub mod torch_control;
pub mod virtual_page;

pub use backend::{
    AttentionPathMode, BackendDescriptor, BackendDevice, CpuReferenceBackend, PageBackend,
};
#[cfg(feature = "candle")]
pub use backend::{CandleDeviceSelector, CandlePageBackend};
pub use cache::{LayerCache, PageStore, PagedKvCache, SeqCache};
#[cfg(feature = "candle")]
pub use candle_model::{CandleCausalLm, RequestMetrics, RequestSessionMetrics};
pub use decode::{
    decode_one_head, decode_one_head_owned, decode_query_batch_owned, decode_virtual_one_head,
    decode_virtual_one_head_owned, softmax_in_place,
};
#[cfg(feature = "hf")]
pub use hf::{HfHubModelSource, HfModelArtifacts, HfModelWeightIndex};
pub use model::{greedy_generate, CausalLm, GreedyGeneration, ModelArchitecture, ModelFamily};
pub use model::{RuntimeMode, RuntimeStageMetrics};
pub use page::{KvPage, PageId};
pub use page_mode::{
    PageEscapeDType, PageModePolicy, PageModeSpec, PageModeTag, PageQuantScheme, PageSideKind,
};
#[cfg(feature = "candle")]
pub use policy::{default_prompt_policy_table, PromptBucketPolicy, PromptBucketPolicyTable};
#[cfg(feature = "qwen35-minimal")]
pub use qwen35_minimal::{
    MinimalQwen35Config, MinimalQwen35KvCache, MinimalQwen35LinearAttentionLayerSpec,
    MinimalQwen35Runner, MinimalQwen35Weights,
};
#[cfg(feature = "qwen35-minimal")]
pub use qwen35_fast::{Qwen35FastRunner, Qwen35FastTopology};
#[cfg(feature = "candle")]
pub use session::HybridCacheState;
pub use session::{
    KvRow, LayerDecodePlan, SessionDecodePlan, SessionId, SessionMetrics, SessionPrefix,
    SessionRequestKind, SessionRuntime, SessionState, SessionTokenRows,
};
pub use virtual_page::{
    VirtualCacheMetrics, VirtualLayerCache, VirtualPage, VirtualPageId, VirtualPageTable,
    VirtualPagedKvCache, VirtualSeqCache,
};

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    ConversionOverflow {
        field: &'static str,
        value: usize,
    },
    InvalidLayer {
        layer: usize,
        layer_count: usize,
    },
    InvalidKvHead {
        kv_head: usize,
        kv_head_count: usize,
    },
    InvalidPageId {
        page_id: usize,
        page_count: usize,
    },
    InvalidVirtualPageId {
        virtual_page_id: usize,
        page_count: usize,
    },
    InvalidSessionId {
        session_id: usize,
        session_count: usize,
    },
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    PositionMismatch {
        expected: u32,
        got: u32,
    },
    EmptyDecode,
    SealedPage {
        layer: u16,
        kv_head: u16,
    },
    PageBufferMismatch {
        page_id: usize,
        buffer: &'static str,
        expected: usize,
        got: usize,
    },
    EmptyInput {
        context: &'static str,
    },
    UnsupportedPageModeForValue {
        mode: String,
    },
    UnsupportedPageMode {
        mode: String,
        context: &'static str,
    },
    FusedAttentionRequiresExactPages {
        page_id: usize,
        key_mode: String,
        value_mode: String,
    },
    MissingAsset {
        model_id: String,
        filename: String,
    },
    UnsupportedModelFamily {
        family: String,
    },
    BackendUnavailable {
        backend: &'static str,
        device: String,
    },
    External {
        context: &'static str,
        message: String,
    },
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConversionOverflow { field, value } => {
                write!(
                    f,
                    "{field} value {value} does not fit in the runtime metadata width"
                )
            }
            Self::InvalidLayer { layer, layer_count } => {
                write!(f, "layer {layer} is out of range for {layer_count} layers")
            }
            Self::InvalidKvHead {
                kv_head,
                kv_head_count,
            } => {
                write!(
                    f,
                    "kv_head {kv_head} is out of range for {kv_head_count} KV heads"
                )
            }
            Self::InvalidPageId {
                page_id,
                page_count,
            } => {
                write!(
                    f,
                    "page id {page_id} is out of range for {page_count} pages"
                )
            }
            Self::InvalidVirtualPageId {
                virtual_page_id,
                page_count,
            } => {
                write!(
                    f,
                    "virtual page id {virtual_page_id} is out of range for {page_count} virtual pages"
                )
            }
            Self::InvalidSessionId {
                session_id,
                session_count,
            } => {
                write!(
                    f,
                    "session id {session_id} is out of range for {session_count} sessions"
                )
            }
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => {
                write!(f, "{context} expected width {expected}, got {got}")
            }
            Self::PositionMismatch { expected, got } => {
                write!(f, "append expected token position {expected}, got {got}")
            }
            Self::EmptyDecode => write!(f, "decode requires at least one page"),
            Self::SealedPage { layer, kv_head } => {
                write!(
                    f,
                    "page for layer {layer} kv_head {kv_head} is already sealed"
                )
            }
            Self::PageBufferMismatch {
                page_id,
                buffer,
                expected,
                got,
            } => {
                write!(
                    f,
                    "page {page_id} {buffer} buffer expected {expected} elements, got {got}"
                )
            }
            Self::EmptyInput { context } => write!(f, "{context} requires at least one token"),
            Self::UnsupportedPageModeForValue { mode } => {
                write!(f, "page mode {mode} is not supported for value pages")
            }
            Self::UnsupportedPageMode { mode, context } => {
                write!(f, "{context} does not support page mode {mode} yet")
            }
            Self::FusedAttentionRequiresExactPages {
                page_id,
                key_mode,
                value_mode,
            } => {
                write!(
                    f,
                    "fused attention requires exact pages, but page {page_id} uses key_mode={key_mode} value_mode={value_mode}"
                )
            }
            Self::MissingAsset { model_id, filename } => {
                write!(f, "model {model_id} is missing required asset {filename}")
            }
            Self::UnsupportedModelFamily { family } => {
                write!(f, "unsupported model family {family}")
            }
            Self::BackendUnavailable { backend, device } => {
                write!(f, "backend {backend} is unavailable on device {device}")
            }
            Self::External { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<std::io::Error> for RuntimeError {
    fn from(err: std::io::Error) -> Self {
        Self::External {
            context: "io",
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

#[cfg(feature = "hf")]
impl From<serde_json::Error> for RuntimeError {
    fn from(err: serde_json::Error) -> Self {
        Self::External {
            context: "serde_json",
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "hf")]
impl From<tokenizers::Error> for RuntimeError {
    fn from(err: tokenizers::Error) -> Self {
        Self::External {
            context: "tokenizers",
            message: err.to_string(),
        }
    }
}

#[cfg(any(feature = "candle", feature = "qwen35-minimal"))]
impl From<candle_core::Error> for RuntimeError {
    fn from(err: candle_core::Error) -> Self {
        Self::External {
            context: "candle",
            message: err.to_string(),
        }
    }
}
