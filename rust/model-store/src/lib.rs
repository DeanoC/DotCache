use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use candle_core::{op::BackpropOp, Device, Tensor, WithDType};
pub use dotcache_runtime_core::{
    BackendKind, BufferMutability, BufferViewDesc, ImmutableBufferView, ScalarType, TargetSpec,
};
use hf_hub::api::sync::{Api, ApiBuilder};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PACKAGE_ALIGNMENT: u64 = 4096;
const MANIFEST_FILENAME: &str = "manifest.json";
const WEIGHTS_FILENAME: &str = "weights.bin";
const CONFIG_FILENAME: &str = "config.json";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

pub type Result<T> = std::result::Result<T, ModelStoreError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStoreError {
    MissingAsset {
        model_id: String,
        filename: String,
    },
    UnsupportedBackend {
        backend: String,
    },
    UnsupportedModelFamily {
        family: String,
    },
    UnsupportedSourceFormat {
        message: String,
    },
    External {
        context: &'static str,
        message: String,
    },
}

impl std::fmt::Display for ModelStoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAsset { model_id, filename } => {
                write!(f, "missing asset `{filename}` for model `{model_id}`")
            }
            Self::UnsupportedBackend { backend } => {
                write!(f, "unsupported backend `{backend}` for model-store v1")
            }
            Self::UnsupportedModelFamily { family } => {
                write!(f, "unsupported model family `{family}`")
            }
            Self::UnsupportedSourceFormat { message } => f.write_str(message),
            Self::External { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl std::error::Error for ModelStoreError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageKey {
    pub model_family: String,
    pub model_id: String,
    pub revision: String,
    pub target: TargetSpec,
    pub schema_version: u32,
    pub converter_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTensorEncoding {
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayoutTag {
    StandardContiguous,
    DepthwiseConvSqueezed,
    HeadBiasReshaped,
    HeadExpReshaped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PreparedDType {
    U8,
    U32,
    I16,
    I32,
    I64,
    BF16,
    F16,
    F32,
}

impl PreparedDType {
    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::U8 => Ok(Self::U8),
            safetensors::Dtype::U32 => Ok(Self::U32),
            safetensors::Dtype::I16 => Ok(Self::I16),
            safetensors::Dtype::I32 => Ok(Self::I32),
            safetensors::Dtype::I64 => Ok(Self::I64),
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F16 => Ok(Self::F16),
            safetensors::Dtype::F32 => Ok(Self::F32),
            other => Err(ModelStoreError::UnsupportedSourceFormat {
                message: format!(
                    "unsupported safetensors dtype {other:?} in model-store v1 conversion"
                ),
            }),
        }
    }

    fn scalar_type(self) -> ScalarType {
        match self {
            Self::U8 => ScalarType::U8,
            Self::U32 => ScalarType::U32,
            Self::I16 => ScalarType::I16,
            Self::I32 => ScalarType::I32,
            Self::I64 => ScalarType::I64,
            Self::BF16 => ScalarType::BF16,
            Self::F16 => ScalarType::F16,
            Self::F32 => ScalarType::F32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: PreparedDType,
    pub encoding: PreparedTensorEncoding,
    pub layout: TensorLayoutTag,
    pub blob: String,
    pub offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPackageManifest {
    pub schema_version: u32,
    pub converter_version: u32,
    pub model_family: String,
    pub model_id: String,
    pub revision: String,
    pub target_backend: String,
    pub target_family: String,
    pub config_filename: String,
    pub tokenizer_filename: String,
    pub tensors: Vec<PreparedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PreparedPackageAlias {
    revision: String,
    package_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPackageStats {
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub weights_blob_bytes: u64,
    pub total_package_bytes: u64,
    pub standard_tensor_count: usize,
    pub standard_bytes: u64,
    pub prepacked_tensor_count: usize,
    pub prepacked_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorLoadStatEntry {
    pub name: String,
    pub calls: u64,
    pub bytes: u64,
    pub millis: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightLoadStats {
    pub tensor_get_calls: u64,
    pub unique_tensors: usize,
    pub tensor_bytes: u64,
    pub tensor_load_millis: f64,
    pub top_by_bytes: Vec<TensorLoadStatEntry>,
    pub top_by_millis: Vec<TensorLoadStatEntry>,
}

#[derive(Debug, Clone)]
pub struct WeightView<'a> {
    entry: &'a PreparedTensorEntry,
    bytes: &'a [u8],
}

impl<'a> WeightView<'a> {
    pub fn name(&self) -> &str {
        &self.entry.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    pub fn dtype(&self) -> PreparedDType {
        self.entry.dtype
    }

    pub fn layout(&self) -> &TensorLayoutTag {
        &self.entry.layout
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn buffer_view_desc(&self) -> BufferViewDesc {
        BufferViewDesc {
            scalar_type: self.entry.dtype.scalar_type(),
            shape: self.entry.shape.clone(),
            byte_offset: self.entry.offset,
            byte_len: self.entry.byte_len,
            mutability: BufferMutability::Immutable,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImmutableWeightHandle {
    package: Arc<PreparedPackage>,
    tensor_idx: usize,
}

impl ImmutableWeightHandle {
    fn entry(&self) -> &PreparedTensorEntry {
        &self.package.manifest.tensors[self.tensor_idx]
    }

    pub fn name(&self) -> &str {
        &self.entry().name
    }

    pub fn shape(&self) -> &[usize] {
        &self.entry().shape
    }

    pub fn dtype(&self) -> PreparedDType {
        self.entry().dtype
    }

    pub fn layout(&self) -> &TensorLayoutTag {
        &self.entry().layout
    }

    pub fn backend(&self) -> &str {
        &self.package.manifest.target_backend
    }

    pub fn family(&self) -> &str {
        &self.package.manifest.target_family
    }

    pub fn target_spec(&self) -> TargetSpec {
        TargetSpec {
            backend: match self.package.manifest.target_backend.as_str() {
                "cpu" => BackendKind::Cpu,
                "hip" => BackendKind::Hip,
                "cuda" => BackendKind::Cuda,
                "metal" => BackendKind::Metal,
                other => panic!("unknown prepared package backend `{other}`"),
            },
            family: self.package.manifest.target_family.clone(),
        }
    }

    pub fn offset(&self) -> u64 {
        self.entry().offset
    }

    pub fn byte_len(&self) -> u64 {
        self.entry().byte_len
    }

    pub fn bytes(&self) -> &[u8] {
        let entry = self.entry();
        let start = usize::try_from(entry.offset).expect("validated package offset fits usize");
        let byte_len =
            usize::try_from(entry.byte_len).expect("validated package byte_len fits usize");
        &self.package.weights[start..start + byte_len]
    }

    pub fn buffer_view_desc(&self) -> BufferViewDesc {
        BufferViewDesc {
            scalar_type: self.entry().dtype.scalar_type(),
            shape: self.entry().shape.clone(),
            byte_offset: self.entry().offset,
            byte_len: self.entry().byte_len,
            mutability: BufferMutability::Immutable,
        }
    }

    pub fn immutable_buffer_view(&self) -> ImmutableBufferView<'_> {
        ImmutableBufferView {
            target: self.target_spec(),
            desc: self.buffer_view_desc(),
            bytes: self.bytes(),
        }
    }

    pub fn materialize(&self, device: &Device) -> Result<Tensor> {
        self.package.load_tensor(self.name(), device)
    }
}

pub trait WeightProvider: Clone {
    fn device(&self) -> &Device;
    fn pp<T: Display>(&self, component: T) -> Self;
    fn get(&self, name: &str) -> candle_core::Result<Tensor>;
    fn get_immutable(&self, name: &str) -> candle_core::Result<Option<ImmutableWeightHandle>>;
    fn contains_tensor(&self, name: &str) -> bool;
}

pub trait ModelFamilyConverter {
    fn model_family(&self) -> &'static str;
    fn converter_version(&self) -> u32;
    fn build_package(
        &self,
        artifacts: &HfModelArtifacts,
        target: &TargetSpec,
        package_root: &Path,
    ) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct PreparedPackage {
    root: PathBuf,
    manifest: PreparedPackageManifest,
    weights: Arc<Mmap>,
    tensor_index: Arc<BTreeMap<String, usize>>,
}

impl PreparedPackage {
    pub fn resolve_or_build<C: ModelFamilyConverter>(
        model_id: &str,
        device: &Device,
        converter: &C,
    ) -> Result<Self> {
        let target = detect_target_spec(device)?;
        if let Some(alias) = read_alias(converter.model_family(), model_id, &target)? {
            if alias.package_root.exists() {
                let package = Self::open(&alias.package_root)?;
                if package.manifest.schema_version == PACKAGE_SCHEMA_VERSION
                    && package.manifest.converter_version == converter.converter_version()
                {
                    return Ok(package);
                }
            }
        }

        let source = HfHubModelSource::new()?;
        let artifacts = source.snapshot(model_id)?;
        let key = PackageKey {
            model_family: converter.model_family().to_string(),
            model_id: artifacts.model_id.clone(),
            revision: artifacts.revision.clone(),
            target: target.clone(),
            schema_version: PACKAGE_SCHEMA_VERSION,
            converter_version: converter.converter_version(),
        };
        let package_root = package_root(&key)?;
        if !package_root.exists() {
            build_package_with_lock(&package_root, || {
                converter.build_package(&artifacts, &target, &package_root)
            })?;
        }
        write_alias(
            converter.model_family(),
            &artifacts.model_id,
            &target,
            &PreparedPackageAlias {
                revision: artifacts.revision.clone(),
                package_root: package_root.clone(),
            },
        )?;
        Self::open(&package_root)
    }

    pub fn resolve_or_build_qwen35_minimal(model_id: &str, device: &Device) -> Result<Self> {
        Self::resolve_or_build(model_id, device, &Qwen35MinimalConverter)
    }

    pub fn open(root: &Path) -> Result<Self> {
        let manifest_path = root.join(MANIFEST_FILENAME);
        let manifest: PreparedPackageManifest =
            serde_json::from_slice(&fs::read(&manifest_path).map_err(|err| {
                ModelStoreError::External {
                    context: "model-store",
                    message: format!("failed to read {}: {err}", manifest_path.display()),
                }
            })?)
            .map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: format!("failed to parse {}: {err}", manifest_path.display()),
            })?;
        let weights_file =
            File::open(root.join(WEIGHTS_FILENAME)).map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: format!(
                    "failed to open {}: {err}",
                    root.join(WEIGHTS_FILENAME).display()
                ),
            })?;
        let weights =
            unsafe { Mmap::map(&weights_file) }.map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: format!("failed to mmap weights blob: {err}"),
            })?;
        let tensor_index = manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.name.clone(), idx))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            weights: Arc::new(weights),
            tensor_index: Arc::new(tensor_index),
        })
    }

    pub fn manifest(&self) -> &PreparedPackageManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.root.join(&self.manifest.tokenizer_filename)
    }

    pub fn config_path(&self) -> PathBuf {
        self.root.join(&self.manifest.config_filename)
    }

    pub fn stats(&self) -> Result<PreparedPackageStats> {
        let mut payload_bytes = 0u64;
        let mut standard_tensor_count = 0usize;
        let mut standard_bytes = 0u64;
        let mut prepacked_tensor_count = 0usize;
        let mut prepacked_bytes = 0u64;

        for entry in &self.manifest.tensors {
            payload_bytes += entry.byte_len;
            match entry.layout {
                TensorLayoutTag::StandardContiguous => {
                    standard_tensor_count += 1;
                    standard_bytes += entry.byte_len;
                }
                _ => {
                    prepacked_tensor_count += 1;
                    prepacked_bytes += entry.byte_len;
                }
            }
        }

        let weights_blob_bytes = fs::metadata(self.root.join(WEIGHTS_FILENAME))
            .map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: format!("failed to stat weights blob: {err}"),
            })?
            .len();
        let total_package_bytes = weights_blob_bytes
            + fs::metadata(self.root.join(MANIFEST_FILENAME))
                .map_err(|err| ModelStoreError::External {
                    context: "model-store",
                    message: format!("failed to stat manifest: {err}"),
                })?
                .len()
            + fs::metadata(self.root.join(CONFIG_FILENAME))
                .map_err(|err| ModelStoreError::External {
                    context: "model-store",
                    message: format!("failed to stat config: {err}"),
                })?
                .len()
            + fs::metadata(self.root.join(TOKENIZER_FILENAME))
                .map_err(|err| ModelStoreError::External {
                    context: "model-store",
                    message: format!("failed to stat tokenizer: {err}"),
                })?
                .len();

        Ok(PreparedPackageStats {
            tensor_count: self.manifest.tensors.len(),
            payload_bytes,
            weights_blob_bytes,
            total_package_bytes,
            standard_tensor_count,
            standard_bytes,
            prepacked_tensor_count,
            prepacked_bytes,
        })
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    pub fn immutable_handle(&self, name: &str) -> Result<ImmutableWeightHandle> {
        let tensor_idx = *self.tensor_index.get(name).ok_or_else(|| {
            ModelStoreError::External {
                context: "model-store",
                message: format!("missing tensor {name} in prepared package"),
            }
        })?;
        Ok(ImmutableWeightHandle {
            package: Arc::new(self.clone()),
            tensor_idx,
        })
    }

    pub fn weight_view(&self, name: &str) -> Result<WeightView<'_>> {
        let entry = self.tensor_entry(name)?;
        let start = usize::try_from(entry.offset).map_err(|_| ModelStoreError::External {
            context: "model-store",
            message: format!("tensor {} offset {} does not fit usize", name, entry.offset),
        })?;
        let byte_len = usize::try_from(entry.byte_len).map_err(|_| ModelStoreError::External {
            context: "model-store",
            message: format!(
                "tensor {} length {} does not fit usize",
                name, entry.byte_len
            ),
        })?;
        let end = start + byte_len;
        if end > self.weights.len() {
            return Err(ModelStoreError::External {
                context: "model-store",
                message: format!(
                    "tensor {} extends past weights blob (end={} blob={})",
                    name,
                    end,
                    self.weights.len()
                ),
            });
        }
        Ok(WeightView {
            entry,
            bytes: &self.weights[start..end],
        })
    }

    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        Ok(self.load_tensor_with_byte_len(name, device)?.0)
    }

    pub fn load_tensor_with_byte_len(&self, name: &str, device: &Device) -> Result<(Tensor, u64)> {
        let view = self.weight_view(name)?;
        let start = usize::try_from(view.entry.offset).map_err(|_| ModelStoreError::External {
            context: "model-store",
            message: format!(
                "tensor {} offset {} does not fit usize",
                name, view.entry.offset
            ),
        })?;
        let byte_len =
            usize::try_from(view.entry.byte_len).map_err(|_| ModelStoreError::External {
                context: "model-store",
                message: format!(
                    "tensor {} length {} does not fit usize",
                    name, view.entry.byte_len
                ),
            })?;
        let tensor =
            load_tensor_from_prepared_bytes(view.bytes(), view.dtype(), view.shape(), device)?;
        release_mmap_range(&self.weights, start, byte_len);
        Ok((tensor, view.entry.byte_len))
    }

    fn tensor_entry(&self, name: &str) -> Result<&PreparedTensorEntry> {
        self.tensor_index
            .get(name)
            .and_then(|idx| self.manifest.tensors.get(*idx))
            .ok_or_else(|| ModelStoreError::External {
                context: "model-store",
                message: format!("missing tensor {name} in prepared package"),
            })
    }
}

#[derive(Debug, Clone)]
pub struct CandleWeightProvider {
    package: Arc<PreparedPackage>,
    device: Device,
    prefix: String,
    load_stats: Option<Arc<Mutex<BTreeMap<String, TensorLoadAccumulator>>>>,
}

impl CandleWeightProvider {
    pub fn new(package: Arc<PreparedPackage>, device: Device) -> Self {
        Self::with_stats(package, device, false)
    }

    pub fn new_profiled(package: Arc<PreparedPackage>, device: Device) -> Self {
        Self::with_stats(package, device, true)
    }

    fn with_stats(package: Arc<PreparedPackage>, device: Device, collect_stats: bool) -> Self {
        Self {
            package,
            device,
            prefix: String::new(),
            load_stats: collect_stats.then(|| Arc::new(Mutex::new(BTreeMap::new()))),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn pp<T: Display>(&self, component: T) -> Self {
        let component = component.to_string();
        let prefix = if self.prefix.is_empty() {
            component
        } else {
            format!("{}.{}", self.prefix, component)
        };
        Self {
            package: self.package.clone(),
            device: self.device.clone(),
            prefix,
            load_stats: self.load_stats.clone(),
        }
    }

    pub fn get(&self, name: &str) -> candle_core::Result<Tensor> {
        let full_name = self.full_name(name);
        let started = self
            .load_stats
            .as_ref()
            .map(|_| std::time::Instant::now());
        let (tensor, byte_len) = self
            .package
            .load_tensor_with_byte_len(&full_name, &self.device)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        if let (Some(started), Some(load_stats)) = (started, self.load_stats.as_ref()) {
            let elapsed_millis = started.elapsed().as_secs_f64() * 1000.0;
            let mut stats = load_stats.lock().expect("load stats mutex poisoned");
            let entry = stats.entry(full_name.clone()).or_default();
            entry.calls += 1;
            entry.bytes += byte_len;
            entry.millis += elapsed_millis;
        }
        Ok(tensor)
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.package.contains_tensor(&self.full_name(name))
    }

    pub fn get_immutable(&self, name: &str) -> candle_core::Result<Option<ImmutableWeightHandle>> {
        let full_name = self.full_name(name);
        if !self.device.is_hip()
            || self.package.manifest.target_backend != BackendKind::Hip.as_str()
            || !self.package.contains_tensor(&full_name)
        {
            return Ok(None);
        }
        let immutable_supported = full_name == "model.language_model.embed_tokens.weight"
            || full_name.ends_with(".mlp.gate_proj.weight")
            || full_name.ends_with(".mlp.up_proj.weight")
            || full_name.ends_with(".mlp.down_proj.weight");
        let immutable_supported =
            immutable_supported || full_name.ends_with(".linear_attn.in_proj_qkv.weight");
        if !immutable_supported {
            return Ok(None);
        }
        let handle = self
            .package
            .immutable_handle(&full_name)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        match handle.layout() {
            TensorLayoutTag::StandardContiguous => Ok(Some(handle)),
            _ => Ok(None),
        }
    }

    pub fn load_stats(&self) -> WeightLoadStats {
        let Some(load_stats) = self.load_stats.as_ref() else {
            return WeightLoadStats {
                tensor_get_calls: 0,
                unique_tensors: 0,
                tensor_bytes: 0,
                tensor_load_millis: 0.0,
                top_by_bytes: Vec::new(),
                top_by_millis: Vec::new(),
            };
        };
        let stats = load_stats.lock().expect("load stats mutex poisoned");
        let tensor_get_calls = stats.values().map(|entry| entry.calls).sum();
        let tensor_bytes = stats.values().map(|entry| entry.bytes).sum();
        let tensor_load_millis = stats.values().map(|entry| entry.millis).sum();
        let unique_tensors = stats.len();
        let mut entries = stats
            .iter()
            .map(|(name, entry)| TensorLoadStatEntry {
                name: name.clone(),
                calls: entry.calls,
                bytes: entry.bytes,
                millis: entry.millis,
            })
            .collect::<Vec<_>>();
        let mut top_by_bytes = entries.clone();
        top_by_bytes.sort_by(|lhs, rhs| {
            rhs.bytes
                .cmp(&lhs.bytes)
                .then_with(|| lhs.name.cmp(&rhs.name))
        });
        top_by_bytes.truncate(10);
        entries.sort_by(|lhs, rhs| {
            rhs.millis
                .partial_cmp(&lhs.millis)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs.name.cmp(&rhs.name))
        });
        entries.truncate(10);
        WeightLoadStats {
            tensor_get_calls,
            unique_tensors,
            tensor_bytes,
            tensor_load_millis,
            top_by_bytes,
            top_by_millis: entries,
        }
    }

    fn full_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else if name.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}

impl WeightProvider for CandleWeightProvider {
    fn device(&self) -> &Device {
        self.device()
    }

    fn pp<T: Display>(&self, component: T) -> Self {
        self.pp(component)
    }

    fn get(&self, name: &str) -> candle_core::Result<Tensor> {
        self.get(name)
    }

    fn get_immutable(&self, name: &str) -> candle_core::Result<Option<ImmutableWeightHandle>> {
        self.get_immutable(name)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_tensor(name)
    }
}

pub mod adapters {
    pub mod candle {
        pub use crate::CandleWeightProvider;
    }
}

#[derive(Debug, Clone, Default)]
struct TensorLoadAccumulator {
    calls: u64,
    bytes: u64,
    millis: f64,
}

#[derive(Debug, Clone)]
pub struct HfModelArtifacts {
    pub model_id: String,
    pub revision: String,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HfModelWeightIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

impl HfModelWeightIndex {
    pub fn unique_weight_filenames(&self) -> Vec<String> {
        self.weight_map
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[derive(Debug)]
pub struct HfHubModelSource {
    api: Api,
}

impl HfHubModelSource {
    pub fn new() -> Result<Self> {
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|err| ModelStoreError::External {
                context: "hf-hub",
                message: err.to_string(),
            })?;
        Ok(Self { api })
    }

    pub fn snapshot(&self, model_id: &str) -> Result<HfModelArtifacts> {
        let repo = self.api.model(model_id.to_string());
        let info = repo.info().map_err(|err| ModelStoreError::External {
            context: "hf-hub",
            message: err.to_string(),
        })?;
        let filenames = info
            .siblings
            .iter()
            .map(|entry| entry.rfilename.as_str())
            .collect::<std::collections::BTreeSet<_>>();

        let config_path = repo
            .get("config.json")
            .map_err(|err| ModelStoreError::External {
                context: "hf-hub",
                message: err.to_string(),
            })?;
        let tokenizer_path = if filenames.contains("tokenizer.json") {
            repo.get("tokenizer.json")
                .map_err(|err| ModelStoreError::External {
                    context: "hf-hub",
                    message: err.to_string(),
                })?
        } else {
            return Err(ModelStoreError::MissingAsset {
                model_id: model_id.to_string(),
                filename: "tokenizer.json".to_string(),
            });
        };

        let weight_paths = if filenames.contains("model.safetensors.index.json") {
            let index_path = repo.get("model.safetensors.index.json").map_err(|err| {
                ModelStoreError::External {
                    context: "hf-hub",
                    message: err.to_string(),
                }
            })?;
            let index: HfModelWeightIndex =
                serde_json::from_slice(&fs::read(&index_path).map_err(|err| {
                    ModelStoreError::External {
                        context: "hf-hub",
                        message: err.to_string(),
                    }
                })?)
                .map_err(|err| ModelStoreError::External {
                    context: "hf-hub",
                    message: err.to_string(),
                })?;
            index
                .unique_weight_filenames()
                .into_iter()
                .map(|filename| {
                    repo.get(&filename)
                        .map_err(|err| ModelStoreError::External {
                            context: "hf-hub",
                            message: err.to_string(),
                        })
                })
                .collect::<Result<Vec<_>>>()?
        } else if filenames.contains("model.safetensors") {
            vec![repo
                .get("model.safetensors")
                .map_err(|err| ModelStoreError::External {
                    context: "hf-hub",
                    message: err.to_string(),
                })?]
        } else {
            return Err(ModelStoreError::MissingAsset {
                model_id: model_id.to_string(),
                filename: "model.safetensors or model.safetensors.index.json".to_string(),
            });
        };

        Ok(HfModelArtifacts {
            model_id: model_id.to_string(),
            revision: info.sha,
            config_path,
            tokenizer_path,
            weight_paths,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35MinimalConverter;

impl ModelFamilyConverter for Qwen35MinimalConverter {
    fn model_family(&self) -> &'static str {
        "qwen35-minimal"
    }

    fn converter_version(&self) -> u32 {
        4
    }

    fn build_package(
        &self,
        artifacts: &HfModelArtifacts,
        target: &TargetSpec,
        package_root: &Path,
    ) -> Result<()> {
        build_qwen35_minimal_package(artifacts, target, package_root, self.converter_version())
    }
}

fn build_qwen35_minimal_package(
    artifacts: &HfModelArtifacts,
    target: &TargetSpec,
    package_root: &Path,
    converter_version: u32,
) -> Result<()> {
    let temp_root = temp_package_dir(package_root)?;
    fs::create_dir_all(&temp_root).map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!("failed to create {}: {err}", temp_root.display()),
    })?;

    fs::copy(&artifacts.config_path, temp_root.join(CONFIG_FILENAME)).map_err(|err| {
        ModelStoreError::External {
            context: "model-store",
            message: format!(
                "failed to copy config {} -> {}: {err}",
                artifacts.config_path.display(),
                temp_root.join(CONFIG_FILENAME).display()
            ),
        }
    })?;
    fs::copy(
        &artifacts.tokenizer_path,
        temp_root.join(TOKENIZER_FILENAME),
    )
    .map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!(
            "failed to copy tokenizer {} -> {}: {err}",
            artifacts.tokenizer_path.display(),
            temp_root.join(TOKENIZER_FILENAME).display()
        ),
    })?;

    let mut tensors = Vec::new();
    let weights_path = temp_root.join(WEIGHTS_FILENAME);
    let mut weights_file =
        File::create(&weights_path).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to create {}: {err}", weights_path.display()),
        })?;
    let mut offset = 0u64;

    for weight_path in &artifacts.weight_paths {
        let file = File::open(weight_path).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to open {}: {err}", weight_path.display()),
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to mmap {}: {err}", weight_path.display()),
        })?;
        let tensors_file =
            SafeTensors::deserialize(&mmap).map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: format!("failed to deserialize {}: {err}", weight_path.display()),
            })?;

        for name in tensors_file.names() {
            if !qwen35_minimal_keeps_tensor(name) {
                continue;
            }
            let view = tensors_file
                .tensor(name)
                .map_err(|err| ModelStoreError::External {
                    context: "model-store",
                    message: format!(
                        "failed to read tensor {name} from {}: {err}",
                        weight_path.display()
                    ),
                })?;
            let dtype = PreparedDType::from_safetensors(view.dtype())?;
            let prepacked = maybe_prepack_qwen35_tensor(name, &view, dtype)?;
            if prepacked
                .as_ref()
                .map(|prepared| !prepared.replaces_raw)
                .unwrap_or(true)
            {
                write_tensor_entry(
                    &mut weights_file,
                    &mut offset,
                    &mut tensors,
                    name,
                    view.shape(),
                    dtype,
                    TensorLayoutTag::StandardContiguous,
                    view.data(),
                )?;
            }
            if let Some(prepared) = prepacked {
                write_tensor_entry(
                    &mut weights_file,
                    &mut offset,
                    &mut tensors,
                    &prepared.name,
                    &prepared.shape,
                    dtype,
                    prepared.layout,
                    &prepared.bytes,
                )?;
            }
        }
    }
    weights_file
        .flush()
        .map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to flush weights blob: {err}"),
        })?;

    tensors.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    let manifest = PreparedPackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        converter_version,
        model_family: "qwen35-minimal".to_string(),
        model_id: artifacts.model_id.clone(),
        revision: artifacts.revision.clone(),
        target_backend: target.backend.as_str().to_string(),
        target_family: target.family.clone(),
        config_filename: CONFIG_FILENAME.to_string(),
        tokenizer_filename: TOKENIZER_FILENAME.to_string(),
        tensors,
    };
    fs::write(
        temp_root.join(MANIFEST_FILENAME),
        serde_json::to_vec_pretty(&manifest).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to serialize manifest: {err}"),
        })?,
    )
    .map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!("failed to write manifest: {err}"),
    })?;

    fs::rename(&temp_root, package_root).map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!(
            "failed to publish package {} -> {}: {err}",
            temp_root.display(),
            package_root.display()
        ),
    })?;
    Ok(())
}

fn qwen35_minimal_keeps_tensor(name: &str) -> bool {
    name.starts_with("model.language_model.") || name == "lm_head.weight"
}

fn write_tensor_entry(
    weights_file: &mut File,
    offset: &mut u64,
    tensors: &mut Vec<PreparedTensorEntry>,
    name: &str,
    shape: &[usize],
    dtype: PreparedDType,
    layout: TensorLayoutTag,
    data: &[u8],
) -> Result<()> {
    *offset = align_up(*offset, PACKAGE_ALIGNMENT);
    weights_file
        .seek(SeekFrom::Start(*offset))
        .map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to seek weights blob: {err}"),
        })?;
    weights_file
        .write_all(data)
        .map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to write tensor {name}: {err}"),
        })?;
    let byte_len = data.len() as u64;
    tensors.push(PreparedTensorEntry {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype,
        encoding: PreparedTensorEncoding::Plain,
        layout,
        blob: WEIGHTS_FILENAME.to_string(),
        offset: *offset,
        byte_len,
        alignment: PACKAGE_ALIGNMENT,
    });
    *offset += byte_len;
    Ok(())
}

struct PrepackedTensor {
    name: String,
    shape: Vec<usize>,
    layout: TensorLayoutTag,
    bytes: Vec<u8>,
    replaces_raw: bool,
}

fn maybe_prepack_qwen35_tensor(
    name: &str,
    view: &safetensors::tensor::TensorView<'_>,
    dtype: PreparedDType,
) -> Result<Option<PrepackedTensor>> {
    if name.ends_with("conv1d.weight") {
        let shape = view.shape();
        if shape.len() == 3 && shape[1] == 1 {
            let prepared_name = format!("{name}.__dotcache_depthwise_squeezed");
            return Ok(Some(PrepackedTensor {
                name: prepared_name,
                shape: vec![shape[0], shape[2]],
                layout: TensorLayoutTag::DepthwiseConvSqueezed,
                bytes: view.data().to_vec(),
                replaces_raw: true,
            }));
        }
    }

    if name.ends_with("dt_bias") {
        let prepared_name = format!("{name}.__dotcache_head_bias_reshaped");
        return Ok(Some(PrepackedTensor {
            name: prepared_name,
            shape: vec![1, 1, view.shape()[0]],
            layout: TensorLayoutTag::HeadBiasReshaped,
            bytes: view.data().to_vec(),
            replaces_raw: true,
        }));
    }

    if name.ends_with("A_log") {
        let prepared_name = format!("{name}.__dotcache_head_exp_reshaped");
        let prepared_bytes = match dtype {
            PreparedDType::F16 => {
                let src = cast_bytes::<half::f16>(view.data())?;
                src.iter()
                    .map(|v| half::f16::from_f32(v.to_f32().exp()))
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            }
            PreparedDType::BF16 => {
                let src = cast_bytes::<half::bf16>(view.data())?;
                src.iter()
                    .map(|v| half::bf16::from_f32(v.to_f32().exp()))
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            }
            PreparedDType::F32 => {
                let src = cast_bytes::<f32>(view.data())?;
                src.iter()
                    .map(|v| v.exp())
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            }
            _ => return Ok(None),
        };
        return Ok(Some(PrepackedTensor {
            name: prepared_name,
            shape: vec![1, 1, view.shape()[0]],
            layout: TensorLayoutTag::HeadExpReshaped,
            bytes: prepared_bytes,
            replaces_raw: true,
        }));
    }

    Ok(None)
}

fn load_tensor_from_prepared_bytes(
    data: &[u8],
    dtype: PreparedDType,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    match dtype {
        PreparedDType::U8 => load_typed_tensor::<u8>(data, shape, device),
        PreparedDType::U32 => load_typed_tensor::<u32>(data, shape, device),
        PreparedDType::I16 => load_typed_tensor::<i16>(data, shape, device),
        PreparedDType::I32 => load_typed_tensor::<i32>(data, shape, device),
        PreparedDType::I64 => load_typed_tensor::<i64>(data, shape, device),
        PreparedDType::BF16 => load_typed_tensor::<half::bf16>(data, shape, device),
        PreparedDType::F16 => load_typed_tensor::<half::f16>(data, shape, device),
        PreparedDType::F32 => load_typed_tensor::<f32>(data, shape, device),
    }
}

fn load_typed_tensor<T: WithDType>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let elem_size = std::mem::size_of::<T>();
    if !data.len().is_multiple_of(elem_size) {
        return Err(ModelStoreError::External {
            context: "model-store",
            message: format!(
                "tensor byte length {} is not a multiple of element size {}",
                data.len(),
                elem_size
            ),
        });
    }
    let elem_count = data.len() / elem_size;
    let expected_elem_count = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| ModelStoreError::External {
            context: "model-store",
            message: format!("tensor shape {:?} overflows element count", shape),
        })?;
    if elem_count != expected_elem_count {
        return Err(ModelStoreError::External {
            context: "model-store",
            message: format!(
                "tensor element count mismatch: bytes imply {} elems but shape {:?} implies {}",
                elem_count, shape, expected_elem_count
            ),
        });
    }
    let storage = if (data.as_ptr() as usize).is_multiple_of(elem_size) {
        let typed = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        device
            .storage_from_slice(typed)
            .map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: err.to_string(),
            })?
    } else {
        let mut owned = Vec::<T>::with_capacity(elem_count);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), owned.as_mut_ptr() as *mut u8, data.len());
            owned.set_len(elem_count);
        }
        device
            .storage_owned(owned)
            .map_err(|err| ModelStoreError::External {
                context: "model-store",
                message: err.to_string(),
            })?
    };
    Ok(Tensor::from_storage(
        storage,
        shape,
        BackpropOp::none(),
        false,
    ))
}

fn cast_bytes<T: Copy>(data: &[u8]) -> Result<&[T]> {
    let (head, body, tail) = unsafe { data.align_to::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(ModelStoreError::External {
            context: "model-store",
            message: "unaligned tensor bytes while preparing package".to_string(),
        });
    }
    Ok(body)
}

fn build_package_with_lock<F>(package_root: &Path, build: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    if package_root.exists() {
        return Ok(());
    }
    if let Some(parent) = package_root.parent() {
        fs::create_dir_all(parent).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to create {}: {err}", parent.display()),
        })?;
    }
    let lock_path = package_root.with_extension("lock");
    match OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(lock_file) => {
            let _guard = PackageBuildLock {
                path: lock_path.clone(),
                _file: lock_file,
            };
            if !package_root.exists() {
                build()?;
            }
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            for _ in 0..300 {
                if package_root.exists() {
                    return Ok(());
                }
                if !lock_path.exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(200));
            }
            if package_root.exists() {
                Ok(())
            } else {
                Err(ModelStoreError::External {
                    context: "model-store",
                    message: format!(
                        "timed out waiting for package build lock {}",
                        lock_path.display()
                    ),
                })
            }
        }
        Err(err) => Err(ModelStoreError::External {
            context: "model-store",
            message: format!("failed to create build lock {}: {err}", lock_path.display()),
        }),
    }
}

struct PackageBuildLock {
    path: PathBuf,
    _file: File,
}

impl Drop for PackageBuildLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn package_root(key: &PackageKey) -> Result<PathBuf> {
    Ok(package_cache_root()?
        .join(&key.model_family)
        .join(sanitize_path_component(&key.model_id))
        .join(sanitize_path_component(&key.revision))
        .join(format!("converter-v{}", key.converter_version))
        .join(key.target.backend.as_str())
        .join(&key.target.family))
}

fn package_alias_path(model_family: &str, model_id: &str, target: &TargetSpec) -> Result<PathBuf> {
    Ok(package_cache_root()?
        .join(model_family)
        .join(sanitize_path_component(model_id))
        .join("aliases")
        .join(target.backend.as_str())
        .join(&target.family)
        .join("active.json"))
}

fn temp_package_dir(package_root: &Path) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("system time error: {err}"),
        })?
        .as_nanos();
    Ok(package_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".tmp-{}-{}", std::process::id(), nanos)))
}

fn package_cache_root() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| ModelStoreError::External {
        context: "model-store",
        message: "HOME is not set".to_string(),
    })?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("dotcache")
        .join("model-packages"))
}

fn read_alias(
    model_family: &str,
    model_id: &str,
    target: &TargetSpec,
) -> Result<Option<PreparedPackageAlias>> {
    let path = package_alias_path(model_family, model_id, target)?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!("failed to read alias {}: {err}", path.display()),
    })?;
    let alias = serde_json::from_slice(&bytes).map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!("failed to parse alias {}: {err}", path.display()),
    })?;
    Ok(Some(alias))
}

fn write_alias(
    model_family: &str,
    model_id: &str,
    target: &TargetSpec,
    alias: &PreparedPackageAlias,
) -> Result<()> {
    let path = package_alias_path(model_family, model_id, target)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to create {}: {err}", parent.display()),
        })?;
    }
    fs::write(
        &path,
        serde_json::to_vec_pretty(alias).map_err(|err| ModelStoreError::External {
            context: "model-store",
            message: format!("failed to serialize alias {}: {err}", path.display()),
        })?,
    )
    .map_err(|err| ModelStoreError::External {
        context: "model-store",
        message: format!("failed to write alias {}: {err}", path.display()),
    })?;
    Ok(())
}

fn sanitize_path_component(value: &str) -> OsString {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-' => encoded.push(*byte as char),
            _ => encoded.push_str(&format!("~{:02X}", byte)),
        }
    }
    OsString::from(encoded)
}

fn detect_target_spec(device: &Device) -> Result<TargetSpec> {
    let target = TargetSpec::detect(device);
    if matches!(target.backend, BackendKind::Metal) {
        return Err(ModelStoreError::UnsupportedBackend {
            backend: "metal".to_string(),
        });
    }
    Ok(target)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn release_mmap_range(mmap: &Mmap, start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return;
    }
    let page_size = page_size as usize;
    let page_start = start / page_size * page_size;
    let page_end = (start + len).div_ceil(page_size) * page_size;
    let page_end = page_end.min(mmap.len());
    if page_end <= page_start {
        return;
    }
    let ptr = unsafe { mmap.as_ptr().add(page_start) } as *mut libc::c_void;
    let span = page_end - page_start;
    unsafe {
        libc::madvise(ptr, span, libc::MADV_DONTNEED);
    }
}

pub type PreparedModelPackage = PreparedPackage;
pub type PreparedModelManifest = PreparedPackageManifest;
pub type PreparedPackageSummary = PreparedPackageStats;
pub type PreparedTensorLayout = TensorLayoutTag;
pub type ModelTarget = TargetSpec;

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{tensor::TensorView, Dtype};

    #[test]
    fn sanitize_path_component_replaces_path_separators() {
        assert_eq!(
            sanitize_path_component("Qwen/Qwen3.5-4B"),
            OsString::from("Qwen~2FQwen3.5-4B")
        );
    }

    #[test]
    fn sanitize_path_component_preserves_model_id_uniqueness() {
        assert_ne!(
            sanitize_path_component("foo/bar"),
            sanitize_path_component("foo-bar")
        );
    }

    #[test]
    fn align_up_rounds_to_alignment() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    #[test]
    fn load_typed_tensor_rejects_misaligned_byte_lengths() {
        let err = load_typed_tensor::<u32>(&[1, 2, 3], &[1], &Device::Cpu).unwrap_err();
        assert!(format!("{err}").contains("not a multiple of element size"));
    }

    #[test]
    fn qwen35_conv1d_prepack_replaces_raw_tensor() {
        let view = TensorView::new(
            Dtype::F16,
            vec![8, 1, 4],
            &[0u8; 8 * 1 * 4 * std::mem::size_of::<half::f16>()],
        )
        .unwrap();
        let prepared =
            maybe_prepack_qwen35_tensor("layer.linear_attn.conv1d.weight", &view, PreparedDType::F16)
                .unwrap()
                .expect("conv1d weight should be prepacked");
        assert!(prepared.replaces_raw);
        assert_eq!(prepared.layout, TensorLayoutTag::DepthwiseConvSqueezed);
        assert_eq!(prepared.shape, vec![8, 4]);
    }

    #[test]
    fn qwen35_tensor_filter_keeps_only_minimal_runtime_weights() {
        assert!(qwen35_minimal_keeps_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(qwen35_minimal_keeps_tensor("model.language_model.embed_tokens.weight"));
        assert!(qwen35_minimal_keeps_tensor("lm_head.weight"));
        assert!(!qwen35_minimal_keeps_tensor("model.visual.patch_embed.proj.weight"));
        assert!(!qwen35_minimal_keeps_tensor("mtp.layers.0.weight"));
    }
}
