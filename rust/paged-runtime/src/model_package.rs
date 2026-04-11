use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use candle_core::{op::BackpropOp, Device, DeviceLocation, Tensor, WithDType};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::{HfHubModelSource, Result, RuntimeError};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PACKAGE_CONVERTER_VERSION: u32 = 2;
const PACKAGE_ALIGNMENT: u64 = 4096;
const MANIFEST_FILENAME: &str = "manifest.json";
const WEIGHTS_FILENAME: &str = "weights.bin";
const CONFIG_FILENAME: &str = "config.json";
const TOKENIZER_FILENAME: &str = "tokenizer.json";
const MODEL_FAMILY_QWEN35_MINIMAL: &str = "qwen35-minimal";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTarget {
    pub backend: String,
    pub family: String,
}

impl ModelTarget {
    pub fn detect(device: &Device) -> Self {
        if let Ok(override_family) = std::env::var("DOTCACHE_MODEL_PACKAGE_FAMILY") {
            let backend = match device.location() {
                DeviceLocation::Cpu => "cpu",
                DeviceLocation::Cuda { .. } => "cuda",
                DeviceLocation::Hip { .. } => "hip",
                DeviceLocation::Metal { .. } => "metal",
            };
            return Self {
                backend: backend.to_string(),
                family: override_family,
            };
        }

        match device.location() {
            DeviceLocation::Cpu => Self {
                backend: "cpu".to_string(),
                family: "host".to_string(),
            },
            DeviceLocation::Hip { .. } => Self {
                backend: "hip".to_string(),
                family: detect_hip_family().unwrap_or_else(|| "hip-generic".to_string()),
            },
            DeviceLocation::Cuda { .. } => Self {
                backend: "cuda".to_string(),
                family: detect_cuda_family().unwrap_or_else(|| "cuda-generic".to_string()),
            },
            DeviceLocation::Metal { .. } => Self {
                backend: "metal".to_string(),
                family: "metal-generic".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTensorEncoding {
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTensorLayout {
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
            other => Err(RuntimeError::External {
                context: "prepared-model-package",
                message: format!(
                    "unsupported safetensors dtype {other:?} in v1 prepared package conversion"
                ),
            }),
        }
    }

}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: PreparedDType,
    pub encoding: PreparedTensorEncoding,
    pub layout: PreparedTensorLayout,
    pub blob: String,
    pub offset: u64,
    pub byte_len: u64,
    pub alignment: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedModelManifest {
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

#[derive(Debug, Clone)]
pub struct PreparedModelPackage {
    root: PathBuf,
    manifest: PreparedModelManifest,
    weights: Arc<Mmap>,
    tensor_index: Arc<BTreeMap<String, usize>>,
}

impl PreparedModelPackage {
    pub fn resolve_or_build_qwen35_minimal(model_id: &str, device: &Device) -> Result<Self> {
        let target = ModelTarget::detect(device);
        if let Some(alias) = read_alias(MODEL_FAMILY_QWEN35_MINIMAL, model_id, &target)? {
            if alias.package_root.exists() {
                let package = Self::open(&alias.package_root)?;
                if package.manifest.schema_version == PACKAGE_SCHEMA_VERSION
                    && package.manifest.converter_version == PACKAGE_CONVERTER_VERSION
                {
                    return Ok(package);
                }
            }
        }

        let source = HfHubModelSource::new()?;
        let artifacts = source.snapshot(model_id)?;
        let package_root = package_root(
            MODEL_FAMILY_QWEN35_MINIMAL,
            &artifacts.model_id,
            &artifacts.revision,
            &target,
        )?;

        if !package_root.exists() {
            build_package_with_lock(&package_root, || {
                build_qwen35_minimal_package(&artifacts, &target, &package_root)
            })?;
        }

        write_alias(
            MODEL_FAMILY_QWEN35_MINIMAL,
            &artifacts.model_id,
            &target,
            &PreparedPackageAlias {
                revision: artifacts.revision.clone(),
                package_root: package_root.clone(),
            },
        )?;
        Self::open(&package_root)
    }

    pub fn open(root: &Path) -> Result<Self> {
        let manifest_path = root.join(MANIFEST_FILENAME);
        let manifest: PreparedModelManifest =
            serde_json::from_slice(&fs::read(&manifest_path).map_err(|err| RuntimeError::External {
                context: "prepared-model-package",
                message: format!("failed to read {}: {err}", manifest_path.display()),
            })?)?;
        let weights_file = File::open(root.join(WEIGHTS_FILENAME)).map_err(|err| {
            RuntimeError::External {
                context: "prepared-model-package",
                message: format!(
                    "failed to open {}: {err}",
                    root.join(WEIGHTS_FILENAME).display()
                ),
            }
        })?;
        let weights = unsafe { Mmap::map(&weights_file) }.map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
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

    pub fn manifest(&self) -> &PreparedModelManifest {
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

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let entry = self.tensor_entry(name)?;
        let start = usize::try_from(entry.offset).map_err(|_| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("tensor {} offset {} does not fit usize", name, entry.offset),
        })?;
        let byte_len = usize::try_from(entry.byte_len).map_err(|_| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("tensor {} length {} does not fit usize", name, entry.byte_len),
        })?;
        let end = start + byte_len;
        if end > self.weights.len() {
            return Err(RuntimeError::External {
                context: "prepared-model-package",
                message: format!(
                    "tensor {} extends past weights blob (end={} blob={})",
                    name,
                    end,
                    self.weights.len()
                ),
            });
        }
        let data = &self.weights[start..end];
        let tensor = load_tensor_from_prepared_bytes(data, entry.dtype, &entry.shape, device)?;
        release_mmap_range(&self.weights, start, byte_len);
        Ok(tensor)
    }

    fn tensor_entry(&self, name: &str) -> Result<&PreparedTensorEntry> {
        self.tensor_index
            .get(name)
            .and_then(|idx| self.manifest.tensors.get(*idx))
            .ok_or_else(|| RuntimeError::External {
                context: "prepared-model-package",
                message: format!("missing tensor {name} in prepared package"),
            })
    }
}

fn build_qwen35_minimal_package(
    artifacts: &crate::HfModelArtifacts,
    target: &ModelTarget,
    package_root: &Path,
) -> Result<()> {
    let temp_root = temp_package_dir(package_root)?;
    fs::create_dir_all(&temp_root).map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to create {}: {err}", temp_root.display()),
    })?;

    fs::copy(&artifacts.config_path, temp_root.join(CONFIG_FILENAME)).map_err(|err| {
        RuntimeError::External {
            context: "prepared-model-package",
            message: format!(
                "failed to copy config {} -> {}: {err}",
                artifacts.config_path.display(),
                temp_root.join(CONFIG_FILENAME).display()
            ),
        }
    })?;
    fs::copy(&artifacts.tokenizer_path, temp_root.join(TOKENIZER_FILENAME)).map_err(|err| {
        RuntimeError::External {
            context: "prepared-model-package",
            message: format!(
                "failed to copy tokenizer {} -> {}: {err}",
                artifacts.tokenizer_path.display(),
                temp_root.join(TOKENIZER_FILENAME).display()
            ),
        }
    })?;

    let mut tensors = Vec::new();
    let weights_path = temp_root.join(WEIGHTS_FILENAME);
    let mut weights_file = File::create(&weights_path).map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to create {}: {err}", weights_path.display()),
    })?;
    let mut offset = 0u64;

    for weight_path in &artifacts.weight_paths {
        let file = File::open(weight_path).map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to open {}: {err}", weight_path.display()),
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to mmap {}: {err}", weight_path.display()),
        })?;
        let tensors_file =
            SafeTensors::deserialize(&mmap).map_err(|err| RuntimeError::External {
                context: "prepared-model-package",
                message: format!("failed to deserialize {}: {err}", weight_path.display()),
            })?;

        for name in tensors_file.names() {
            let view = tensors_file.tensor(name).map_err(|err| RuntimeError::External {
                context: "prepared-model-package",
                message: format!("failed to read tensor {name} from {}: {err}", weight_path.display()),
            })?;
            let dtype = PreparedDType::from_safetensors(view.dtype())?;
            write_tensor_entry(
                &mut weights_file,
                &mut offset,
                &mut tensors,
                name,
                view.shape(),
                dtype,
                PreparedTensorLayout::StandardContiguous,
                view.data(),
            )?;

            if let Some((prepared_name, prepared_shape, prepared_layout, prepared_bytes)) =
                maybe_prepack_tensor(name, &view, dtype)?
            {
                write_tensor_entry(
                    &mut weights_file,
                    &mut offset,
                    &mut tensors,
                    &prepared_name,
                    &prepared_shape,
                    dtype,
                    prepared_layout,
                    &prepared_bytes,
                )?;
            }
        }
    }
    weights_file.flush().map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to flush weights blob: {err}"),
    })?;

    tensors.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    let manifest = PreparedModelManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        converter_version: PACKAGE_CONVERTER_VERSION,
        model_family: MODEL_FAMILY_QWEN35_MINIMAL.to_string(),
        model_id: artifacts.model_id.clone(),
        revision: artifacts.revision.clone(),
        target_backend: target.backend.clone(),
        target_family: target.family.clone(),
        config_filename: CONFIG_FILENAME.to_string(),
        tokenizer_filename: TOKENIZER_FILENAME.to_string(),
        tensors,
    };
    fs::write(
        temp_root.join(MANIFEST_FILENAME),
        serde_json::to_vec_pretty(&manifest).map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to serialize manifest: {err}"),
        })?,
    )
    .map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to write manifest: {err}"),
    })?;

    fs::rename(&temp_root, package_root).map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!(
            "failed to publish package {} -> {}: {err}",
            temp_root.display(),
            package_root.display()
        ),
    })?;
    Ok(())
}

fn write_tensor_entry(
    weights_file: &mut File,
    offset: &mut u64,
    tensors: &mut Vec<PreparedTensorEntry>,
    name: &str,
    shape: &[usize],
    dtype: PreparedDType,
    layout: PreparedTensorLayout,
    data: &[u8],
) -> Result<()> {
    *offset = align_up(*offset, PACKAGE_ALIGNMENT);
    weights_file
        .seek(SeekFrom::Start(*offset))
        .map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to seek weights blob: {err}"),
        })?;
    weights_file
        .write_all(data)
        .map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
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

fn maybe_prepack_tensor(
    name: &str,
    view: &safetensors::tensor::TensorView<'_>,
    dtype: PreparedDType,
) -> Result<Option<(String, Vec<usize>, PreparedTensorLayout, Vec<u8>)>> {
    if name.ends_with("conv1d.weight") {
        let shape = view.shape();
        if shape.len() == 3 && shape[1] == 1 {
            let prepared_name = format!("{name}.__dotcache_depthwise_squeezed");
            return Ok(Some((
                prepared_name,
                vec![shape[0], shape[2]],
                PreparedTensorLayout::DepthwiseConvSqueezed,
                view.data().to_vec(),
            )));
        }
    }

    if name.ends_with("dt_bias") {
        let prepared_name = format!("{name}.__dotcache_head_bias_reshaped");
        return Ok(Some((
            prepared_name,
            vec![1, 1, view.shape()[0]],
            PreparedTensorLayout::HeadBiasReshaped,
            view.data().to_vec(),
        )));
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
            _ => {
                return Ok(None);
            }
        };
        return Ok(Some((
            prepared_name,
            vec![1, 1, view.shape()[0]],
            PreparedTensorLayout::HeadExpReshaped,
            prepared_bytes,
        )));
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

fn load_typed_tensor<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let elem_size = std::mem::size_of::<T>();
    if !data.len().is_multiple_of(elem_size) {
        return Err(RuntimeError::External {
            context: "prepared-model-package",
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
        .ok_or_else(|| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("tensor shape {:?} overflows element count", shape),
        })?;
    if elem_count != expected_elem_count {
        return Err(RuntimeError::External {
            context: "prepared-model-package",
            message: format!(
                "tensor element count mismatch: bytes imply {} elems but shape {:?} implies {}",
                elem_count, shape, expected_elem_count
            ),
        });
    }
    let storage = if (data.as_ptr() as usize).is_multiple_of(elem_size) {
        let typed =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        device.storage_from_slice(typed).map_err(RuntimeError::from)?
    } else {
        let mut owned = Vec::<T>::with_capacity(elem_count);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), owned.as_mut_ptr() as *mut u8, data.len());
            owned.set_len(elem_count);
        }
        device.storage_owned(owned).map_err(RuntimeError::from)?
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
        return Err(RuntimeError::External {
            context: "prepared-model-package",
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
        fs::create_dir_all(parent).map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
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
                Err(RuntimeError::External {
                    context: "prepared-model-package",
                    message: format!(
                        "timed out waiting for package build lock {}",
                        lock_path.display()
                    ),
                })
            }
        }
        Err(err) => Err(RuntimeError::External {
            context: "prepared-model-package",
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

fn package_root(
    model_family: &str,
    model_id: &str,
    revision: &str,
    target: &ModelTarget,
) -> Result<PathBuf> {
    Ok(package_cache_root()?.join(model_family).join(sanitize_path_component(model_id)).join(
        sanitize_path_component(revision),
    )
    .join(format!("converter-v{}", PACKAGE_CONVERTER_VERSION))
    .join(&target.backend)
    .join(&target.family))
}

fn package_alias_path(model_family: &str, model_id: &str, target: &ModelTarget) -> Result<PathBuf> {
    Ok(package_cache_root()?
        .join(model_family)
        .join(sanitize_path_component(model_id))
        .join("aliases")
        .join(&target.backend)
        .join(&target.family)
        .join("active.json"))
}

fn temp_package_dir(package_root: &Path) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("system time error: {err}"),
        })?
        .as_nanos();
    let temp_name = format!(
        ".tmp-{}-{}",
        std::process::id(),
        nanos
    );
    Ok(package_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(temp_name))
}

fn package_cache_root() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| RuntimeError::External {
        context: "prepared-model-package",
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
    target: &ModelTarget,
) -> Result<Option<PreparedPackageAlias>> {
    let path = package_alias_path(model_family, model_id, target)?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to read alias {}: {err}", path.display()),
    })?;
    let alias = serde_json::from_slice(&bytes).map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to parse alias {}: {err}", path.display()),
    })?;
    Ok(Some(alias))
}

fn write_alias(
    model_family: &str,
    model_id: &str,
    target: &ModelTarget,
    alias: &PreparedPackageAlias,
) -> Result<()> {
    let path = package_alias_path(model_family, model_id, target)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to create {}: {err}", parent.display()),
        })?;
    }
    fs::write(
        &path,
        serde_json::to_vec_pretty(alias).map_err(|err| RuntimeError::External {
            context: "prepared-model-package",
            message: format!("failed to serialize alias {}: {err}", path.display()),
        })?,
    )
    .map_err(|err| RuntimeError::External {
        context: "prepared-model-package",
        message: format!("failed to write alias {}: {err}", path.display()),
    })?;
    Ok(())
}

fn sanitize_path_component(value: &str) -> OsString {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-' => {
                encoded.push(*byte as char)
            }
            _ => encoded.push_str(&format!("~{:02X}", byte)),
        }
    }
    OsString::from(encoded)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn detect_hip_family() -> Option<String> {
    let output = Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|token| token.starts_with("gfx"))
        .map(|token| token.trim().to_string())
}

fn detect_cuda_family() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let cap = stdout.lines().next()?.trim();
    let digits = cap.chars().filter(|ch| ch.is_ascii_digit()).collect::<String>();
    if digits.is_empty() {
        None
    } else {
        Some(format!("sm{digits}"))
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
