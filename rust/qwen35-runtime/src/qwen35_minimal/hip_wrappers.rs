#[cfg(feature = "qwen35-minimal-hip")]
use super::hip;
use super::model::{
    hip_output_bytes_to_cpu_storage, hip_tensor_from_host_bytes, trace_hip_wrapper_fallback,
};
use super::frontend::ImmutableEmbedding;
use super::model::DeltaNetScanMode;
use candle::{DType, Device, DeviceLocation, Result, Tensor};
use candle_core as candle;
use std::sync::atomic::{AtomicBool, Ordering};

struct HipSwigluMul;

impl candle::CustomOp2 for HipSwigluMul {
    fn name(&self) -> &'static str {
        "hip-swiglu-mul"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-swiglu-mul has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        gate: &candle::HipStorage,
        gate_layout: &candle::Layout,
        up: &candle::HipStorage,
        up_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(gate_layout.is_contiguous() && up_layout.is_contiguous()) {
            candle::bail!("hip-swiglu-mul requires contiguous inputs")
        }
        if gate_layout.shape() != up_layout.shape() {
            candle::bail!(
                "hip-swiglu-mul shape mismatch: gate={:?} up={:?}",
                gate_layout.shape().dims(),
                up_layout.shape().dims()
            )
        }
        if gate.dtype() != up.dtype() {
            candle::bail!(
                "hip-swiglu-mul requires matching dtypes, got gate={:?} up={:?}",
                gate.dtype(),
                up.dtype()
            )
        }

        let device = gate.device().clone();
        let storage_dtype = gate.dtype();
        let elem_count = gate_layout.shape().elem_count();
        let out_shape = gate_layout.shape().clone();
        let mut output = vec![0u8; elem_count.saturating_mul(storage_dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_swiglu_mul(
                hip::dtype_code(storage_dtype)?,
                device.ordinal(),
                elem_count,
                gate.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
                up.raw_device_ptr_with_offset(up_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage_dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_swiglu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_swiglu_mul_host_buffer(gate, up)? {
        return hip_tensor_from_host_bytes(gate.device(), gate.dtype(), shape, output);
    }
    trace_hip_wrapper_fallback("hip_swiglu_mul", gate);
    gate.apply_op2_no_bwd(up, &HipSwigluMul)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_swiglu_mul_host_buffer(
    gate: &Tensor,
    up: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let ordinal = match gate.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !up.device().same_device(gate.device()) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(gate.dtype()) else {
        return Ok(None);
    };
    if gate.dtype() != up.dtype() {
        return Ok(None);
    }
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (up_storage, up_layout) = up.storage_and_layout();
    let (Storage::Hip(gate_storage), Storage::Hip(up_storage)) = (&*gate_storage, &*up_storage) else {
        return Ok(None);
    };
    if !(gate_layout.is_contiguous() && up_layout.is_contiguous()) {
        return Ok(None);
    }
    if gate_layout.shape() != up_layout.shape() {
        return Ok(None);
    }
    let shape = gate_layout.shape().dims().to_vec();
    let elem_count = gate_layout.shape().elem_count();
    let mut out =
        vec![0u8; elem_count.saturating_mul(gate.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_swiglu_mul(
            dtype_code,
            ordinal,
            elem_count,
            gate_storage.raw_device_ptr_with_offset(gate_layout.start_offset())? as *const c_void,
            up_storage.raw_device_ptr_with_offset(up_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-swiglu-mul-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_swiglu_mul_host_buffer(
    gate: &Tensor,
    up: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (gate, up);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipEmbeddingLookup {
    vocab_size: usize,
    hidden_size: usize,
}

impl candle::CustomOp2 for HipEmbeddingLookup {
    fn name(&self) -> &'static str {
        "hip-embedding-lookup"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-embedding-lookup has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        embeddings: &candle::HipStorage,
        embeddings_layout: &candle::Layout,
        indexes: &candle::HipStorage,
        indexes_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(embeddings_layout.is_contiguous() && indexes_layout.is_contiguous()) {
            candle::bail!("hip-embedding-lookup requires contiguous inputs")
        }
        let dims = embeddings_layout.shape().dims();
        if dims.len() != 2 {
            candle::bail!(
                "hip-embedding-lookup expected [vocab, hidden] embeddings, got {:?}",
                dims
            )
        }
        if dims[0] != self.vocab_size || dims[1] != self.hidden_size {
            candle::bail!(
                "hip-embedding-lookup embedding shape mismatch got {:?} expected [{}, {}]",
                dims,
                self.vocab_size,
                self.hidden_size
            )
        }

        let mut out_dims = indexes_layout.shape().dims().to_vec();
        out_dims.push(self.hidden_size);
        let out_shape = candle::Shape::from(out_dims);
        let device = embeddings.device().clone();
        let token_count = indexes_layout.shape().elem_count();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(embeddings.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let device_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_embedding_lookup(
                hip::dtype_code(embeddings.dtype())?,
                hip::index_dtype_code(indexes.dtype())?,
                device.ordinal(),
                token_count,
                self.vocab_size,
                self.hidden_size,
                embeddings.raw_device_ptr_with_offset(embeddings_layout.start_offset())?
                    as *const c_void,
                indexes.raw_device_ptr_with_offset(indexes_layout.start_offset())? as *const c_void,
                device_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(embeddings.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_embedding_lookup(embeddings: &Tensor, indexes: &Tensor) -> Result<Tensor> {
    let embeddings = embeddings.contiguous()?;
    let indexes = indexes.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_embedding_lookup_host_buffer(&embeddings, &indexes)? {
        return hip_tensor_from_host_bytes(embeddings.device(), embeddings.dtype(), shape, output);
    }
    let (vocab_size, hidden_size) = embeddings.dims2()?;
    trace_hip_wrapper_fallback("hip_embedding_lookup", &embeddings);
    embeddings.apply_op2_no_bwd(
        &indexes,
        &HipEmbeddingLookup {
            vocab_size,
            hidden_size,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_embedding_lookup_host_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let embeddings = embeddings.contiguous()?;
    let indexes = indexes.contiguous()?;
    let ordinal = match embeddings.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !indexes.device().same_device(embeddings.device()) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(embeddings.dtype()) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.dtype()) else {
        return Ok(None);
    };
    let (embeddings_storage, embeddings_layout) = embeddings.storage_and_layout();
    let (indexes_storage, indexes_layout) = indexes.storage_and_layout();
    let (Storage::Hip(embeddings_storage), Storage::Hip(indexes_storage)) =
        (&*embeddings_storage, &*indexes_storage)
    else {
        return Ok(None);
    };
    if !(embeddings_layout.is_contiguous() && indexes_layout.is_contiguous()) {
        return Ok(None);
    }
    let (vocab_size, hidden_size) = embeddings_layout.shape().dims2()?;
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(hidden_size);
    let mut out =
        vec![0u8; token_count.saturating_mul(hidden_size).saturating_mul(embeddings.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            vocab_size,
            hidden_size,
            embeddings_storage
                .raw_device_ptr_with_offset(embeddings_layout.start_offset())? as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-embedding-lookup-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_embedding_lookup_host_buffer(
    embeddings: &Tensor,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embeddings, indexes);
    Ok(None)
}

#[derive(Debug, Clone)]
struct HipImmutableEmbeddingLookup {
    embedding: ImmutableEmbedding,
}

impl candle::CustomOp1 for HipImmutableEmbeddingLookup {
    fn name(&self) -> &'static str {
        "hip-immutable-embedding-lookup"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-immutable-embedding-lookup has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        indexes: &candle::HipStorage,
        indexes_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !indexes_layout.is_contiguous() {
            candle::bail!("hip-immutable-embedding-lookup requires contiguous indexes")
        }
        let device = indexes.device().clone();
        let device_ptr = self.embedding.registered_device_ptr(device.ordinal())?;
        let token_count = indexes_layout.shape().elem_count();
        let mut out_dims = indexes_layout.shape().dims().to_vec();
        out_dims.push(self.embedding.meta.hidden_size);
        let out_shape = candle::Shape::from(out_dims);
        let elem_count = out_shape.elem_count();
        let mut output =
            vec![0u8; elem_count.saturating_mul(self.embedding.meta.dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_embedding_lookup(
                hip::dtype_code(self.embedding.meta.dtype)?,
                hip::index_dtype_code(indexes.dtype())?,
                device.ordinal(),
                token_count,
                self.embedding.meta.vocab_size,
                self.embedding.meta.hidden_size,
                device_ptr as *const c_void,
                indexes.raw_device_ptr_with_offset(indexes_layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(self.embedding.meta.dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_immutable_embedding_lookup(embedding: &ImmutableEmbedding, indexes: &Tensor) -> Result<Tensor> {
    let indexes = indexes.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_immutable_embedding_lookup_host_buffer(embedding, &indexes)? {
        return hip_tensor_from_host_bytes(indexes.device(), embedding.meta.dtype, shape, output);
    }
    trace_hip_wrapper_fallback("hip_immutable_embedding_lookup", &indexes);
    indexes.apply_op1_no_bwd(&HipImmutableEmbeddingLookup {
        embedding: embedding.clone(),
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_immutable_embedding_lookup_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let indexes = indexes.contiguous()?;
    let ordinal = match indexes.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(embedding.meta.dtype) else {
        return Ok(None);
    };
    let Ok(index_dtype_code) = hip::index_dtype_code(indexes.dtype()) else {
        return Ok(None);
    };
    let (indexes_storage, indexes_layout) = indexes.storage_and_layout();
    let Storage::Hip(indexes_storage) = &*indexes_storage else {
        return Ok(None);
    };
    if !indexes_layout.is_contiguous() {
        return Ok(None);
    }
    let token_count = indexes_layout.shape().elem_count();
    let mut shape = indexes_layout.shape().dims().to_vec();
    shape.push(embedding.meta.hidden_size);
    let mut out = vec![
        0u8;
        token_count
            .saturating_mul(embedding.meta.hidden_size)
            .saturating_mul(embedding.meta.dtype.size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let embedding_ptr = embedding.registered_device_ptr(ordinal)?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_embedding_lookup(
            dtype_code,
            index_dtype_code,
            ordinal,
            token_count,
            embedding.meta.vocab_size,
            embedding.meta.hidden_size,
            embedding_ptr as *const c_void,
            indexes_storage.raw_device_ptr_with_offset(indexes_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "hip-immutable-embedding-lookup-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_immutable_embedding_lookup_host_buffer(
    embedding: &ImmutableEmbedding,
    indexes: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embedding, indexes);
    Ok(None)
}

#[derive(Debug, Clone)]
struct HipImmutableOutputProjection {
    embedding: ImmutableEmbedding,
}

impl candle::CustomOp1 for HipImmutableOutputProjection {
    fn name(&self) -> &'static str {
        "hip-immutable-output-projection"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-immutable-output-projection has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        hidden: &candle::HipStorage,
        hidden_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !hidden_layout.is_contiguous() {
            candle::bail!("hip-immutable-output-projection requires contiguous hidden states")
        }
        let dims = hidden_layout.shape().dims();
        let hidden_size = *dims.last().ok_or_else(|| candle::Error::Msg("hidden state rank must be >= 1".to_string()))?;
        if hidden_size != self.embedding.meta.hidden_size {
            candle::bail!(
                "hip-immutable-output-projection hidden size mismatch got {} expected {}",
                hidden_size,
                self.embedding.meta.hidden_size
            )
        }
        let rows = hidden_layout.shape().elem_count() / hidden_size;
        let device = hidden.device().clone();
        let weight_ptr = self.embedding.registered_device_ptr(device.ordinal())?;
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().expect("validated non-empty dims") = self.embedding.meta.vocab_size;
        let out_shape = candle::Shape::from(out_dims);
        let elem_count = out_shape.elem_count();
        let mut output =
            vec![0u8; elem_count.saturating_mul(self.embedding.meta.dtype.size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
                hip::dtype_code(self.embedding.meta.dtype)?,
                device.ordinal(),
                rows,
                self.embedding.meta.hidden_size,
                self.embedding.meta.vocab_size,
                hidden.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
                weight_ptr,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(self.embedding.meta.dtype, output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn immutable_output_projection(embedding: &ImmutableEmbedding, hidden_states: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if embedding.meta.device.is_hip() && hidden_states.device().is_hip() {
        let hidden_states = hidden_states.contiguous()?;
        if let Some((output, shape)) = immutable_output_projection_host_buffer(embedding, &hidden_states)? {
            return hip_tensor_from_host_bytes(hidden_states.device(), embedding.meta.dtype, shape, output);
        }
        trace_hip_wrapper_fallback("immutable_output_projection", &hidden_states);
        return hidden_states.apply_op1_no_bwd(&HipImmutableOutputProjection {
            embedding: embedding.clone(),
        });
    }

    let fallback = embedding.ensure_fallback_embedding()?;
    let weight = fallback.embeddings().t()?;
    hidden_states.matmul(&weight)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn immutable_output_projection_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    if !(embedding.device().is_hip() && hidden_states.device().is_hip()) {
        return Ok(None);
    }
    let hidden_states = hidden_states.contiguous()?;
    let ordinal = match hidden_states.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let (hidden_storage, hidden_layout) = hidden_states.storage_and_layout();
    let Storage::Hip(hidden_storage) = &*hidden_storage else {
        return Ok(None);
    };
    if !hidden_layout.is_contiguous() {
        return Ok(None);
    }
    let dims = hidden_layout.shape().dims();
    let hidden_size = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("hidden state rank must be >= 1".to_string()))?;
    if hidden_size != embedding.meta.hidden_size {
        return Ok(None);
    }
    let rows = hidden_layout.shape().elem_count() / hidden_size;
    let mut shape = dims.to_vec();
    *shape.last_mut().expect("validated non-empty dims") = embedding.meta.vocab_size;
    let mut out = vec![
        0u8;
        shape.iter().product::<usize>().saturating_mul(embedding.dtype().size_in_bytes())
    ];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let weight_ptr = embedding.registered_device_ptr(ordinal)?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_output_projection_lookup(
            hip::dtype_code(embedding.dtype())?,
            ordinal,
            rows,
            embedding.meta.hidden_size,
            embedding.meta.vocab_size,
            hidden_storage.raw_device_ptr_with_offset(hidden_layout.start_offset())? as *const c_void,
            weight_ptr,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error(
            "hip-immutable-output-projection-host-buffer",
            status,
        ));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn immutable_output_projection_host_buffer(
    embedding: &ImmutableEmbedding,
    hidden_states: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (embedding, hidden_states);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipCausalMask {
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
}

impl candle::CustomOp1 for HipCausalMask {
    fn name(&self) -> &'static str {
        "hip-causal-mask"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-causal-mask has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        let device = storage.device().clone();
        let kv_len = self.tgt_len + self.seqlen_offset;
        let out_shape = candle::Shape::from((self.batch_size, 1usize, self.tgt_len, kv_len));
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_causal_mask(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.batch_size,
                self.tgt_len,
                self.seqlen_offset,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_causal_mask(device: &Device, dtype: DType, batch_size: usize, tgt_len: usize, seqlen_offset: usize) -> Result<Tensor> {
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) =
        hip_causal_mask_host_buffer(device, dtype, batch_size, tgt_len, seqlen_offset)?
    {
        return hip_tensor_from_host_bytes(device, dtype, shape, output);
    }
    let seed = Tensor::zeros(1usize, dtype, device)?;
    trace_hip_wrapper_fallback("hip_causal_mask", &seed);
    seed.apply_op1_no_bwd(&HipCausalMask {
        batch_size,
        tgt_len,
        seqlen_offset,
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_causal_mask_host_buffer(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let ordinal = match device.location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(dtype) else {
        return Ok(None);
    };
    let kv_len = tgt_len + seqlen_offset;
    let shape = vec![batch_size, 1, tgt_len, kv_len];
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const std::ffi::c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_causal_mask(
            dtype_code,
            ordinal,
            batch_size,
            tgt_len,
            seqlen_offset,
            device_ptr as *mut std::ffi::c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-causal-mask-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_causal_mask_host_buffer(
    device: &Device,
    dtype: DType,
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (device, dtype, batch_size, tgt_len, seqlen_offset);
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipCumsumLastDim {
    rows: usize,
    cols: usize,
}

impl candle::CustomOp1 for HipCumsumLastDim {
    fn name(&self) -> &'static str {
        "hip-cumsum-last-dim"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("hip-cumsum-last-dim has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !layout.is_contiguous() {
            candle::bail!("hip-cumsum-last-dim requires contiguous input")
        }
        let dims = layout.shape().dims();
        let cols = *dims.last().ok_or_else(|| {
            candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
        })?;
        let rows = layout.shape().elem_count() / cols;
        if rows != self.rows || cols != self.cols {
            candle::bail!(
                "hip-cumsum-last-dim shape mismatch input={:?} expected_rows={} expected_cols={}",
                dims,
                self.rows,
                self.cols
            )
        }

        let device = storage.device().clone();
        let out_shape = layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.rows,
                self.cols,
                storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_cumsum_last_dim(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_cumsum_last_dim_host_buffer(&xs)? {
        return hip_tensor_from_host_bytes(xs.device(), xs.dtype(), shape, output);
    }
    let dims = xs.dims();
    let cols = *dims.last().ok_or_else(|| {
        candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = xs.elem_count() / cols;
    trace_hip_wrapper_fallback("hip_cumsum_last_dim", &xs);
    xs.apply_op1_no_bwd(&HipCumsumLastDim { rows, cols })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_cumsum_last_dim_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let cols = *shape.last().ok_or_else(|| {
        candle::Error::Msg("hip-cumsum-last-dim requires non-empty shape".into())
    })?;
    let rows = layout.shape().elem_count() / cols;
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cumsum_last_dim(
            dtype_code,
            ordinal,
            rows,
            cols,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-cumsum-last-dim-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_cumsum_last_dim_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_exp_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_exp(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-exp-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_exp_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_recip_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_recip(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-recip-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_recip_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_sigmoid_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sigmoid(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-sigmoid-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_sigmoid_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_log_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_log(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-log-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_log_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_cast_host_buffer(
    xs: &Tensor,
    output_dtype: DType,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(input_dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let Ok(output_dtype_code) = hip::dtype_code(output_dtype) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(output_dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_cast(
            input_dtype_code,
            output_dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-cast-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_cast_host_buffer(
    xs: &Tensor,
    output_dtype: DType,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, output_dtype);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn hip_binary_broadcast_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: i32,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    const MAX_RANK: usize = 8;

    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    let ordinal = match lhs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    match rhs.device().location() {
        DeviceLocation::Hip { gpu_id } if gpu_id == ordinal => {}
        _ => return Ok(None),
    }
    if lhs.dtype() != rhs.dtype() {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(lhs.dtype()) else {
        return Ok(None);
    };
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let rank = lhs_dims.len().max(rhs_dims.len());
    if rank == 0 || rank > MAX_RANK {
        return Ok(None);
    }

    let mut out_dims = [1i32; MAX_RANK];
    let mut lhs_strides = [0i32; MAX_RANK];
    let mut rhs_strides = [0i32; MAX_RANK];

    let mut lhs_contig = vec![0usize; lhs_dims.len()];
    let mut rhs_contig = vec![0usize; rhs_dims.len()];
    let mut stride = 1usize;
    for (i, dim) in lhs_dims.iter().enumerate().rev() {
        lhs_contig[i] = stride;
        stride = stride.saturating_mul(*dim);
    }
    stride = 1usize;
    for (i, dim) in rhs_dims.iter().enumerate().rev() {
        rhs_contig[i] = stride;
        stride = stride.saturating_mul(*dim);
    }

    let lhs_pad = rank - lhs_dims.len();
    let rhs_pad = rank - rhs_dims.len();
    let mut total_elems = 1usize;
    for dim in 0..rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_dims[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_dims[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_dims[dim] = i32::try_from(out_dim)
            .map_err(|_| candle::Error::Msg("broadcast dim overflow".into()))?;
        total_elems = total_elems.saturating_mul(out_dim);
        lhs_strides[dim] = if dim < lhs_pad || lhs_dim == 1 {
            0
        } else {
            i32::try_from(lhs_contig[dim - lhs_pad])
                .map_err(|_| candle::Error::Msg("lhs stride overflow".into()))?
        };
        rhs_strides[dim] = if dim < rhs_pad || rhs_dim == 1 {
            0
        } else {
            i32::try_from(rhs_contig[dim - rhs_pad])
                .map_err(|_| candle::Error::Msg("rhs stride overflow".into()))?
        };
    }

    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let Storage::Hip(lhs_storage) = &*lhs_storage else {
        return Ok(None);
    };
    let Storage::Hip(rhs_storage) = &*rhs_storage else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }

    let shape: Vec<usize> = out_dims[..rank].iter().map(|&d| d as usize).collect();
    let mut out = vec![0u8; total_elems.saturating_mul(lhs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_binary_broadcast(
            op,
            dtype_code,
            ordinal,
            i32::try_from(rank).map_err(|_| candle::Error::Msg("rank overflow".into()))?,
            total_elems,
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            lhs_strides.as_ptr(),
            rhs_strides.as_ptr(),
            out_dims.as_ptr(),
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-binary-broadcast-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn hip_binary_broadcast_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: i32,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (lhs, rhs, op);
    Ok(None)
}

pub(crate) fn hip_broadcast_add_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 0)
}

pub(crate) fn hip_broadcast_sub_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 1)
}

pub(crate) fn hip_broadcast_mul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 2)
}

pub(crate) fn hip_broadcast_div_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_binary_broadcast_host_buffer(lhs, rhs, 3)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_matmul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    const MAX_BATCH_RANK: usize = 8;

    let lhs = lhs.contiguous()?;
    let rhs = rhs.contiguous()?;
    let ordinal = match lhs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    match rhs.device().location() {
        DeviceLocation::Hip { gpu_id } if gpu_id == ordinal => {}
        _ => return Ok(None),
    }
    if lhs.dtype() != rhs.dtype() {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(lhs.dtype()) else {
        return Ok(None);
    };

    let lhs_shape = lhs.dims();
    let rhs_shape = rhs.dims();
    if lhs_shape.is_empty() || rhs_shape.is_empty() {
        return Ok(None);
    }
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    let lhs_k = lhs_shape[lhs_rank - 1];
    let rhs_k = rhs_shape[rhs_rank.saturating_sub(2)];
    if lhs_k != rhs_k {
        return Ok(None);
    }
    let m = if lhs_rank >= 2 { lhs_shape[lhs_rank - 2] } else { 1 };
    let n = rhs_shape[rhs_rank - 1];
    let lhs_batch = &lhs_shape[..lhs_rank.saturating_sub(2)];
    let rhs_batch = &rhs_shape[..rhs_rank.saturating_sub(2)];

    let batch_rank = lhs_batch.len().max(rhs_batch.len());
    if batch_rank > MAX_BATCH_RANK {
        return Ok(None);
    }
    let mut out_batch_dims = [1i32; MAX_BATCH_RANK];
    let mut lhs_batch_dims = [1i32; MAX_BATCH_RANK];
    let mut rhs_batch_dims = [1i32; MAX_BATCH_RANK];
    let lhs_pad = batch_rank.saturating_sub(lhs_batch.len());
    let rhs_pad = batch_rank.saturating_sub(rhs_batch.len());
    let mut batch_elems = 1usize;
    for dim in 0..batch_rank {
        let lhs_dim = if dim < lhs_pad { 1 } else { lhs_batch[dim - lhs_pad] };
        let rhs_dim = if dim < rhs_pad { 1 } else { rhs_batch[dim - rhs_pad] };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Ok(None);
        }
        let out_dim = lhs_dim.max(rhs_dim);
        out_batch_dims[dim] = i32::try_from(out_dim)
            .map_err(|_| candle::Error::Msg("matmul batch dim overflow".into()))?;
        lhs_batch_dims[dim] = i32::try_from(lhs_dim)
            .map_err(|_| candle::Error::Msg("matmul lhs batch dim overflow".into()))?;
        rhs_batch_dims[dim] = i32::try_from(rhs_dim)
            .map_err(|_| candle::Error::Msg("matmul rhs batch dim overflow".into()))?;
        batch_elems = batch_elems.saturating_mul(out_dim);
    }

    let (lhs_storage, lhs_layout) = lhs.storage_and_layout();
    let (rhs_storage, rhs_layout) = rhs.storage_and_layout();
    let Storage::Hip(lhs_storage) = &*lhs_storage else {
        return Ok(None);
    };
    let Storage::Hip(rhs_storage) = &*rhs_storage else {
        return Ok(None);
    };
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Ok(None);
    }

    let mut out_shape = lhs_batch
        .iter()
        .zip(rhs_batch.iter())
        .map(|(a, b)| (*a).max(*b))
        .collect::<Vec<_>>();
    if out_shape.len() != batch_rank {
        out_shape = out_batch_dims[..batch_rank].iter().map(|&d| d as usize).collect();
    }
    if lhs_rank >= 2 {
        out_shape.push(m);
    }
    out_shape.push(n);

    let total_elems = batch_elems
        .saturating_mul(m)
        .saturating_mul(n);
    let mut out = vec![0u8; total_elems.saturating_mul(lhs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_batched_matmul(
            dtype_code,
            ordinal,
            i32::try_from(batch_rank).map_err(|_| candle::Error::Msg("matmul batch rank overflow".into()))?,
            batch_elems,
            i32::try_from(m).map_err(|_| candle::Error::Msg("matmul m overflow".into()))?,
            i32::try_from(n).map_err(|_| candle::Error::Msg("matmul n overflow".into()))?,
            i32::try_from(lhs_k).map_err(|_| candle::Error::Msg("matmul k overflow".into()))?,
            lhs_batch_dims.as_ptr(),
            rhs_batch_dims.as_ptr(),
            out_batch_dims.as_ptr(),
            lhs_storage.raw_device_ptr_with_offset(lhs_layout.start_offset())? as *const c_void,
            rhs_storage.raw_device_ptr_with_offset(rhs_layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-matmul-host-buffer", status));
    }
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_matmul_host_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (lhs, rhs);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_mul_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_mul_scalar(
            dtype_code,
            ordinal,
            total_elems,
            scalar as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-mul-scalar-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_mul_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, scalar);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
fn hip_reduce_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
    sum: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    if dim >= shape.len() {
        return Ok(None);
    }
    let outer = shape[..dim].iter().product::<usize>().max(1);
    let reduce = shape[dim];
    let inner = shape[dim + 1..].iter().product::<usize>().max(1);
    let mut out_shape = shape.clone();
    out_shape[dim] = 1;
    let mut out = vec![0u8; out_shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_reduce_keepdim(
            dtype_code,
            ordinal,
            outer,
            reduce,
            inner,
            if sum { 1 } else { 0 },
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-reduce-keepdim-host-buffer", status));
    }
    Ok(Some((out, out_shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
fn hip_reduce_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
    sum: bool,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, dim, sum);
    Ok(None)
}

pub(crate) fn hip_sum_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_reduce_keepdim_host_buffer(xs, dim, true)
}

pub(crate) fn hip_max_keepdim_host_buffer(
    xs: &Tensor,
    dim: usize,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    hip_reduce_keepdim_host_buffer(xs, dim, false)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_add_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_add_scalar(
            dtype_code,
            ordinal,
            total_elems,
            scalar as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-add-scalar-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_add_scalar_host_buffer(
    xs: &Tensor,
    scalar: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, scalar);
    Ok(None)
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_sqrt_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let total_elems = layout.shape().elem_count();
    let mut out = vec![0u8; total_elems.saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_sqrt(
            dtype_code,
            ordinal,
            total_elems,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("hip-sqrt-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_sqrt_host_buffer(xs: &Tensor) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = xs;
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
struct HipL2Norm {
    n_rows: usize,
    n_cols: usize,
    eps: f32,
}

impl candle::CustomOp1 for HipL2Norm {
    fn name(&self) -> &'static str {
        "dotcache-hip-l2norm"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle::CpuStorage,
        _layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("dotcache-hip-l2norm has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        storage: &candle::HipStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !layout.is_contiguous() {
            candle::bail!("dotcache-hip-l2norm requires contiguous input")
        }
        let dims = layout.shape().dims();
        let n_cols = *dims.last().ok_or_else(|| {
            candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into())
        })?;
        let n_rows = layout.shape().elem_count() / n_cols;
        if n_rows != self.n_rows || n_cols != self.n_cols {
            candle::bail!(
                "dotcache-hip-l2norm shape mismatch input={:?} expected_rows={} expected_cols={}",
                layout.shape().dims(),
                self.n_rows,
                self.n_cols
            )
        }

        let device = storage.device().clone();
        let out_shape = layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(storage.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_l2norm(
                hip::dtype_code(storage.dtype())?,
                device.ordinal(),
                self.n_rows,
                self.n_cols,
                self.eps,
                storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(storage.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_l2norm_host_buffer(&xs, eps)? {
        return hip_tensor_from_host_bytes(xs.device(), xs.dtype(), shape, output);
    }
    let dims = xs.dims();
    let n_cols = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = xs.elem_count() / n_cols;
    trace_hip_wrapper_fallback("hip_l2norm", &xs);
    xs.apply_op1_no_bwd(&HipL2Norm {
        n_rows,
        n_cols,
        eps: eps as f32,
    })
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_l2norm_host_buffer(
    xs: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let xs = xs.contiguous()?;
    let ordinal = match xs.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    let Ok(dtype_code) = hip::dtype_code(xs.dtype()) else {
        return Ok(None);
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Hip(storage) = &*storage else {
        return Ok(None);
    };
    if !layout.is_contiguous() {
        return Ok(None);
    }
    let shape = layout.shape().dims().to_vec();
    let n_cols = *shape
        .last()
        .ok_or_else(|| candle::Error::Msg("dotcache-hip-l2norm requires non-empty shape".into()))?;
    let n_rows = layout.shape().elem_count() / n_cols;
    let mut out = vec![0u8; shape.iter().product::<usize>().saturating_mul(xs.dtype().size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_l2norm(
            dtype_code,
            ordinal,
            n_rows,
            n_cols,
            eps as f32,
            storage.raw_device_ptr_with_offset(layout.start_offset())? as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-l2norm-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_l2norm_host_buffer(
    xs: &Tensor,
    eps: f64,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (xs, eps);
    Ok(None)
}

pub(crate) fn softplus(xs: &Tensor) -> Result<Tensor> {
    ((xs.exp()? + 1.0)?).log()
}

#[derive(Debug, Clone, Copy)]
struct HipValueDecay {
    total_elems: usize,
    num_heads: usize,
}

impl candle::CustomOp3 for HipValueDecay {
    fn name(&self) -> &'static str {
        "dotcache-hip-value-decay"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("dotcache-hip-value-decay has no cpu implementation")
    }

    #[cfg(feature = "qwen35-minimal-hip")]
    fn hip_fwd(
        &self,
        a: &candle::HipStorage,
        a_layout: &candle::Layout,
        dt_bias: &candle::HipStorage,
        dt_bias_layout: &candle::Layout,
        a_log_exp: &candle::HipStorage,
        a_log_exp_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(a_layout.is_contiguous()
            && dt_bias_layout.is_contiguous()
            && a_log_exp_layout.is_contiguous())
        {
            candle::bail!("dotcache-hip-value-decay requires contiguous inputs")
        }
        if a.dtype() != dt_bias.dtype() || a.dtype() != a_log_exp.dtype() {
            candle::bail!(
                "dotcache-hip-value-decay requires matching dtypes, got a={:?} dt_bias={:?} a_log_exp={:?}",
                a.dtype(),
                dt_bias.dtype(),
                a_log_exp.dtype()
            )
        }

        let a_elems = a_layout.shape().elem_count();
        let dt_bias_elems = dt_bias_layout.shape().elem_count();
        let a_log_exp_elems = a_log_exp_layout.shape().elem_count();
        if a_elems != self.total_elems
            || dt_bias_elems != self.num_heads
            || a_log_exp_elems != self.num_heads
        {
            candle::bail!(
                "dotcache-hip-value-decay shape mismatch a={:?} dt_bias={:?} a_log_exp={:?} expected_total={} expected_heads={}",
                a_layout.shape().dims(),
                dt_bias_layout.shape().dims(),
                a_log_exp_layout.shape().dims(),
                self.total_elems,
                self.num_heads
            )
        }

        let device = a.device().clone();
        let out_shape = a_layout.shape().clone();
        let elem_count = out_shape.elem_count();
        let mut output = vec![0u8; elem_count.saturating_mul(a.dtype().size_in_bytes())];
        let host_ptr = output.as_mut_ptr() as *const c_void;
        let output_ptr =
            hip::register_host_mapping_for_device(device.ordinal(), host_ptr, output.len())?;
        let status = unsafe {
            hip::ffi::dotcache_qwen35_hip_value_decay(
                hip::dtype_code(a.dtype())?,
                device.ordinal(),
                self.total_elems,
                self.num_heads,
                a.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
                dt_bias.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
                a_log_exp.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                    as *const c_void,
                output_ptr as *mut c_void,
            )
        };
        hip::unregister_host_mapping(host_ptr);
        if status != 0 {
            return Err(hip::hip_error(self.name(), status));
        }
        Ok((
            candle::HipStorage::wrap_cpu_storage(
                hip_output_bytes_to_cpu_storage(a.dtype(), output)?,
                device,
            ),
            out_shape,
        ))
    }
}

pub(crate) fn hip_value_decay(a: &Tensor, dt_bias: &Tensor, a_log_exp: &Tensor) -> Result<Tensor> {
    let a = a.contiguous()?;
    let target_dtype = a.dtype();
    let dt_bias = dt_bias.contiguous()?;
    let dt_bias = if dt_bias.dtype() == target_dtype {
        dt_bias
    } else {
        dt_bias.to_dtype(target_dtype)?
    };
    let a_log_exp = a_log_exp.contiguous()?;
    let a_log_exp = if a_log_exp.dtype() == target_dtype {
        a_log_exp
    } else {
        a_log_exp.to_dtype(target_dtype)?
    };
    #[cfg(feature = "qwen35-minimal-hip")]
    if let Some((output, shape)) = hip_value_decay_host_buffer(&a, &dt_bias, &a_log_exp)? {
        return hip_tensor_from_host_bytes(a.device(), a.dtype(), shape, output);
    }
    let total_elems = a.elem_count();
    let num_heads = dt_bias.elem_count();
    trace_hip_wrapper_fallback("hip_value_decay", &a);
    a.apply_op3_no_bwd(
        &dt_bias,
        &a_log_exp,
        &HipValueDecay {
            total_elems,
            num_heads,
        },
    )
}

#[cfg(feature = "qwen35-minimal-hip")]
pub(crate) fn hip_value_decay_host_buffer(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    use candle::Storage;
    use std::ffi::c_void;

    let a = a.contiguous()?;
    let target_dtype = a.dtype();
    let dt_bias = dt_bias.contiguous()?;
    let dt_bias = if dt_bias.dtype() == target_dtype {
        dt_bias
    } else {
        dt_bias.to_dtype(target_dtype)?
    };
    let a_log_exp = a_log_exp.contiguous()?;
    let a_log_exp = if a_log_exp.dtype() == target_dtype {
        a_log_exp
    } else {
        a_log_exp.to_dtype(target_dtype)?
    };
    let ordinal = match a.device().location() {
        DeviceLocation::Hip { gpu_id } => gpu_id,
        _ => return Ok(None),
    };
    if !(dt_bias.device().same_device(a.device()) && a_log_exp.device().same_device(a.device())) {
        return Ok(None);
    }
    let Ok(dtype_code) = hip::dtype_code(target_dtype) else {
        return Ok(None);
    };
    let (a_storage, a_layout) = a.storage_and_layout();
    let (dt_bias_storage, dt_bias_layout) = dt_bias.storage_and_layout();
    let (a_log_exp_storage, a_log_exp_layout) = a_log_exp.storage_and_layout();
    let (Storage::Hip(a_storage), Storage::Hip(dt_bias_storage), Storage::Hip(a_log_exp_storage)) =
        (&*a_storage, &*dt_bias_storage, &*a_log_exp_storage)
    else {
        return Ok(None);
    };
    if !(a_layout.is_contiguous() && dt_bias_layout.is_contiguous() && a_log_exp_layout.is_contiguous()) {
        return Ok(None);
    }
    let total_elems = a_layout.shape().elem_count();
    let num_heads = dt_bias_layout.shape().elem_count();
    if a_log_exp_layout.shape().elem_count() != num_heads {
        return Ok(None);
    }
    let shape = a_layout.shape().dims().to_vec();
    let mut out = vec![0u8; total_elems.saturating_mul(target_dtype.size_in_bytes())];
    let host_ptr = out.as_mut_ptr() as *const c_void;
    let device_ptr = hip::register_host_mapping_for_device(ordinal, host_ptr, out.len())?;
    let status = unsafe {
        hip::ffi::dotcache_qwen35_hip_value_decay(
            dtype_code,
            ordinal,
            total_elems,
            num_heads,
            a_storage.raw_device_ptr_with_offset(a_layout.start_offset())? as *const c_void,
            dt_bias_storage.raw_device_ptr_with_offset(dt_bias_layout.start_offset())? as *const c_void,
            a_log_exp_storage.raw_device_ptr_with_offset(a_log_exp_layout.start_offset())?
                as *const c_void,
            device_ptr as *mut c_void,
        )
    };
    hip::unregister_host_mapping(host_ptr);
    if status != 0 {
        return Err(hip::hip_error("dotcache-hip-value-decay-host-buffer", status));
    }
    Ok(Some((out, shape)))
}

#[cfg(not(feature = "qwen35-minimal-hip"))]
pub(crate) fn hip_value_decay_host_buffer(
    a: &Tensor,
    dt_bias: &Tensor,
    a_log_exp: &Tensor,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let _ = (a, dt_bias, a_log_exp);
    Ok(None)
}

pub(crate) fn linear_attention_compute_dtype(device: &Device, input_dtype: DType) -> DType {
    match (device.location(), input_dtype) {
        (DeviceLocation::Metal { .. }, DType::F16 | DType::BF16) => input_dtype,
        _ => DType::F32,
    }
}

fn recommended_metal_linear_chunk_size(sequence_length: usize) -> usize {
    match sequence_length {
        0..=1024 => 16,
        _ => 24,
    }
}

fn recommended_hip_linear_chunk_size(sequence_length: usize) -> usize {
    match sequence_length {
        0..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

fn use_hip_short_linear_chunks() -> bool {
    matches!(
        std::env::var("DOTCACHE_QWEN35_HIP_SHORT_LINEAR_CHUNKS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn debug_linear_chunk_choice(sequence_length: usize, chunk_size: usize) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if std::env::var("CANDLE_QWEN35_DEBUG_CHUNK").is_ok() && !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "qwen3.5 linear chunk choice: sequence_length={} chunk_size={}",
            sequence_length, chunk_size
        );
    }
}

pub(crate) fn linear_attention_chunk_size(device: &Device, sequence_length: usize) -> usize {
    if let Ok(raw_value) = std::env::var("CANDLE_QWEN35_LINEAR_CHUNK_SIZE") {
        if let Ok(parsed) = raw_value.trim().parse::<usize>() {
            if parsed > 0 {
                debug_linear_chunk_choice(sequence_length, parsed);
                return parsed;
            }
        }
    }
    let chunk_size = match device.location() {
        DeviceLocation::Metal { .. } => recommended_metal_linear_chunk_size(sequence_length),
        DeviceLocation::Hip { .. } if use_hip_short_linear_chunks() => {
            recommended_hip_linear_chunk_size(sequence_length)
        }
        _ => 64,
    };
    debug_linear_chunk_choice(sequence_length, chunk_size);
    chunk_size
}

pub(crate) fn use_delta_state_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= min_sequence
        && matches!(
            std::env::var("CANDLE_QWEN35_DELTA_STATE_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

pub(crate) fn use_delta_state_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

pub(crate) fn use_delta_chunk_fused_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

pub(crate) fn use_delta_full_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    let min_sequence = std::env::var("DOTCACHE_QWEN35_DELTA_KERNEL_MIN_SEQUENCE")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096);
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= min_sequence)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Hip { .. } => match std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL") {
            Ok(raw) => !matches!(
                raw.trim(),
                "0" | "false" | "FALSE" | "no" | "NO"
            ),
            Err(_) => true,
        },
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

pub(crate) fn use_hip_exact_multi_chunk_full_scan_prefill(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    num_chunks: usize,
    chunk_size: usize,
) -> bool {
    if !matches!(device.location(), DeviceLocation::Hip { .. }) {
        return false;
    }
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && num_chunks > 1
        && num_chunks <= 4
        && sequence_length > chunk_size
        && chunk_size <= 64)
    {
        return false;
    }

    match std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL") {
        Ok(raw) => !matches!(raw.trim(), "0" | "false" | "FALSE" | "no" | "NO"),
        Err(_) => true,
    }
}

pub(crate) fn use_delta_recurrent_prefill_kernel(device: &Device, sequence_length: usize) -> bool {
    sequence_length >= 4096
        && match device.location() {
            DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } => matches!(
                std::env::var("CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            ),
            _ => false,
        }
}

pub(crate) fn use_delta_chunk_step_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 2048
        && chunk_size <= 24)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}
