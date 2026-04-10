use crate::cache::PageStore;
use crate::page::{KvPage, PageId};
use crate::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendDevice {
    Cpu,
    Metal { ordinal: usize },
    Cuda { ordinal: usize },
    Hip { ordinal: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    pub name: &'static str,
    pub device: BackendDevice,
    pub supports_prepare_cache: bool,
    pub supports_virtual_pages: bool,
    pub supports_device_resident_pages: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPathMode {
    Paged,
    Fused,
}

impl AttentionPathMode {
    pub fn default_for_backend_device(device: &BackendDevice) -> Self {
        match device {
            BackendDevice::Metal { .. } => Self::Fused,
            BackendDevice::Cpu | BackendDevice::Cuda { .. } | BackendDevice::Hip { .. } => {
                Self::Paged
            }
        }
    }

    #[cfg(feature = "candle")]
    pub fn default_for_selector(selector: &CandleDeviceSelector) -> Self {
        Self::default_for_backend_device(&selector.backend_device())
    }
}

impl std::fmt::Display for AttentionPathMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Paged => write!(f, "paged"),
            Self::Fused => write!(f, "fused"),
        }
    }
}

impl std::str::FromStr for AttentionPathMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "paged" => Ok(Self::Paged),
            "fused" => Ok(Self::Fused),
            _ => Err(format!(
                "invalid attention path `{value}`, expected `paged` or `fused`"
            )),
        }
    }
}

pub trait PageBackend {
    type Prepared<'a>
    where
        Self: 'a;

    fn descriptor(&self) -> BackendDescriptor;
    fn prepare<'a>(&self, page_id: PageId, page: &'a KvPage) -> Result<Self::Prepared<'a>>;
    fn prepare_cached<'a>(
        &'a self,
        _page_id: PageId,
        _expected_head_dim: usize,
    ) -> Result<Option<Self::Prepared<'a>>> {
        Ok(None)
    }
    fn score(&self, q: &[f32], page: &Self::Prepared<'_>, logits_out: &mut Vec<f32>) -> Result<()>;
    fn mix(&self, weights: &[f32], page: &Self::Prepared<'_>, out: &mut [f32]) -> Result<()>;

    fn decode_fused(
        &self,
        _store: &PageStore,
        _page_ids: &[PageId],
        _q: &[f32],
        _out: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    fn decode_batch_fused(
        &self,
        _store: &PageStore,
        _page_ids_by_query: &[&[PageId]],
        _queries: &[&[f32]],
        _outputs: &mut [Vec<f32>],
    ) -> Result<bool> {
        Ok(false)
    }

    fn score_batch(
        &self,
        queries: &[&[f32]],
        page: &Self::Prepared<'_>,
        logits_outs: &mut [Vec<f32>],
    ) -> Result<()> {
        if queries.len() != logits_outs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "score batch outputs",
                expected: queries.len(),
                got: logits_outs.len(),
            });
        }

        for (query, logits_out) in queries.iter().zip(logits_outs.iter_mut()) {
            self.score(query, page, logits_out)?;
        }

        Ok(())
    }

    fn mix_batch(
        &self,
        weights_batch: &[&[f32]],
        page: &Self::Prepared<'_>,
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        if weights_batch.len() != outs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "mix batch outputs",
                expected: weights_batch.len(),
                got: outs.len(),
            });
        }

        for (weights, out) in weights_batch.iter().zip(outs.iter_mut()) {
            self.mix(weights, page, out)?;
        }

        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuReferenceBackend;

impl PageBackend for CpuReferenceBackend {
    type Prepared<'a>
        = &'a KvPage
    where
        Self: 'a;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "cpu_ref",
            device: BackendDevice::Cpu,
            supports_prepare_cache: false,
            supports_virtual_pages: true,
            supports_device_resident_pages: false,
        }
    }

    fn prepare<'a>(&self, _page_id: PageId, page: &'a KvPage) -> Result<Self::Prepared<'a>> {
        Ok(page)
    }

    fn score(&self, q: &[f32], page: &Self::Prepared<'_>, logits_out: &mut Vec<f32>) -> Result<()> {
        let page = *page;
        page.score_keys(q, logits_out)
    }

    fn mix(&self, weights: &[f32], page: &Self::Prepared<'_>, out: &mut [f32]) -> Result<()> {
        let page = *page;
        page.mix_values(weights, out)
    }
}

#[cfg(feature = "candle-metal")]
#[derive(Debug, Clone, Copy)]
struct PagedAttentionDecodeMegakernel {
    batch_size: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    seqlen_offset: usize,
}

#[cfg(feature = "candle-metal")]
impl candle_core::CustomOp3 for PagedAttentionDecodeMegakernel {
    fn name(&self) -> &'static str {
        "paged-attention-decode-megakernel"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("paged-attention-decode-megakernel has no cpu implementation")
    }

    fn metal_fwd(
        &self,
        query: &candle_core::MetalStorage,
        query_layout: &candle_core::Layout,
        key: &candle_core::MetalStorage,
        key_layout: &candle_core::Layout,
        value: &candle_core::MetalStorage,
        value_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalError};

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle_core::bail!("paged-attention-decode-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != 1
            || key_batch != 1
            || value_batch != 1
            || q_heads != self.batch_size
            || q_len != 1
            || kv_heads != 1
            || value_kv_heads != 1
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle_core::bail!(
                "paged-attention-decode-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device();
        let dtype = match query.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => {
                candle_core::bail!("paged-attention-decode-megakernel unsupported dtype {other:?}")
            }
        };
        let out_shape = candle_core::Shape::from((1, self.batch_size, 1, self.head_dim));
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            query.dtype(),
            "paged-attention-decode-megakernel",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("paged-attention-decode-megakernel");
        candle_metal_kernels::call_full_attention_prefill(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            1,
            self.batch_size,
            1,
            1,
            self.kv_len,
            self.head_dim,
            self.batch_size,
            self.scale,
            self.seqlen_offset,
            candle_metal_kernels::BufferOffset {
                buffer: query.buffer(),
                offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: key.buffer(),
                offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: value.buffer(),
                offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(MetalError::from)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elem_count, query.dtype()),
            out_shape,
        ))
    }
}

#[cfg(feature = "candle-metal")]
fn paged_attention_decode_megakernel(
    queries: &candle_core::Tensor,
    key: &candle_core::Tensor,
    value: &candle_core::Tensor,
) -> Result<candle_core::Tensor> {
    let (batch_size, head_dim) = queries.dims2()?;
    let (kv_len, key_head_dim) = key.dims2()?;
    let (value_kv_len, value_head_dim) = value.dims2()?;
    if key_head_dim != head_dim || value_head_dim != head_dim || value_kv_len != kv_len {
        return Err(RuntimeError::DimensionMismatch {
            context: "paged decode megakernel shape",
            expected: kv_len * head_dim,
            got: value_kv_len * value_head_dim,
        });
    }
    let query = queries
        .contiguous()?
        .reshape((1, batch_size, 1, head_dim))?;
    let key = key.contiguous()?.reshape((1, 1, kv_len, head_dim))?;
    let value = value.contiguous()?.reshape((1, 1, kv_len, head_dim))?;
    Ok(query
        .apply_op3_no_bwd(
            &key,
            &value,
            &PagedAttentionDecodeMegakernel {
                batch_size,
                head_dim,
                kv_len,
                scale: attention_score_scale(head_dim),
                seqlen_offset: kv_len.saturating_sub(1),
            },
        )?
        .reshape((batch_size, head_dim))?)
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandleDeviceSelector {
    Cpu,
    Metal { ordinal: usize },
    Cuda { ordinal: usize },
    Hip { ordinal: usize },
}

#[cfg(feature = "candle")]
impl CandleDeviceSelector {
    pub fn slug(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Metal { ordinal } => format!("metal-{ordinal}"),
            Self::Cuda { ordinal } => format!("cuda-{ordinal}"),
            Self::Hip { ordinal } => format!("hip-{ordinal}"),
        }
    }

    pub fn resolve(&self) -> Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            Self::Metal { ordinal } => candle_core::Device::new_metal(*ordinal).map_err(Into::into),
            Self::Cuda { ordinal } => candle_core::Device::new_cuda(*ordinal).map_err(Into::into),
            Self::Hip { ordinal } => {
                #[cfg(feature = "candle-hip")]
                {
                    candle_core::Device::new_hip(*ordinal).map_err(Into::into)
                }
                #[cfg(not(feature = "candle-hip"))]
                {
                    let _ = ordinal;
                    Err(crate::RuntimeError::External {
                        context: "backend_device",
                        message:
                            "HIP support requires a Candle checkout with the hip backend enabled"
                                .to_string(),
                    })
                }
            }
        }
    }

    pub fn backend_device(&self) -> BackendDevice {
        match self {
            Self::Cpu => BackendDevice::Cpu,
            Self::Metal { ordinal } => BackendDevice::Metal { ordinal: *ordinal },
            Self::Cuda { ordinal } => BackendDevice::Cuda { ordinal: *ordinal },
            Self::Hip { ordinal } => BackendDevice::Hip { ordinal: *ordinal },
        }
    }
}

#[cfg(feature = "candle")]
impl std::fmt::Display for CandleDeviceSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Metal { ordinal } => {
                if *ordinal == 0 {
                    write!(f, "metal")
                } else {
                    write!(f, "metal:{ordinal}")
                }
            }
            Self::Cuda { ordinal } => {
                if *ordinal == 0 {
                    write!(f, "cuda")
                } else {
                    write!(f, "cuda:{ordinal}")
                }
            }
            Self::Hip { ordinal } => {
                if *ordinal == 0 {
                    write!(f, "hip")
                } else {
                    write!(f, "hip:{ordinal}")
                }
            }
        }
    }
}

#[cfg(feature = "candle")]
impl std::str::FromStr for CandleDeviceSelector {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        let (kind, ordinal) = match normalized.split_once(':') {
            Some((kind, ordinal)) => {
                let ordinal = ordinal
                    .parse::<usize>()
                    .map_err(|err| format!("invalid device ordinal in `{value}`: {err}"))?;
                (kind, ordinal)
            }
            None => (normalized.as_str(), 0),
        };

        match kind {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal { ordinal }),
            "cuda" => Ok(Self::Cuda { ordinal }),
            "hip" => Ok(Self::Hip { ordinal }),
            _ => Err(format!(
                "invalid device `{value}`, expected `cpu`, `metal[:ordinal]`, `cuda[:ordinal]`, or `hip[:ordinal]`"
            )),
        }
    }
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
pub struct CandlePreparedPage {
    key: candle_core::Tensor,
    value: candle_core::Tensor,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
struct CachedPreparedPage {
    prepared: CandlePreparedPage,
    head_dim: usize,
    last_access: u64,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
pub struct CandlePageBackend {
    device: candle_core::Device,
    selector: CandleDeviceSelector,
    prepared_pages: std::cell::RefCell<std::collections::HashMap<PageId, CachedPreparedPage>>,
    pinned_pages: std::cell::RefCell<std::collections::HashMap<PageId, usize>>,
    device_primary_pages: std::cell::RefCell<std::collections::HashSet<PageId>>,
    cache_hits: std::cell::Cell<usize>,
    cache_misses: std::cell::Cell<usize>,
    cache_evictions: std::cell::Cell<usize>,
    access_clock: std::cell::Cell<u64>,
    cache_page_budget: std::cell::Cell<Option<usize>>,
    attention_path: std::cell::Cell<AttentionPathMode>,
}

#[cfg(feature = "candle")]
impl CandlePageBackend {
    pub fn new(selector: CandleDeviceSelector) -> Result<Self> {
        let device = selector.resolve()?;
        Self::new_with_device(selector, device)
    }

    pub fn new_with_device(
        selector: CandleDeviceSelector,
        device: candle_core::Device,
    ) -> Result<Self> {
        let attention_path = AttentionPathMode::default_for_selector(&selector);
        Ok(Self {
            device,
            selector,
            prepared_pages: std::cell::RefCell::new(std::collections::HashMap::new()),
            pinned_pages: std::cell::RefCell::new(std::collections::HashMap::new()),
            device_primary_pages: std::cell::RefCell::new(std::collections::HashSet::new()),
            cache_hits: std::cell::Cell::new(0),
            cache_misses: std::cell::Cell::new(0),
            cache_evictions: std::cell::Cell::new(0),
            access_clock: std::cell::Cell::new(0),
            cache_page_budget: std::cell::Cell::new(None),
            attention_path: std::cell::Cell::new(attention_path),
        })
    }

    pub fn prepared_page_count(&self) -> usize {
        self.prepared_pages.borrow().len()
    }

    pub fn cache_hits(&self) -> usize {
        self.cache_hits.get()
    }

    pub fn cache_misses(&self) -> usize {
        self.cache_misses.get()
    }

    pub fn cache_evictions(&self) -> usize {
        self.cache_evictions.get()
    }

    pub fn prepare_cache_page_budget(&self) -> Option<usize> {
        self.cache_page_budget.get()
    }

    pub fn attention_path(&self) -> AttentionPathMode {
        self.attention_path.get()
    }

    pub fn set_attention_path(&self, path: AttentionPathMode) {
        self.attention_path.set(path);
    }

    pub fn set_prepare_cache_page_budget(&self, budget: Option<usize>) {
        self.cache_page_budget.set(budget);
        self.evict_to_budget();
    }

    pub fn pin_page(&self, page_id: PageId) {
        let mut pinned_pages = self.pinned_pages.borrow_mut();
        *pinned_pages.entry(page_id).or_insert(0) += 1;
    }

    pub fn pin_pages(&self, page_ids: &[PageId]) {
        for &page_id in page_ids {
            self.pin_page(page_id);
        }
    }

    pub fn unpin_page(&self, page_id: PageId) -> Result<()> {
        let mut pinned_pages = self.pinned_pages.borrow_mut();
        match pinned_pages.get_mut(&page_id) {
            Some(pin_count) if *pin_count > 1 => {
                *pin_count -= 1;
                Ok(())
            }
            Some(_) => {
                pinned_pages.remove(&page_id);
                self.evict_to_budget();
                Ok(())
            }
            None => Err(RuntimeError::External {
                context: "candle_backend",
                message: format!("page {page_id} is not pinned"),
            }),
        }
    }

    pub fn unpin_pages(&self, page_ids: &[PageId]) -> Result<()> {
        for &page_id in page_ids {
            self.unpin_page(page_id)?;
        }
        Ok(())
    }

    pub fn pinned_page_count(&self) -> usize {
        self.pinned_pages.borrow().len()
    }

    pub fn is_page_pinned(&self, page_id: PageId) -> bool {
        self.pinned_pages.borrow().contains_key(&page_id)
    }

    pub fn is_page_prepared(&self, page_id: PageId) -> bool {
        self.prepared_pages.borrow().contains_key(&page_id)
    }

    pub fn can_promote_page_device_primary(&self) -> bool {
        !matches!(self.selector, CandleDeviceSelector::Cpu)
            && self.cache_page_budget.get().is_none()
    }

    pub fn mark_page_device_primary(&self, page_id: PageId) {
        self.device_primary_pages.borrow_mut().insert(page_id);
    }

    pub fn is_page_device_primary(&self, page_id: PageId) -> bool {
        self.device_primary_pages.borrow().contains(&page_id)
    }

    pub fn ensure_page_resident(&self, page_id: PageId, page: &KvPage) -> Result<bool> {
        if !page.sealed {
            return Ok(false);
        }
        if let Some(prepared) = self.prepared_pages.borrow_mut().get_mut(&page_id) {
            prepared.last_access = self.next_access_clock();
            return Ok(false);
        }

        let prepared = self.build_prepared_page(page)?;
        self.cache_misses.set(self.cache_misses.get() + 1);
        self.cache_prepared_page(page_id, prepared);
        Ok(self.is_page_prepared(page_id))
    }

    pub fn release_page(&self, page_id: PageId) -> bool {
        self.pinned_pages.borrow_mut().remove(&page_id);
        self.device_primary_pages.borrow_mut().remove(&page_id);
        self.prepared_pages.borrow_mut().remove(&page_id).is_some()
    }

    pub fn release_pages(&self, page_ids: &[PageId]) -> usize {
        page_ids
            .iter()
            .filter(|&&page_id| self.release_page(page_id))
            .count()
    }

    pub fn reset_page_state(&self) {
        self.prepared_pages.borrow_mut().clear();
        self.pinned_pages.borrow_mut().clear();
        self.device_primary_pages.borrow_mut().clear();
        self.cache_hits.set(0);
        self.cache_misses.set(0);
        self.cache_evictions.set(0);
        self.access_clock.set(0);
    }

    fn next_access_clock(&self) -> u64 {
        let next = self.access_clock.get() + 1;
        self.access_clock.set(next);
        next
    }

    fn build_prepared_page(&self, page: &KvPage) -> Result<CachedPreparedPage> {
        let token_count = page.token_len();
        let head_dim = page.head_dim_usize();
        let key = page.dense_key_storage_f32();
        let value = page.dense_value_storage_f32();
        Ok(CachedPreparedPage {
            prepared: CandlePreparedPage {
                key: candle_core::Tensor::from_vec(key, (token_count, head_dim), &self.device)?,
                value: candle_core::Tensor::from_vec(value, (token_count, head_dim), &self.device)?,
            },
            head_dim,
            last_access: 0,
        })
    }

    fn cache_prepared_page(&self, page_id: PageId, mut prepared: CachedPreparedPage) {
        prepared.last_access = self.next_access_clock();
        let Some(budget) = self.cache_page_budget.get() else {
            self.prepared_pages.borrow_mut().insert(page_id, prepared);
            return;
        };

        if budget == 0 {
            return;
        }

        self.evict_for_insert(budget);
        if self.prepared_pages.borrow().len() >= budget {
            return;
        }

        self.prepared_pages.borrow_mut().insert(page_id, prepared);
    }

    fn evict_for_insert(&self, budget: usize) {
        while self.prepared_pages.borrow().len() >= budget {
            if !self.evict_one_unpinned_page() {
                break;
            }
        }
    }

    fn evict_to_budget(&self) {
        let Some(budget) = self.cache_page_budget.get() else {
            return;
        };

        if budget == 0 {
            let evicted = self.prepared_pages.borrow().len();
            self.prepared_pages.borrow_mut().clear();
            self.cache_evictions
                .set(self.cache_evictions.get() + evicted);
            return;
        }

        while self.prepared_pages.borrow().len() > budget {
            if !self.evict_one_unpinned_page() {
                break;
            }
        }
    }

    fn evict_one_unpinned_page(&self) -> bool {
        let pinned_pages = self.pinned_pages.borrow();
        let device_primary_pages = self.device_primary_pages.borrow();
        let eviction_candidate = self
            .prepared_pages
            .borrow()
            .iter()
            .filter(|(page_id, _)| {
                !pinned_pages.contains_key(page_id) && !device_primary_pages.contains(page_id)
            })
            .min_by_key(|(_, cached_page)| cached_page.last_access)
            .map(|(page_id, _)| *page_id);
        drop(pinned_pages);
        drop(device_primary_pages);

        if let Some(page_id) = eviction_candidate {
            self.prepared_pages.borrow_mut().remove(&page_id);
            self.cache_evictions.set(self.cache_evictions.get() + 1);
            true
        } else {
            false
        }
    }

    fn prepare_sequence(
        &self,
        store: &PageStore,
        page_ids: &[PageId],
        expected_head_dim: usize,
    ) -> Result<CandlePreparedPage> {
        let mut key_tensors = Vec::with_capacity(page_ids.len());
        let mut value_tensors = Vec::with_capacity(page_ids.len());

        for &page_id in page_ids {
            let prepared = if let Some(prepared) = self.prepare_cached(page_id, expected_head_dim)? {
                prepared
            } else {
                let page = store.page(page_id)?;
                if !page.is_exact_fused_compatible() {
                    return Err(RuntimeError::FusedAttentionRequiresExactPages {
                        page_id,
                        key_mode: page.key_mode().describe(),
                        value_mode: page.value_mode().describe(),
                    });
                }
                page.validate_layout(page_id)?;
                if page.head_dim_usize() != expected_head_dim {
                    return Err(RuntimeError::DimensionMismatch {
                        context: "fused query",
                        expected: page.head_dim_usize(),
                        got: expected_head_dim,
                    });
                }
                self.prepare(page_id, page)?
            };
            key_tensors.push(prepared.key);
            value_tensors.push(prepared.value);
        }

        if key_tensors.len() == 1 {
            return Ok(CandlePreparedPage {
                key: key_tensors.pop().expect("single prepared key"),
                value: value_tensors.pop().expect("single prepared value"),
            });
        }

        let key_refs = key_tensors.iter().collect::<Vec<_>>();
        let value_refs = value_tensors.iter().collect::<Vec<_>>();
        Ok(CandlePreparedPage {
            key: candle_core::Tensor::cat(&key_refs, 0)?,
            value: candle_core::Tensor::cat(&value_refs, 0)?,
        })
    }

    pub fn decode_tensor_fused(
        &self,
        store: &PageStore,
        page_ids: &[PageId],
        queries: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        if self.attention_path.get() != AttentionPathMode::Fused {
            return Err(RuntimeError::External {
                context: "candle_backend",
                message: "decode_tensor_fused requires fused attention path".to_string(),
            });
        }
        if page_ids.is_empty() {
            return Err(RuntimeError::EmptyDecode);
        }
        let (batch_size, head_dim) = queries.dims2()?;
        let prepared = self.prepare_sequence(store, page_ids, head_dim)?;
        #[cfg(feature = "candle-metal")]
        if matches!(self.selector, CandleDeviceSelector::Metal { .. }) {
            let mixed = paged_attention_decode_megakernel(queries, &prepared.key, &prepared.value)?;
            let (out_batch, out_dim) = mixed.dims2()?;
            if out_batch != batch_size || out_dim != head_dim {
                return Err(RuntimeError::DimensionMismatch {
                    context: "decode_tensor_fused metal output",
                    expected: batch_size * head_dim,
                    got: out_batch * out_dim,
                });
            }
            return Ok(mixed);
        }
        #[cfg(feature = "candle-cuda")]
        if matches!(self.selector, CandleDeviceSelector::Cuda { .. }) {
            let mixed = candle_transformers::models::qwen3_5::paged_attention_decode_megakernel(
                queries,
                &prepared.key,
                &prepared.value,
            )?;
            let (out_batch, out_dim) = mixed.dims2()?;
            if out_batch != batch_size || out_dim != head_dim {
                return Err(RuntimeError::DimensionMismatch {
                    context: "decode_tensor_fused cuda output",
                    expected: batch_size * head_dim,
                    got: out_batch * out_dim,
                });
            }
            return Ok(mixed);
        }
        let logits = queries.matmul(&prepared.key.transpose(0, 1)?)?;
        let logits = (logits * attention_score_scale(head_dim) as f64)?;
        let logits = candle_nn::ops::softmax_last_dim(&logits)?;
        let mixed = logits.matmul(&prepared.value)?;
        let (out_batch, out_dim) = mixed.dims2()?;
        if out_batch != batch_size || out_dim != head_dim {
            return Err(RuntimeError::DimensionMismatch {
                context: "decode_tensor_fused output",
                expected: batch_size * head_dim,
                got: out_batch * out_dim,
            });
        }
        Ok(mixed)
    }
}

#[cfg(feature = "candle")]
impl PageBackend for CandlePageBackend {
    type Prepared<'a>
        = CandlePreparedPage
    where
        Self: 'a;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "candle",
            device: self.selector.backend_device(),
            supports_prepare_cache: true,
            supports_virtual_pages: true,
            supports_device_resident_pages: true,
        }
    }

    fn prepare<'a>(&self, page_id: PageId, page: &'a KvPage) -> Result<Self::Prepared<'a>> {
        if page.sealed {
            if let Some(prepared) = self.prepared_pages.borrow_mut().get_mut(&page_id) {
                prepared.last_access = self.next_access_clock();
                self.cache_hits.set(self.cache_hits.get() + 1);
                return Ok(prepared.prepared.clone());
            }
        }

        let prepared = self.build_prepared_page(page)?;
        self.cache_misses.set(self.cache_misses.get() + 1);

        if page.sealed {
            self.cache_prepared_page(page_id, prepared.clone());
        }

        Ok(prepared.prepared)
    }

    fn prepare_cached<'a>(
        &'a self,
        page_id: PageId,
        expected_head_dim: usize,
    ) -> Result<Option<Self::Prepared<'a>>> {
        let mut prepared_pages = self.prepared_pages.borrow_mut();
        let Some(prepared) = prepared_pages.get_mut(&page_id) else {
            return Ok(None);
        };
        if prepared.head_dim != expected_head_dim {
            return Err(RuntimeError::DimensionMismatch {
                context: "cached query",
                expected: prepared.head_dim,
                got: expected_head_dim,
            });
        }
        prepared.last_access = self.next_access_clock();
        self.cache_hits.set(self.cache_hits.get() + 1);
        Ok(Some(prepared.prepared.clone()))
    }

    fn decode_fused(
        &self,
        store: &PageStore,
        page_ids: &[PageId],
        q: &[f32],
        out: &mut [f32],
    ) -> Result<bool> {
        if self.attention_path.get() != AttentionPathMode::Fused {
            return Ok(false);
        }
        if page_ids.is_empty() {
            return Err(RuntimeError::EmptyDecode);
        }
        if out.len() != q.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "decode output",
                expected: q.len(),
                got: out.len(),
            });
        }

        out.fill(0.0);
        let prepared = self.prepare_sequence(store, page_ids, q.len())?;
        let query = candle_core::Tensor::from_slice(q, (1, q.len()), &self.device)?;
        let mut logits = query
            .matmul(&prepared.key.transpose(0, 1)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        apply_softmax_in_place(&mut logits, attention_score_scale(q.len()));
        let weights = candle_core::Tensor::from_slice(&logits, (1, logits.len()), &self.device)?;
        let mixed = weights
            .matmul(&prepared.value)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (out_value, value) in out.iter_mut().zip(mixed.into_iter()) {
            *out_value += value;
        }
        Ok(true)
    }

    fn decode_batch_fused(
        &self,
        store: &PageStore,
        page_ids_by_query: &[&[PageId]],
        queries: &[&[f32]],
        outputs: &mut [Vec<f32>],
    ) -> Result<bool> {
        if self.attention_path.get() != AttentionPathMode::Fused {
            return Ok(false);
        }
        if queries.len() != page_ids_by_query.len() || queries.len() != outputs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "fused batch query count",
                expected: queries.len(),
                got: page_ids_by_query.len().max(outputs.len()),
            });
        }
        if queries.is_empty() {
            return Ok(true);
        }

        let head_dim = queries[0].len();
        let mut grouped_queries = std::collections::BTreeMap::<Vec<PageId>, Vec<usize>>::new();
        for (query_index, (query, page_ids)) in
            queries.iter().zip(page_ids_by_query.iter()).enumerate()
        {
            if page_ids.is_empty() {
                return Err(RuntimeError::EmptyDecode);
            }
            if query.len() != head_dim {
                return Err(RuntimeError::DimensionMismatch {
                    context: "fused batch query",
                    expected: head_dim,
                    got: query.len(),
                });
            }
            outputs[query_index].fill(0.0);
            grouped_queries
                .entry((*page_ids).to_vec())
                .or_default()
                .push(query_index);
        }

        for (page_ids, query_indices) in grouped_queries {
            let prepared = self.prepare_sequence(store, &page_ids, head_dim)?;
            let flat_queries = query_indices
                .iter()
                .flat_map(|&query_index| queries[query_index].iter().copied())
                .collect::<Vec<_>>();
            let query = candle_core::Tensor::from_slice(
                &flat_queries,
                (query_indices.len(), head_dim),
                &self.device,
            )?;
            let mut logits = query
                .matmul(&prepared.key.transpose(0, 1)?)?
                .to_vec2::<f32>()?;
            for row in &mut logits {
                apply_softmax_in_place(row, attention_score_scale(head_dim));
            }
            let token_count = logits.first().map(|row| row.len()).unwrap_or(0);
            let flat_weights = logits.into_iter().flatten().collect::<Vec<_>>();
            let weights = candle_core::Tensor::from_slice(
                &flat_weights,
                (query_indices.len(), token_count),
                &self.device,
            )?;
            let mixed = weights.matmul(&prepared.value)?.to_vec2::<f32>()?;
            for (batch_index, mixed_row) in mixed.into_iter().enumerate() {
                outputs[query_indices[batch_index]].copy_from_slice(&mixed_row);
            }
        }

        Ok(true)
    }

    fn score(&self, q: &[f32], page: &Self::Prepared<'_>, logits_out: &mut Vec<f32>) -> Result<()> {
        let query = candle_core::Tensor::from_slice(q, (1, q.len()), &self.device)?;
        let logits = query
            .matmul(&page.key.transpose(0, 1)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        logits_out.extend(logits);
        Ok(())
    }

    fn mix(&self, weights: &[f32], page: &Self::Prepared<'_>, out: &mut [f32]) -> Result<()> {
        let weights = candle_core::Tensor::from_slice(weights, (1, weights.len()), &self.device)?;
        let mixed = weights
            .matmul(&page.value)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (out_value, value) in out.iter_mut().zip(mixed.into_iter()) {
            *out_value += value;
        }
        Ok(())
    }

    fn score_batch(
        &self,
        queries: &[&[f32]],
        page: &Self::Prepared<'_>,
        logits_outs: &mut [Vec<f32>],
    ) -> Result<()> {
        if queries.len() != logits_outs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "score batch outputs",
                expected: queries.len(),
                got: logits_outs.len(),
            });
        }
        if queries.is_empty() {
            return Ok(());
        }

        let head_dim = queries[0].len();
        for query in queries.iter().skip(1) {
            if query.len() != head_dim {
                return Err(RuntimeError::DimensionMismatch {
                    context: "score batch query",
                    expected: head_dim,
                    got: query.len(),
                });
            }
        }

        let flat_queries = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        let query = candle_core::Tensor::from_slice(
            &flat_queries,
            (queries.len(), head_dim),
            &self.device,
        )?;
        let logits = query.matmul(&page.key.transpose(0, 1)?)?.to_vec2::<f32>()?;
        for (logits_out, page_logits) in logits_outs.iter_mut().zip(logits.into_iter()) {
            logits_out.extend(page_logits);
        }

        Ok(())
    }

    fn mix_batch(
        &self,
        weights_batch: &[&[f32]],
        page: &Self::Prepared<'_>,
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        if weights_batch.len() != outs.len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "mix batch outputs",
                expected: weights_batch.len(),
                got: outs.len(),
            });
        }
        if weights_batch.is_empty() {
            return Ok(());
        }

        let token_count = weights_batch[0].len();
        let head_dim = outs[0].len();
        for weights in weights_batch.iter().skip(1) {
            if weights.len() != token_count {
                return Err(RuntimeError::DimensionMismatch {
                    context: "mix batch weights",
                    expected: token_count,
                    got: weights.len(),
                });
            }
        }
        for out in outs.iter().skip(1) {
            if out.len() != head_dim {
                return Err(RuntimeError::DimensionMismatch {
                    context: "mix batch output",
                    expected: head_dim,
                    got: out.len(),
                });
            }
        }

        let flat_weights = weights_batch
            .iter()
            .flat_map(|weights| weights.iter().copied())
            .collect::<Vec<_>>();
        let weights = candle_core::Tensor::from_slice(
            &flat_weights,
            (weights_batch.len(), token_count),
            &self.device,
        )?;
        let mixed = weights.matmul(&page.value)?.to_vec2::<f32>()?;
        for (out, mixed_row) in outs.iter_mut().zip(mixed.into_iter()) {
            for (out_value, value) in out.iter_mut().zip(mixed_row.into_iter()) {
                *out_value += value;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "candle")]
fn attention_score_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

#[cfg(feature = "candle")]
fn apply_softmax_in_place(logits: &mut [f32], score_scale: f32) {
    if logits.is_empty() {
        return;
    }

    for logit in logits.iter_mut() {
        *logit *= score_scale;
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_logit).exp();
        denom += *logit;
    }

    if denom > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= denom;
        }
    }
}
