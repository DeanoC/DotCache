use crate::page::PageId;
use crate::virtual_page::{
    AppendPageResult, VirtualCacheMetrics, VirtualPageId, VirtualPagedKvCache, VirtualSeqCache,
};
use crate::{Result, RuntimeError};

#[cfg(feature = "candle")]
#[derive(Clone, Debug)]
pub enum HybridCacheState {
    Qwen35(candle_transformers::models::qwen3_5::CacheState),
}

pub type SessionId = usize;

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionRequestKind {
    Prefill,
    Decode,
    BatchDecode,
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionMetrics {
    pub request_count: usize,
    pub prefill_request_count: usize,
    pub decode_request_count: usize,
    pub batch_decode_request_count: usize,
    pub input_token_count: usize,
    pub spill_count: usize,
    pub restore_count: usize,
    pub spilled_bytes: usize,
    pub restored_bytes: usize,
    pub cooldown_hit_count: usize,
}

impl SessionMetrics {
    pub(crate) fn record_request(
        &mut self,
        kind: SessionRequestKind,
        input_token_count: usize,
        cache_delta: &VirtualCacheMetrics,
    ) {
        self.request_count += 1;
        self.input_token_count += input_token_count;
        match kind {
            SessionRequestKind::Prefill => self.prefill_request_count += 1,
            SessionRequestKind::Decode => self.decode_request_count += 1,
            SessionRequestKind::BatchDecode => self.batch_decode_request_count += 1,
        }
        self.spill_count += cache_delta.spill_count;
        self.restore_count += cache_delta.restore_count;
        self.spilled_bytes += cache_delta.spilled_bytes;
        self.restored_bytes += cache_delta.restored_bytes;
        self.cooldown_hit_count += cache_delta.cooldown_hit_count;
    }
}

#[derive(Clone, Debug)]
pub struct SessionPrefix {
    prompt_len: usize,
    token_count: usize,
    next_position: u32,
    seq: VirtualSeqCache,
    #[cfg(feature = "candle")]
    hybrid_cache_state: Option<HybridCacheState>,
}

impl SessionPrefix {
    fn new(
        prompt_len: usize,
        token_count: usize,
        next_position: u32,
        seq: VirtualSeqCache,
    ) -> Self {
        Self {
            prompt_len,
            token_count,
            next_position,
            seq,
            #[cfg(feature = "candle")]
            hybrid_cache_state: None,
        }
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    pub fn virtual_seq(&self) -> &VirtualSeqCache {
        &self.seq
    }

    #[cfg(feature = "candle")]
    pub fn hybrid_cache_state(&self) -> Option<&HybridCacheState> {
        self.hybrid_cache_state.as_ref()
    }

    #[cfg(feature = "candle")]
    pub fn set_hybrid_cache_state(&mut self, state: Option<HybridCacheState>) {
        self.hybrid_cache_state = state;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerDecodePlan {
    layer: usize,
    kv_heads: Vec<Vec<PageId>>,
}

impl LayerDecodePlan {
    fn new(layer: usize, kv_heads: Vec<Vec<PageId>>) -> Self {
        Self { layer, kv_heads }
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn kv_head_count(&self) -> usize {
        self.kv_heads.len()
    }

    pub fn page_ids(&self, kv_head: usize) -> Result<&[PageId]> {
        self.kv_heads
            .get(kv_head)
            .map(|page_ids| page_ids.as_slice())
            .ok_or(RuntimeError::InvalidKvHead {
                kv_head,
                kv_head_count: self.kv_heads.len(),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionDecodePlan {
    session_id: SessionId,
    layers: Vec<LayerDecodePlan>,
}

impl SessionDecodePlan {
    fn new(session_id: SessionId, layers: Vec<LayerDecodePlan>) -> Self {
        Self { session_id, layers }
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer: usize) -> Result<&LayerDecodePlan> {
        self.layers.get(layer).ok_or(RuntimeError::InvalidLayer {
            layer,
            layer_count: self.layers.len(),
        })
    }

    pub fn layers(&self) -> &[LayerDecodePlan] {
        self.layers.as_slice()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvRow {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

impl KvRow {
    pub fn new(key: Vec<f32>, value: Vec<f32>) -> Self {
        Self { key, value }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTokenRows {
    pub layers: Vec<Vec<KvRow>>,
}

impl SessionTokenRows {
    pub fn new(layers: Vec<Vec<KvRow>>) -> Self {
        Self { layers }
    }

    fn validate(
        &self,
        expected_layer_count: usize,
        expected_kv_head_count: usize,
        expected_head_dim: usize,
    ) -> Result<()> {
        if self.layers.len() != expected_layer_count {
            return Err(RuntimeError::InvalidLayer {
                layer: self.layers.len(),
                layer_count: expected_layer_count,
            });
        }

        for layer_rows in &self.layers {
            if layer_rows.len() != expected_kv_head_count {
                return Err(RuntimeError::InvalidKvHead {
                    kv_head: layer_rows.len(),
                    kv_head_count: expected_kv_head_count,
                });
            }

            for row in layer_rows {
                if row.key.len() != expected_head_dim {
                    return Err(RuntimeError::DimensionMismatch {
                        context: "session token key row",
                        expected: expected_head_dim,
                        got: row.key.len(),
                    });
                }
                if row.value.len() != expected_head_dim {
                    return Err(RuntimeError::DimensionMismatch {
                        context: "session token value row",
                        expected: expected_head_dim,
                        got: row.value.len(),
                    });
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SessionState {
    id: SessionId,
    prompt_len: usize,
    token_count: usize,
    next_position: u32,
    seq: VirtualSeqCache,
    metrics: SessionMetrics,
    #[cfg(feature = "candle")]
    hybrid_cache_state: Option<HybridCacheState>,
}

impl SessionState {
    fn new(id: SessionId, prompt_len: usize, seq: VirtualSeqCache) -> Self {
        Self {
            id,
            prompt_len,
            token_count: 0,
            next_position: 0,
            seq,
            metrics: SessionMetrics::default(),
            #[cfg(feature = "candle")]
            hybrid_cache_state: None,
        }
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    pub fn virtual_seq(&self) -> &VirtualSeqCache {
        &self.seq
    }

    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    #[cfg(feature = "candle")]
    pub fn hybrid_cache_state(&self) -> Option<&HybridCacheState> {
        self.hybrid_cache_state.as_ref()
    }
}

#[derive(Debug)]
pub struct SessionRuntime {
    cache: VirtualPagedKvCache,
    sessions: Vec<Option<SessionState>>,
}

impl SessionRuntime {
    pub fn new(
        layer_count: usize,
        kv_head_count: usize,
        tokens_per_page: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            cache: VirtualPagedKvCache::new(layer_count, kv_head_count, tokens_per_page, head_dim),
            sessions: Vec::new(),
        }
    }

    pub fn cache(&self) -> &VirtualPagedKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut VirtualPagedKvCache {
        &mut self.cache
    }

    pub fn session_count(&self) -> usize {
        self.sessions
            .iter()
            .filter(|session| session.is_some())
            .count()
    }

    pub fn create_session(&mut self) -> SessionId {
        self.create_session_with_prompt_len(0)
    }

    pub fn create_session_with_prompt_len(&mut self, prompt_len: usize) -> SessionId {
        let session_id = self.sessions.len();
        let seq = VirtualSeqCache::new(self.cache.layer_count(), self.cache.kv_head_count());
        self.sessions
            .push(Some(SessionState::new(session_id, prompt_len, seq)));
        session_id
    }

    pub fn fork_session(&mut self, source_session_id: SessionId) -> Result<SessionId> {
        let source = self.session(source_session_id)?.clone();
        let seq = self.cache.fork_seq(source.virtual_seq())?;
        let session_id = self.sessions.len();
        self.sessions.push(Some(SessionState {
            id: session_id,
            prompt_len: source.prompt_len,
            token_count: source.token_count,
            next_position: source.next_position,
            seq,
            metrics: SessionMetrics::default(),
            #[cfg(feature = "candle")]
            hybrid_cache_state: source.hybrid_cache_state.clone(),
        }));
        Ok(session_id)
    }

    pub fn capture_prefix(&mut self, session_id: SessionId) -> Result<SessionPrefix> {
        let source = self.session(session_id)?.clone();
        let seq = self.cache.fork_seq(source.virtual_seq())?;
        let mut prefix = SessionPrefix::new(
            source.token_count,
            source.token_count,
            source.next_position,
            seq,
        );
        #[cfg(feature = "candle")]
        {
            prefix.hybrid_cache_state = source.hybrid_cache_state.clone();
        }
        Ok(prefix)
    }

    pub fn attach_prefix(&mut self, prefix: &SessionPrefix) -> Result<SessionId> {
        validate_prefix_shape(
            prefix.virtual_seq(),
            self.cache.layer_count(),
            self.cache.kv_head_count(),
        )?;

        let seq = self.cache.fork_seq(prefix.virtual_seq())?;
        let session_id = self.sessions.len();
        self.sessions.push(Some(SessionState {
            id: session_id,
            prompt_len: prefix.prompt_len,
            token_count: prefix.token_count,
            next_position: prefix.next_position,
            seq,
            metrics: SessionMetrics::default(),
            #[cfg(feature = "candle")]
            hybrid_cache_state: prefix.hybrid_cache_state.clone(),
        }));
        Ok(session_id)
    }

    pub fn release_prefix(&mut self, prefix: &SessionPrefix) -> Result<Vec<PageId>> {
        self.cache.release_seq(prefix.virtual_seq())
    }

    pub fn close_session(&mut self, session_id: SessionId) -> Result<Vec<PageId>> {
        let session_count = self.session_count();
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?
            .take()
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?;
        self.cache.release_seq(session.virtual_seq())
    }

    pub fn session(&self, session_id: SessionId) -> Result<&SessionState> {
        self.sessions
            .get(session_id)
            .and_then(|session| session.as_ref())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count: self.session_count(),
            })
    }

    pub fn current_position(&self, session_id: SessionId) -> Result<u32> {
        Ok(self.session(session_id)?.next_position())
    }

    pub fn session_metrics(&self, session_id: SessionId) -> Result<&SessionMetrics> {
        Ok(self.session(session_id)?.metrics())
    }

    pub fn reset_session_metrics(&mut self, session_id: SessionId) -> Result<()> {
        let session_count = self.sessions.len();
        let session = self
            .sessions
            .get_mut(session_id)
            .and_then(|session| session.as_mut())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?;
        session.metrics = SessionMetrics::default();
        Ok(())
    }

    #[cfg(feature = "candle")]
    pub fn set_hybrid_cache_state(
        &mut self,
        session_id: SessionId,
        state: Option<HybridCacheState>,
    ) -> Result<()> {
        let session_count = self.sessions.len();
        let session = self
            .sessions
            .get_mut(session_id)
            .and_then(|session| session.as_mut())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?;
        session.hybrid_cache_state = state;
        Ok(())
    }

    #[cfg(feature = "candle")]
    pub fn hybrid_cache_state(&self, session_id: SessionId) -> Result<Option<&HybridCacheState>> {
        Ok(self.session(session_id)?.hybrid_cache_state.as_ref())
    }

    pub fn record_session_request(
        &mut self,
        session_ids: &[SessionId],
        kind: SessionRequestKind,
        input_token_counts: &[usize],
        cache_delta: &VirtualCacheMetrics,
    ) -> Result<()> {
        if session_ids.len() != input_token_counts.len() {
            return Err(RuntimeError::External {
                context: "session_runtime",
                message: format!(
                    "session_ids length {} did not match input_token_counts length {}",
                    session_ids.len(),
                    input_token_counts.len()
                ),
            });
        }
        if session_ids.is_empty() {
            return Ok(());
        }

        let deltas = apportion_cache_delta(cache_delta, session_ids.len());
        for ((&session_id, &input_token_count), cache_delta) in session_ids
            .iter()
            .zip(input_token_counts.iter())
            .zip(deltas.iter())
        {
            let session_count = self.sessions.len();
            let session = self
                .sessions
                .get_mut(session_id)
                .and_then(|session| session.as_mut())
                .ok_or(RuntimeError::InvalidSessionId {
                    session_id,
                    session_count,
                })?;
            session
                .metrics
                .record_request(kind, input_token_count, cache_delta);
        }
        Ok(())
    }

    pub fn append_kv_row_at(
        &mut self,
        session_id: SessionId,
        layer: usize,
        kv_head: usize,
        position: u32,
        key: &[f32],
        value: &[f32],
    ) -> Result<AppendPageResult> {
        if key.len() != self.cache.head_dim() {
            return Err(RuntimeError::DimensionMismatch {
                context: "session kv row key",
                expected: self.cache.head_dim(),
                got: key.len(),
            });
        }
        if value.len() != self.cache.head_dim() {
            return Err(RuntimeError::DimensionMismatch {
                context: "session kv row value",
                expected: self.cache.head_dim(),
                got: value.len(),
            });
        }

        let (cache, sessions) = (&mut self.cache, &mut self.sessions);
        let session_count = sessions.len();
        let session = sessions
            .get_mut(session_id)
            .and_then(|session| session.as_mut())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?;
        cache.append_token_to(&mut session.seq, layer, kv_head, position, key, value)
    }

    pub fn commit_positions(
        &mut self,
        session_id: SessionId,
        start_position: u32,
        token_count: usize,
    ) -> Result<()> {
        let session_count = self.sessions.len();
        let session = self
            .sessions
            .get_mut(session_id)
            .and_then(|session| session.as_mut())
            .ok_or(RuntimeError::InvalidSessionId {
                session_id,
                session_count,
            })?;
        if session.next_position != start_position {
            return Err(RuntimeError::PositionMismatch {
                expected: session.next_position,
                got: start_position,
            });
        }

        let advance = u32::try_from(token_count).map_err(|_| RuntimeError::ConversionOverflow {
            field: "session token_count",
            value: token_count,
        })?;
        session.token_count += token_count;
        session.next_position += advance;
        Ok(())
    }

    pub fn append_token(
        &mut self,
        session_id: SessionId,
        token_rows: &SessionTokenRows,
    ) -> Result<u32> {
        token_rows.validate(
            self.cache.layer_count(),
            self.cache.kv_head_count(),
            self.cache.head_dim(),
        )?;
        let position = self.current_position(session_id)?;
        for (layer, layer_rows) in token_rows.layers.iter().enumerate() {
            for (kv_head, row) in layer_rows.iter().enumerate() {
                self.append_kv_row_at(session_id, layer, kv_head, position, &row.key, &row.value)?;
            }
        }
        self.commit_positions(session_id, position, 1)?;
        Ok(position)
    }

    pub fn virtual_page_ids(
        &self,
        session_id: SessionId,
        layer: usize,
        kv_head: usize,
    ) -> Result<&[VirtualPageId]> {
        let session = self.session(session_id)?;
        if layer >= session.seq.layers.len() {
            return Err(RuntimeError::InvalidLayer {
                layer,
                layer_count: session.seq.layers.len(),
            });
        }
        let kv_head_count = session.seq.layers[layer].virtual_pages_by_kv_head.len();
        if kv_head >= kv_head_count {
            return Err(RuntimeError::InvalidKvHead {
                kv_head,
                kv_head_count,
            });
        }
        Ok(session.seq.layers[layer].virtual_pages_by_kv_head[kv_head].as_slice())
    }

    pub fn resolve_physical_page_ids(
        &self,
        session_id: SessionId,
        layer: usize,
        kv_head: usize,
    ) -> Result<Vec<PageId>> {
        let virtual_page_ids = self.virtual_page_ids(session_id, layer, kv_head)?;
        self.cache.virtual_table().resolve(virtual_page_ids)
    }

    pub fn sealed_physical_page_ids(&self, session_id: SessionId) -> Result<Vec<PageId>> {
        let seq = self.session(session_id)?.virtual_seq();
        self.sealed_physical_page_ids_for_seq(seq)
    }

    pub fn physical_page_ids(&self, session_id: SessionId) -> Result<Vec<PageId>> {
        let seq = self.session(session_id)?.virtual_seq();
        self.physical_page_ids_for_seq(seq)
    }

    pub fn sealed_physical_page_ids_for_prefix(
        &self,
        prefix: &SessionPrefix,
    ) -> Result<Vec<PageId>> {
        self.sealed_physical_page_ids_for_seq(prefix.virtual_seq())
    }

    pub fn physical_page_ids_for_prefix(&self, prefix: &SessionPrefix) -> Result<Vec<PageId>> {
        self.physical_page_ids_for_seq(prefix.virtual_seq())
    }

    pub fn plan_layer_decode(
        &self,
        session_id: SessionId,
        layer: usize,
    ) -> Result<LayerDecodePlan> {
        let session = self.session(session_id)?;
        let layer_cache = session
            .seq
            .layers
            .get(layer)
            .ok_or(RuntimeError::InvalidLayer {
                layer,
                layer_count: session.seq.layers.len(),
            })?;
        let mut kv_heads = Vec::with_capacity(layer_cache.virtual_pages_by_kv_head.len());
        for virtual_page_ids in &layer_cache.virtual_pages_by_kv_head {
            kv_heads.push(self.cache.virtual_table().resolve(virtual_page_ids)?);
        }
        Ok(LayerDecodePlan::new(layer, kv_heads))
    }

    pub fn plan_session_decode(&self, session_id: SessionId) -> Result<SessionDecodePlan> {
        let layer_count = self.session(session_id)?.seq.layers.len();
        let mut layers = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            layers.push(self.plan_layer_decode(session_id, layer)?);
        }
        Ok(SessionDecodePlan::new(session_id, layers))
    }

    pub fn plan_sessions_decode(
        &self,
        session_ids: &[SessionId],
    ) -> Result<Vec<SessionDecodePlan>> {
        session_ids
            .iter()
            .map(|&session_id| self.plan_session_decode(session_id))
            .collect()
    }

    pub fn plan_sessions_layer_decode(
        &self,
        session_ids: &[SessionId],
        layer: usize,
    ) -> Result<Vec<LayerDecodePlan>> {
        session_ids
            .iter()
            .map(|&session_id| self.plan_layer_decode(session_id, layer))
            .collect()
    }

    fn sealed_physical_page_ids_for_seq(&self, seq: &VirtualSeqCache) -> Result<Vec<PageId>> {
        self.collect_physical_page_ids(seq, true)
    }

    fn physical_page_ids_for_seq(&self, seq: &VirtualSeqCache) -> Result<Vec<PageId>> {
        self.collect_physical_page_ids(seq, false)
    }

    fn collect_physical_page_ids(
        &self,
        seq: &VirtualSeqCache,
        sealed_only: bool,
    ) -> Result<Vec<PageId>> {
        let mut page_ids = std::collections::BTreeSet::new();
        for layer in &seq.layers {
            for virtual_page_ids in &layer.virtual_pages_by_kv_head {
                for &virtual_page_id in virtual_page_ids {
                    let virtual_page = self.cache.virtual_table().virtual_page(virtual_page_id)?;
                    if !sealed_only || virtual_page.sealed {
                        page_ids.insert(virtual_page.physical_page_id);
                    }
                }
            }
        }
        Ok(page_ids.into_iter().collect())
    }
}

fn apportion_cache_delta(
    cache_delta: &VirtualCacheMetrics,
    count: usize,
) -> Vec<VirtualCacheMetrics> {
    fn split(value: usize, count: usize) -> Vec<usize> {
        let base = value / count;
        let remainder = value % count;
        (0..count)
            .map(|index| base + usize::from(index < remainder))
            .collect()
    }

    let spill_counts = split(cache_delta.spill_count, count);
    let restore_counts = split(cache_delta.restore_count, count);
    let spilled_bytes = split(cache_delta.spilled_bytes, count);
    let restored_bytes = split(cache_delta.restored_bytes, count);
    let cooldown_hits = split(cache_delta.cooldown_hit_count, count);

    (0..count)
        .map(|index| VirtualCacheMetrics {
            spill_count: spill_counts[index],
            restore_count: restore_counts[index],
            spilled_bytes: spilled_bytes[index],
            restored_bytes: restored_bytes[index],
            cooldown_hit_count: cooldown_hits[index],
        })
        .collect()
}

fn validate_prefix_shape(
    seq: &VirtualSeqCache,
    expected_layer_count: usize,
    expected_kv_head_count: usize,
) -> Result<()> {
    if seq.layers.len() != expected_layer_count {
        return Err(RuntimeError::InvalidLayer {
            layer: seq.layers.len(),
            layer_count: expected_layer_count,
        });
    }
    for layer in &seq.layers {
        if layer.virtual_pages_by_kv_head.len() != expected_kv_head_count {
            return Err(RuntimeError::InvalidKvHead {
                kv_head: layer.virtual_pages_by_kv_head.len(),
                kv_head_count: expected_kv_head_count,
            });
        }
    }
    Ok(())
}
