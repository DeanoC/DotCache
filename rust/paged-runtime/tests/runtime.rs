use std::cell::RefCell;
use std::collections::HashMap;

use dotcache_paged_runtime::{
    decode_one_head_owned, decode_query_batch_owned, decode_virtual_one_head_owned,
    greedy_generate, softmax_in_place, CausalLm, CpuReferenceBackend, KvRow, ModelArchitecture,
    ModelFamily, PageBackend, PageId, PagedKvCache, RuntimeError, SessionRequestKind,
    SessionRuntime, SessionTokenRows, VirtualCacheMetrics, VirtualPagedKvCache,
};
#[cfg(feature = "candle")]
use dotcache_paged_runtime::{AttentionPathMode, CandleDeviceSelector, CandlePageBackend};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta < 1e-4,
            "mismatch at index {index}: actual={actual}, expected={expected}, delta={delta}"
        );
    }
}

fn scaled_softmax(logits: &mut [f32], head_dim: usize) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for logit in logits.iter_mut() {
        *logit *= scale;
    }
    softmax_in_place(logits);
}

fn single_head_token(key: [f32; 2], value: [f32; 2]) -> SessionTokenRows {
    SessionTokenRows::new(vec![vec![KvRow::new(key.to_vec(), value.to_vec())]])
}

fn two_head_token(
    head0_key: [f32; 2],
    head0_value: [f32; 2],
    head1_key: [f32; 2],
    head1_value: [f32; 2],
) -> SessionTokenRows {
    SessionTokenRows::new(vec![vec![
        KvRow::new(head0_key.to_vec(), head0_value.to_vec()),
        KvRow::new(head1_key.to_vec(), head1_value.to_vec()),
    ]])
}

#[derive(Debug, Clone, Copy)]
struct PreparedHandle {
    page_id: PageId,
}

#[derive(Debug, Default)]
struct CountingPrepareBackend {
    prepared_pages: RefCell<HashMap<PageId, dotcache_paged_runtime::KvPage>>,
    build_counts: RefCell<HashMap<PageId, usize>>,
}

impl CountingPrepareBackend {
    fn build_count(&self, page_id: PageId) -> usize {
        self.build_counts
            .borrow()
            .get(&page_id)
            .copied()
            .unwrap_or(0)
    }
}

impl PageBackend for CountingPrepareBackend {
    type Prepared<'a>
        = PreparedHandle
    where
        Self: 'a;

    fn descriptor(&self) -> dotcache_paged_runtime::BackendDescriptor {
        CpuReferenceBackend::default().descriptor()
    }

    fn prepare<'a>(
        &self,
        page_id: PageId,
        page: &'a dotcache_paged_runtime::KvPage,
    ) -> dotcache_paged_runtime::Result<Self::Prepared<'a>> {
        let mut prepared_pages = self.prepared_pages.borrow_mut();
        let mut build_counts = self.build_counts.borrow_mut();
        if !page.sealed || !prepared_pages.contains_key(&page_id) {
            prepared_pages.insert(page_id, page.clone());
            *build_counts.entry(page_id).or_insert(0) += 1;
        }
        Ok(PreparedHandle { page_id })
    }

    fn score(
        &self,
        q: &[f32],
        page: &Self::Prepared<'_>,
        logits_out: &mut Vec<f32>,
    ) -> dotcache_paged_runtime::Result<()> {
        let prepared_pages = self.prepared_pages.borrow();
        let page = prepared_pages.get(&page.page_id).unwrap();
        for token_index in 0..page.token_len() {
            let logit = q
                .iter()
                .zip(page.key_row(token_index).iter())
                .map(|(lhs, rhs)| lhs * rhs.to_f32())
                .sum();
            logits_out.push(logit);
        }
        Ok(())
    }

    fn mix(
        &self,
        weights: &[f32],
        page: &Self::Prepared<'_>,
        out: &mut [f32],
    ) -> dotcache_paged_runtime::Result<()> {
        let prepared_pages = self.prepared_pages.borrow();
        let page = prepared_pages.get(&page.page_id).unwrap();
        for (token_index, weight) in weights.iter().copied().enumerate() {
            for (out_value, value) in out.iter_mut().zip(page.value_row(token_index).iter()) {
                *out_value += weight * value.to_f32();
            }
        }
        Ok(())
    }
}

#[test]
fn append_seals_pages_and_rotates_the_live_tail() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);

    let first_page = cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();
    let sealed_page = cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[20.0, 200.0])
        .unwrap();
    let next_page = cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[30.0, 300.0])
        .unwrap();

    assert_eq!(first_page, 0);
    assert_eq!(sealed_page, 0);
    assert_eq!(next_page, 1);
    assert_eq!(cache.page_ids(0, 0).unwrap(), &[0, 1]);
    assert_eq!(cache.live_page_id(0, 0).unwrap(), Some(1));
    assert!(cache.page(0).unwrap().sealed);
    assert_eq!(cache.page(1).unwrap().token_start, 2);
}

#[test]
fn append_rejects_position_gaps_within_a_head_stream() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();

    let error = cache
        .append_token(0, 0, 2, &[0.0, 1.0], &[20.0, 200.0])
        .unwrap_err();

    assert_eq!(
        error,
        RuntimeError::PositionMismatch {
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn decode_matches_reference_softmax_across_multiple_pages() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let query = [2.0, 1.0];
    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();
    let backend = CpuReferenceBackend::default();

    let output = decode_one_head_owned(&backend, cache.store(), &page_ids, &query).unwrap();

    let mut weights = vec![2.0, 1.0, 3.0];
    scaled_softmax(&mut weights, query.len());
    let expected = vec![
        weights[0] * 1.0 + weights[1] * 2.0 + weights[2] * 4.0,
        weights[0] * 10.0 + weights[1] * 20.0 + weights[2] * 40.0,
    ];

    assert_eq!(page_ids, vec![0, 1]);
    assert_close(&output, &expected);
}

#[test]
fn batched_decode_matches_individual_decode_for_shared_pages() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let backend = CpuReferenceBackend::default();
    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();
    let queries = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
    let query_refs = queries
        .iter()
        .map(|query| query.as_slice())
        .collect::<Vec<_>>();
    let page_id_refs = vec![page_ids.as_slice(), page_ids.as_slice()];

    let batched =
        decode_query_batch_owned(&backend, cache.store(), &page_id_refs, &query_refs).unwrap();
    let individual = queries
        .iter()
        .map(|query| decode_one_head_owned(&backend, cache.store(), &page_ids, query))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(batched.len(), individual.len());
    for (batched, individual) in batched.iter().zip(individual.iter()) {
        assert_close(batched, individual);
    }
}

#[test]
fn sealed_pages_are_prepared_once_per_decode_when_backend_caches_them() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();
    let backend = CountingPrepareBackend::default();

    let _ = decode_one_head_owned(&backend, cache.store(), &page_ids, &[2.0, 1.0]).unwrap();

    assert_eq!(backend.build_count(0), 1);
    assert_eq!(backend.build_count(1), 2);
}

#[cfg(feature = "candle")]
#[test]
fn candle_prepare_cache_evicts_unpinned_pages_to_fit_budget() {
    let mut cache = PagedKvCache::new(1, 1, 1, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    backend.set_prepare_cache_page_budget(Some(2));
    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();

    let _ = decode_one_head_owned(&backend, cache.store(), &page_ids, &[2.0, 1.0]).unwrap();

    assert_eq!(backend.prepare_cache_page_budget(), Some(2));
    assert_eq!(backend.prepared_page_count(), 2);
    assert!(backend.cache_evictions() > 0);
}

#[cfg(feature = "candle")]
#[test]
fn candle_prepare_cache_keeps_pinned_pages_resident_under_budget_pressure() {
    let mut cache = PagedKvCache::new(1, 1, 1, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    backend.set_prepare_cache_page_budget(Some(2));
    backend.pin_page(0);
    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();

    let _ = decode_one_head_owned(&backend, cache.store(), &page_ids, &[2.0, 1.0]).unwrap();

    assert_eq!(backend.prepared_page_count(), 2);
    assert!(backend.is_page_pinned(0));
    assert!(backend.is_page_prepared(0));
    assert_eq!(backend.pinned_page_count(), 1);
}

#[cfg(feature = "candle")]
#[test]
fn candle_prepare_cache_release_prevents_stale_page_id_reuse() {
    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();

    let mut first_cache = PagedKvCache::new(1, 1, 1, 2);
    first_cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();
    let first_decode =
        decode_one_head_owned(&backend, first_cache.store(), &[0], &[1.0, 0.0]).unwrap();
    assert_close(&first_decode, &[10.0, 100.0]);
    assert!(backend.is_page_prepared(0));

    assert!(backend.release_page(0));
    assert!(!backend.is_page_prepared(0));

    let mut second_cache = PagedKvCache::new(1, 1, 1, 2);
    second_cache
        .append_token(0, 0, 0, &[0.0, 1.0], &[20.0, 200.0])
        .unwrap();
    let second_decode =
        decode_one_head_owned(&backend, second_cache.store(), &[0], &[0.0, 1.0]).unwrap();
    assert_close(&second_decode, &[20.0, 200.0]);
}

#[cfg(feature = "candle")]
#[test]
fn candle_ensure_page_resident_turns_first_decode_into_cache_hits() {
    let mut cache = PagedKvCache::new(1, 1, 1, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();

    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    let page = cache.page(0).unwrap();
    assert!(backend.ensure_page_resident(0, page).unwrap());
    assert_eq!(backend.cache_misses(), 1);
    assert!(backend.is_page_prepared(0));

    let decoded = decode_one_head_owned(&backend, cache.store(), &[0], &[1.0, 0.0]).unwrap();
    assert_close(&decoded, &[10.0, 100.0]);
    assert_eq!(backend.cache_misses(), 1);
    assert_eq!(backend.cache_hits(), 2);
}

#[cfg(feature = "candle")]
#[test]
fn candle_can_decode_from_device_resident_page_after_host_spill() {
    let mut cache = VirtualPagedKvCache::new(1, 1, 1, 2);
    let append_result = cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();
    assert!(append_result.sealed_now);

    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    let page = cache
        .physical()
        .store()
        .page(append_result.physical_page_id)
        .unwrap();
    assert!(backend
        .ensure_page_resident(append_result.physical_page_id, page)
        .unwrap());

    assert!(cache
        .spill_physical_page(append_result.physical_page_id)
        .unwrap());
    assert!(cache
        .physical()
        .store()
        .is_spilled(append_result.physical_page_id));
    assert!(backend.is_page_prepared(append_result.physical_page_id));

    let decoded = decode_one_head_owned(
        &backend,
        cache.physical().store(),
        &[append_result.physical_page_id],
        &[1.0, 0.0],
    )
    .unwrap();
    assert_close(&decoded, &[10.0, 100.0]);
}

#[cfg(feature = "candle")]
#[test]
fn candle_can_decode_from_device_only_page_without_host_payload() {
    let mut cache = VirtualPagedKvCache::new(1, 1, 1, 2);
    let append_result = cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();
    assert!(append_result.sealed_now);

    let backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    let page = cache
        .physical()
        .store()
        .page(append_result.physical_page_id)
        .unwrap();
    assert!(backend
        .ensure_page_resident(append_result.physical_page_id, page)
        .unwrap());
    backend.mark_page_device_primary(append_result.physical_page_id);
    assert!(cache
        .promote_physical_page_device_only(append_result.physical_page_id)
        .unwrap());
    assert_eq!(cache.resident_physical_page_count(), 0);
    assert_eq!(cache.device_only_physical_page_count(), 1);
    assert!(cache
        .physical()
        .store()
        .is_device_only(append_result.physical_page_id));

    let decoded = decode_one_head_owned(
        &backend,
        cache.physical().store(),
        &[append_result.physical_page_id],
        &[1.0, 0.0],
    )
    .unwrap();
    assert_close(&decoded, &[10.0, 100.0]);
}

#[cfg(feature = "candle")]
#[test]
fn candle_fused_attention_matches_paged_attention() {
    let mut cache = PagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let page_ids = cache.page_ids(0, 0).unwrap().to_vec();
    let query = [2.0, 1.0];

    let paged_backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    let fused_backend = CandlePageBackend::new(CandleDeviceSelector::Cpu).unwrap();
    fused_backend.set_attention_path(AttentionPathMode::Fused);

    let paged = decode_one_head_owned(&paged_backend, cache.store(), &page_ids, &query).unwrap();
    let fused = decode_one_head_owned(&fused_backend, cache.store(), &page_ids, &query).unwrap();
    assert_close(&fused, &paged);

    let page_id_refs = vec![page_ids.as_slice(), page_ids.as_slice()];
    let query_refs = vec![&query[..], &[1.0, 3.0][..]];
    let paged_batch =
        decode_query_batch_owned(&paged_backend, cache.store(), &page_id_refs, &query_refs)
            .unwrap();
    let fused_batch =
        decode_query_batch_owned(&fused_backend, cache.store(), &page_id_refs, &query_refs)
            .unwrap();

    for (fused_row, paged_row) in fused_batch.iter().zip(paged_batch.iter()) {
        assert_close(fused_row, paged_row);
    }
}

#[test]
fn decode_planner_resolves_layer_and_session_page_spans_once() {
    let mut runtime = SessionRuntime::new(1, 2, 2, 2);
    let session_id = runtime.create_session();
    runtime
        .append_token(
            session_id,
            &two_head_token([1.0, 0.0], [10.0, 100.0], [0.5, 0.5], [5.0, 50.0]),
        )
        .unwrap();
    runtime
        .append_token(
            session_id,
            &two_head_token([0.0, 1.0], [20.0, 200.0], [0.25, 0.75], [6.0, 60.0]),
        )
        .unwrap();
    runtime
        .append_token(
            session_id,
            &two_head_token([1.0, 1.0], [30.0, 300.0], [0.75, 0.25], [7.0, 70.0]),
        )
        .unwrap();

    let layer_plan = runtime.plan_layer_decode(session_id, 0).unwrap();
    assert_eq!(layer_plan.layer(), 0);
    assert_eq!(layer_plan.kv_head_count(), 2);
    assert_eq!(layer_plan.page_ids(0).unwrap(), &[0, 2]);
    assert_eq!(layer_plan.page_ids(1).unwrap(), &[1, 3]);

    let session_plan = runtime.plan_session_decode(session_id).unwrap();
    assert_eq!(session_plan.session_id(), session_id);
    assert_eq!(session_plan.layer_count(), 1);
    assert_eq!(session_plan.layer(0).unwrap().page_ids(0).unwrap(), &[0, 2]);
    assert_eq!(session_plan.layer(0).unwrap().page_ids(1).unwrap(), &[1, 3]);
}

#[test]
fn virtual_pages_alias_physical_pages_without_copying_data() {
    let mut cache = VirtualPagedKvCache::new(1, 1, 2, 2);
    let append_result = cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[10.0, 100.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[20.0, 200.0])
        .unwrap();

    let alias = cache
        .alias_virtual_page(append_result.virtual_page_id)
        .unwrap();

    assert_eq!(
        cache
            .virtual_table()
            .physical_page_id(append_result.virtual_page_id)
            .unwrap(),
        0
    );
    assert_eq!(cache.virtual_table().physical_page_id(alias).unwrap(), 0);
    assert_eq!(cache.virtual_table().ref_count(0), 2);
}

#[test]
fn decode_can_resolve_virtual_pages() {
    let mut cache = VirtualPagedKvCache::new(1, 1, 2, 2);
    cache
        .append_token(0, 0, 0, &[1.0, 0.0], &[1.0, 10.0])
        .unwrap();
    cache
        .append_token(0, 0, 1, &[0.0, 1.0], &[2.0, 20.0])
        .unwrap();
    cache
        .append_token(0, 0, 2, &[1.0, 1.0], &[4.0, 40.0])
        .unwrap();

    let backend = CpuReferenceBackend::default();
    let output = decode_virtual_one_head_owned(
        &backend,
        cache.physical().store(),
        cache.virtual_table(),
        cache.virtual_page_ids(0, 0).unwrap(),
        &[2.0, 1.0],
    )
    .unwrap();

    let mut weights = vec![2.0, 1.0, 3.0];
    scaled_softmax(&mut weights, 2);
    let expected = vec![
        weights[0] * 1.0 + weights[1] * 2.0 + weights[2] * 4.0,
        weights[0] * 10.0 + weights[1] * 20.0 + weights[2] * 40.0,
    ];

    assert_close(&output, &expected);
}

#[derive(Debug)]
struct MockModel {
    architecture: ModelArchitecture,
    next_logits: Vec<Vec<f32>>,
    reset_calls: usize,
}

impl CausalLm for MockModel {
    fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    fn reset(&mut self) -> dotcache_paged_runtime::Result<()> {
        self.reset_calls += 1;
        Ok(())
    }

    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> dotcache_paged_runtime::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> dotcache_paged_runtime::Result<String> {
        Ok(token_ids
            .iter()
            .map(|id| char::from_u32(*id).unwrap_or('?'))
            .collect())
    }

    fn forward_next_logits(
        &mut self,
        _input_ids: &[u32],
    ) -> dotcache_paged_runtime::Result<Vec<f32>> {
        Ok(self.next_logits.remove(0))
    }
}

#[test]
fn greedy_generation_uses_model_logits_and_stops_on_eos() {
    let mut model = MockModel {
        architecture: ModelArchitecture {
            model_id: "mock".to_string(),
            family: ModelFamily::Llama,
            vocab_size: 256,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            eos_token_ids: vec![33],
        },
        next_logits: vec![vec![0.0; 256], vec![0.0; 256]],
        reset_calls: 0,
    };
    model.next_logits[0][65] = 10.0;
    model.next_logits[1][33] = 10.0;

    let generation = greedy_generate(&mut model, "hi", 4).unwrap();

    assert_eq!(model.reset_calls, 1);
    assert_eq!(generation.generated_token_ids, vec![65, 33]);
}

#[test]
fn session_runtime_tracks_positions_and_page_growth() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let session_id = runtime.create_session_with_prompt_len(3);

    assert_eq!(runtime.session(session_id).unwrap().prompt_len(), 3);
    assert_eq!(runtime.session(session_id).unwrap().next_position(), 0);

    let first_pos = runtime
        .append_token(session_id, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    let second_pos = runtime
        .append_token(session_id, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    let third_pos = runtime
        .append_token(session_id, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();

    assert_eq!(first_pos, 0);
    assert_eq!(second_pos, 1);
    assert_eq!(third_pos, 2);
    assert_eq!(runtime.session(session_id).unwrap().token_count(), 3);
    assert_eq!(runtime.session(session_id).unwrap().next_position(), 3);
    assert_eq!(runtime.virtual_page_ids(session_id, 0, 0).unwrap().len(), 2);
    assert_eq!(
        runtime.resolve_physical_page_ids(session_id, 0, 0).unwrap(),
        vec![0, 1]
    );
}

#[test]
fn forked_sessions_alias_prefix_pages_and_copy_live_tail_on_write() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let root_session_id = runtime.create_session();
    runtime
        .append_token(
            root_session_id,
            &single_head_token([1.0, 0.0], [10.0, 100.0]),
        )
        .unwrap();
    runtime
        .append_token(
            root_session_id,
            &single_head_token([0.0, 1.0], [20.0, 200.0]),
        )
        .unwrap();
    runtime
        .append_token(
            root_session_id,
            &single_head_token([1.0, 1.0], [30.0, 300.0]),
        )
        .unwrap();

    let child_session_id = runtime.fork_session(root_session_id).unwrap();
    let root_prefix_pages = runtime
        .resolve_physical_page_ids(root_session_id, 0, 0)
        .unwrap();
    let child_prefix_pages = runtime
        .resolve_physical_page_ids(child_session_id, 0, 0)
        .unwrap();

    assert_eq!(root_prefix_pages, vec![0, 1]);
    assert_eq!(child_prefix_pages, vec![0, 1]);
    assert_eq!(runtime.cache().virtual_table().ref_count(0), 2);
    assert_eq!(runtime.cache().virtual_table().ref_count(1), 2);

    runtime
        .append_token(
            root_session_id,
            &single_head_token([2.0, 2.0], [40.0, 400.0]),
        )
        .unwrap();
    let root_after_write = runtime
        .resolve_physical_page_ids(root_session_id, 0, 0)
        .unwrap();
    let child_after_root_write = runtime
        .resolve_physical_page_ids(child_session_id, 0, 0)
        .unwrap();

    assert_eq!(root_after_write[0], child_after_root_write[0]);
    assert_ne!(root_after_write[1], child_after_root_write[1]);
    assert_eq!(
        runtime
            .cache()
            .virtual_table()
            .ref_count(child_after_root_write[1]),
        1
    );
    assert_eq!(
        runtime
            .cache()
            .physical()
            .page(child_after_root_write[1])
            .unwrap()
            .token_count,
        1
    );
    assert_eq!(
        runtime
            .cache()
            .physical()
            .page(root_after_write[1])
            .unwrap()
            .token_count,
        2
    );

    runtime
        .append_token(
            child_session_id,
            &single_head_token([3.0, 3.0], [50.0, 500.0]),
        )
        .unwrap();
    let child_after_write = runtime
        .resolve_physical_page_ids(child_session_id, 0, 0)
        .unwrap();

    assert_eq!(child_after_write[1], child_after_root_write[1]);
    assert_eq!(
        runtime
            .cache()
            .physical()
            .page(child_after_write[1])
            .unwrap()
            .token_count,
        2
    );
}

#[test]
fn captured_prefix_can_be_attached_to_new_sessions_and_diverge_on_write() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let source_session_id = runtime.create_session();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 0.0], [10.0, 100.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([0.0, 1.0], [20.0, 200.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 1.0], [30.0, 300.0]),
        )
        .unwrap();

    let prefix = runtime.capture_prefix(source_session_id).unwrap();
    assert_eq!(prefix.prompt_len(), 3);
    assert_eq!(prefix.token_count(), 3);
    assert_eq!(prefix.next_position(), 3);

    let attached_a = runtime.attach_prefix(&prefix).unwrap();
    let attached_b = runtime.attach_prefix(&prefix).unwrap();

    assert_eq!(runtime.session(attached_a).unwrap().prompt_len(), 3);
    assert_eq!(runtime.session(attached_a).unwrap().token_count(), 3);
    assert_eq!(runtime.session(attached_a).unwrap().next_position(), 3);
    assert_eq!(
        runtime
            .resolve_physical_page_ids(source_session_id, 0, 0)
            .unwrap(),
        runtime.resolve_physical_page_ids(attached_a, 0, 0).unwrap()
    );
    assert_eq!(
        runtime.resolve_physical_page_ids(attached_a, 0, 0).unwrap(),
        runtime.resolve_physical_page_ids(attached_b, 0, 0).unwrap()
    );

    runtime
        .append_token(attached_a, &single_head_token([2.0, 2.0], [40.0, 400.0]))
        .unwrap();

    let source_pages = runtime
        .resolve_physical_page_ids(source_session_id, 0, 0)
        .unwrap();
    let attached_a_pages = runtime.resolve_physical_page_ids(attached_a, 0, 0).unwrap();
    let attached_b_pages = runtime.resolve_physical_page_ids(attached_b, 0, 0).unwrap();

    assert_eq!(source_pages, attached_b_pages);
    assert_ne!(attached_a_pages.last(), attached_b_pages.last());
    assert_eq!(runtime.session(attached_a).unwrap().prompt_len(), 3);
    assert_eq!(runtime.session(attached_a).unwrap().token_count(), 4);
    assert_eq!(runtime.session(attached_a).unwrap().next_position(), 4);
}

#[test]
fn attached_prefix_sessions_can_grow_under_resident_page_budget() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    runtime
        .cache_mut()
        .set_resident_page_budget(Some(1))
        .unwrap();

    let seed_session = runtime.create_session();
    runtime
        .append_token(seed_session, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    runtime
        .append_token(seed_session, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    runtime
        .append_token(seed_session, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();

    let prefix = runtime.capture_prefix(seed_session).unwrap();
    runtime.close_session(seed_session).unwrap();

    let attached_session = runtime.attach_prefix(&prefix).unwrap();
    for token_index in 0..6 {
        runtime
            .append_token(
                attached_session,
                &single_head_token(
                    [token_index as f32, 1.0],
                    [token_index as f32 * 10.0, token_index as f32 * 100.0],
                ),
            )
            .unwrap();
        let _ = runtime.cache_mut().spill_to_budget().unwrap();
    }

    assert!(runtime.session(attached_session).is_ok());
    assert_eq!(runtime.cache().resident_page_budget(), Some(1));
}

#[test]
fn closing_sessions_and_releasing_prefix_reclaims_physical_pages() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let source_session_id = runtime.create_session();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 0.0], [10.0, 100.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([0.0, 1.0], [20.0, 200.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 1.0], [30.0, 300.0]),
        )
        .unwrap();
    assert_eq!(runtime.cache().physical_page_count(), 2);

    let prefix = runtime.capture_prefix(source_session_id).unwrap();
    let attached_session_id = runtime.attach_prefix(&prefix).unwrap();

    let reclaimed_from_source = runtime.close_session(source_session_id).unwrap();
    assert!(reclaimed_from_source.is_empty());
    assert_eq!(runtime.cache().physical_page_count(), 2);

    let reclaimed_from_attached = runtime.close_session(attached_session_id).unwrap();
    assert!(reclaimed_from_attached.is_empty());
    assert_eq!(runtime.cache().physical_page_count(), 2);

    let reclaimed_from_prefix = runtime.release_prefix(&prefix).unwrap();
    assert_eq!(reclaimed_from_prefix, vec![0, 1]);
    assert_eq!(runtime.cache().physical_page_count(), 0);
}

#[test]
fn spilled_sealed_pages_can_be_restored_for_decode() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let session_id = runtime.create_session();
    runtime
        .append_token(session_id, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();

    let page_ids = runtime.resolve_physical_page_ids(session_id, 0, 0).unwrap();
    let backend = CpuReferenceBackend::default();
    let before_spill = decode_one_head_owned(
        &backend,
        runtime.cache().physical().store(),
        &page_ids,
        &[2.0, 1.0],
    )
    .unwrap();

    let sealed_page_ids = runtime.sealed_physical_page_ids(session_id).unwrap();
    let spilled = runtime
        .cache_mut()
        .spill_physical_pages(&sealed_page_ids)
        .unwrap();
    assert_eq!(spilled, vec![0]);
    assert_eq!(runtime.cache().resident_physical_page_count(), 1);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 1);

    let restored = runtime
        .cache_mut()
        .restore_physical_pages(&sealed_page_ids)
        .unwrap();
    assert_eq!(restored, vec![0]);
    assert_eq!(runtime.cache().resident_physical_page_count(), 2);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 0);

    let after_restore = decode_one_head_owned(
        &backend,
        runtime.cache().physical().store(),
        &page_ids,
        &[2.0, 1.0],
    )
    .unwrap();
    assert_close(&before_spill, &after_restore);
}

#[test]
fn resident_page_budget_spills_cold_unpinned_pages_before_hot_or_pinned_pages() {
    let mut runtime = SessionRuntime::new(1, 1, 1, 2);
    let session_id = runtime.create_session();
    runtime
        .append_token(session_id, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();

    runtime.cache_mut().pin_physical_page(0);
    runtime.cache_mut().touch_physical_page(1).unwrap();
    runtime.cache_mut().touch_physical_page(2).unwrap();
    runtime
        .cache_mut()
        .set_resident_page_budget(Some(2))
        .unwrap();

    let spilled = runtime.cache_mut().spill_to_budget().unwrap();
    assert!(spilled.is_empty());
    assert_eq!(runtime.cache().resident_page_budget(), Some(2));
    assert_eq!(runtime.cache().resident_physical_page_count(), 2);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 1);
    assert!(runtime.cache().physical().store().is_spilled(1));
    assert!(!runtime.cache().physical().store().is_spilled(0));
    assert!(!runtime.cache().physical().store().is_spilled(2));

    let restored = runtime.cache_mut().restore_physical_pages(&[1]).unwrap();
    assert_eq!(restored, vec![1]);
    runtime.cache_mut().touch_physical_page(1).unwrap();
    runtime.cache_mut().unpin_physical_page(0).unwrap();

    assert_eq!(runtime.cache().resident_physical_page_count(), 2);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 1);
    assert!(runtime.cache().physical().store().is_spilled(0));
    assert!(!runtime.cache().physical().store().is_spilled(1));
    assert!(!runtime.cache().physical().store().is_spilled(2));
}

#[test]
fn resident_page_budget_prefers_less_shared_pages_over_shared_prefix_pages() {
    let mut runtime = SessionRuntime::new(1, 1, 1, 2);
    let source_session_id = runtime.create_session();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 0.0], [10.0, 100.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([0.0, 1.0], [20.0, 200.0]),
        )
        .unwrap();

    let prefix = runtime.capture_prefix(source_session_id).unwrap();
    let attached_session_id = runtime.attach_prefix(&prefix).unwrap();
    assert_eq!(runtime.cache().virtual_table().ref_count(0), 3);
    assert_eq!(runtime.cache().virtual_table().ref_count(1), 3);
    assert_eq!(
        runtime
            .resolve_physical_page_ids(attached_session_id, 0, 0)
            .unwrap(),
        vec![0, 1]
    );

    let unshared_session_id = runtime.create_session();
    runtime
        .append_token(
            unshared_session_id,
            &single_head_token([1.0, 1.0], [30.0, 300.0]),
        )
        .unwrap();
    assert_eq!(runtime.cache().virtual_table().ref_count(2), 1);

    runtime.cache_mut().touch_physical_page(0).unwrap();
    runtime.cache_mut().touch_physical_page(1).unwrap();
    runtime.cache_mut().touch_physical_page(2).unwrap();
    runtime
        .cache_mut()
        .set_resident_page_budget(Some(2))
        .unwrap();

    assert_eq!(runtime.cache().resident_physical_page_count(), 2);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 1);
    assert!(runtime.cache().physical().store().is_spilled(2));
    assert!(!runtime.cache().physical().store().is_spilled(0));
    assert!(!runtime.cache().physical().store().is_spilled(1));
}

#[test]
fn resident_byte_budget_accounts_for_live_tail_memory_pressure() {
    let mut runtime = SessionRuntime::new(1, 1, 2, 2);
    let session_id = runtime.create_session();
    runtime
        .append_token(session_id, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();

    assert_eq!(runtime.cache().resident_physical_page_count(), 2);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 0);
    assert_eq!(runtime.cache().resident_physical_byte_count(), 24);

    runtime
        .cache_mut()
        .set_resident_byte_budget(Some(16))
        .unwrap();

    assert_eq!(runtime.cache().resident_physical_page_count(), 1);
    assert_eq!(runtime.cache().spilled_physical_page_count(), 1);
    assert_eq!(runtime.cache().resident_physical_byte_count(), 8);
    assert_eq!(runtime.cache().spilled_physical_byte_count(), 16);
    assert!(runtime.cache().physical().store().is_spilled(0));
    assert!(!runtime.cache().physical().store().is_spilled(1));
}

#[test]
fn resident_budget_avoids_immediate_re_spill_of_recently_restored_pages() {
    let mut runtime = SessionRuntime::new(1, 1, 1, 2);
    let session_id = runtime.create_session();
    runtime
        .append_token(session_id, &single_head_token([1.0, 0.0], [10.0, 100.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([0.0, 1.0], [20.0, 200.0]))
        .unwrap();
    runtime
        .append_token(session_id, &single_head_token([1.0, 1.0], [30.0, 300.0]))
        .unwrap();
    runtime.cache_mut().set_restore_cooldown_window(16);
    runtime
        .cache_mut()
        .set_resident_page_budget(Some(2))
        .unwrap();

    assert!(runtime.cache().physical().store().is_spilled(0));
    assert!(!runtime.cache().physical().store().is_spilled(1));
    assert!(!runtime.cache().physical().store().is_spilled(2));

    let restored = runtime.cache_mut().restore_physical_pages(&[0]).unwrap();
    assert_eq!(restored, vec![0]);
    assert_eq!(runtime.cache().resident_physical_page_count(), 3);

    let spilled = runtime.cache_mut().spill_to_budget().unwrap();
    assert_eq!(spilled, vec![1]);
    assert!(!runtime.cache().physical().store().is_spilled(0));
    assert!(runtime.cache().physical().store().is_spilled(1));
    assert!(!runtime.cache().physical().store().is_spilled(2));
}

#[test]
fn virtual_cache_metrics_track_spills_restores_and_cooldown_hits() {
    let mut runtime = SessionRuntime::new(1, 1, 1, 2);
    let source_session_id = runtime.create_session();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([1.0, 0.0], [10.0, 100.0]),
        )
        .unwrap();
    runtime
        .append_token(
            source_session_id,
            &single_head_token([0.0, 1.0], [20.0, 200.0]),
        )
        .unwrap();

    let prefix = runtime.capture_prefix(source_session_id).unwrap();
    let _attached_session_id = runtime.attach_prefix(&prefix).unwrap();
    let unshared_session_id = runtime.create_session();
    runtime
        .append_token(
            unshared_session_id,
            &single_head_token([1.0, 1.0], [30.0, 300.0]),
        )
        .unwrap();

    runtime.cache_mut().reset_metrics();
    runtime.cache_mut().set_restore_cooldown_window(16);
    runtime
        .cache_mut()
        .set_resident_page_budget(Some(2))
        .unwrap();
    assert_eq!(runtime.cache().metrics().spill_count, 1);
    assert_eq!(runtime.cache().metrics().restore_count, 0);
    assert_eq!(runtime.cache().metrics().spilled_bytes, 8);
    assert_eq!(runtime.cache().metrics().restored_bytes, 0);
    assert_eq!(runtime.cache().metrics().cooldown_hit_count, 0);

    let restored = runtime.cache_mut().restore_physical_pages(&[2]).unwrap();
    assert_eq!(restored, vec![2]);
    let spilled = runtime.cache_mut().spill_to_budget().unwrap();
    assert_eq!(spilled, vec![0]);

    let metrics = runtime.cache().metrics();
    assert_eq!(metrics.spill_count, 2);
    assert_eq!(metrics.restore_count, 1);
    assert_eq!(metrics.spilled_bytes, 16);
    assert_eq!(metrics.restored_bytes, 8);
    assert_eq!(metrics.cooldown_hit_count, 1);
}

#[test]
fn session_runtime_accumulates_request_metrics_per_session() {
    let mut runtime = SessionRuntime::new(1, 1, 1, 2);
    let session_a = runtime.create_session();
    let session_b = runtime.create_session();

    runtime
        .record_session_request(
            &[session_a],
            SessionRequestKind::Prefill,
            &[4],
            &VirtualCacheMetrics {
                spill_count: 2,
                restore_count: 1,
                spilled_bytes: 16,
                restored_bytes: 8,
                cooldown_hit_count: 1,
            },
        )
        .unwrap();
    runtime
        .record_session_request(
            &[session_a, session_b],
            SessionRequestKind::BatchDecode,
            &[1, 1],
            &VirtualCacheMetrics {
                spill_count: 1,
                restore_count: 3,
                spilled_bytes: 8,
                restored_bytes: 24,
                cooldown_hit_count: 1,
            },
        )
        .unwrap();

    let metrics_a = runtime.session_metrics(session_a).unwrap();
    assert_eq!(metrics_a.request_count, 2);
    assert_eq!(metrics_a.prefill_request_count, 1);
    assert_eq!(metrics_a.batch_decode_request_count, 1);
    assert_eq!(metrics_a.input_token_count, 5);
    assert_eq!(metrics_a.spill_count, 3);
    assert_eq!(metrics_a.restore_count, 3);
    assert_eq!(metrics_a.spilled_bytes, 20);
    assert_eq!(metrics_a.restored_bytes, 20);
    assert_eq!(metrics_a.cooldown_hit_count, 2);

    let metrics_b = runtime.session_metrics(session_b).unwrap();
    assert_eq!(metrics_b.request_count, 1);
    assert_eq!(metrics_b.prefill_request_count, 0);
    assert_eq!(metrics_b.batch_decode_request_count, 1);
    assert_eq!(metrics_b.input_token_count, 1);
    assert_eq!(metrics_b.spill_count, 0);
    assert_eq!(metrics_b.restore_count, 1);
    assert_eq!(metrics_b.spilled_bytes, 4);
    assert_eq!(metrics_b.restored_bytes, 12);
    assert_eq!(metrics_b.cooldown_hit_count, 0);
}
