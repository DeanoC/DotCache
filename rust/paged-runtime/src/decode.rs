use std::collections::BTreeMap;

use crate::backend::PageBackend;
use crate::cache::PageStore;
use crate::page::{KvPage, PageId};
use crate::virtual_page::{VirtualPageId, VirtualPageTable};
use crate::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageSpan {
    page_id: PageId,
    start: usize,
    end: usize,
}

pub fn softmax_in_place(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
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

pub fn decode_one_head<B: PageBackend>(
    backend: &B,
    store: &PageStore,
    page_ids: &[PageId],
    q: &[f32],
    out: &mut [f32],
) -> Result<()> {
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

    if backend.decode_fused(store, page_ids, q, out)? {
        return Ok(());
    }

    out.fill(0.0);
    let score_scale = attention_score_scale(q.len());

    let mut logits = Vec::new();
    let mut spans = Vec::with_capacity(page_ids.len());

    for &page_id in page_ids {
        let prepared = prepare_page(backend, store, page_id, q.len())?;
        let start = logits.len();
        backend.score(q, &prepared, &mut logits)?;
        for logit in &mut logits[start..] {
            *logit *= score_scale;
        }
        spans.push(PageSpan {
            page_id,
            start,
            end: logits.len(),
        });
    }

    softmax_in_place(&mut logits);

    for span in spans {
        let prepared = prepare_page(backend, store, span.page_id, q.len())?;
        backend.mix(&logits[span.start..span.end], &prepared, out)?;
    }

    Ok(())
}

fn attention_score_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

pub fn decode_one_head_owned<B: PageBackend>(
    backend: &B,
    store: &PageStore,
    page_ids: &[PageId],
    q: &[f32],
) -> Result<Vec<f32>> {
    let mut out = vec![0.0; q.len()];
    decode_one_head(backend, store, page_ids, q, &mut out)?;
    Ok(out)
}

pub fn decode_virtual_one_head<B: PageBackend>(
    backend: &B,
    store: &PageStore,
    page_table: &VirtualPageTable,
    virtual_page_ids: &[VirtualPageId],
    q: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let physical_page_ids = page_table.resolve(virtual_page_ids)?;
    decode_one_head(backend, store, &physical_page_ids, q, out)
}

pub fn decode_virtual_one_head_owned<B: PageBackend>(
    backend: &B,
    store: &PageStore,
    page_table: &VirtualPageTable,
    virtual_page_ids: &[VirtualPageId],
    q: &[f32],
) -> Result<Vec<f32>> {
    let mut out = vec![0.0; q.len()];
    decode_virtual_one_head(backend, store, page_table, virtual_page_ids, q, &mut out)?;
    Ok(out)
}

pub fn decode_query_batch_owned<B: PageBackend>(
    backend: &B,
    store: &PageStore,
    page_ids_by_query: &[&[PageId]],
    queries: &[&[f32]],
) -> Result<Vec<Vec<f32>>> {
    if queries.is_empty() {
        return Err(RuntimeError::EmptyInput {
            context: "batched decode queries",
        });
    }
    if page_ids_by_query.len() != queries.len() {
        return Err(RuntimeError::DimensionMismatch {
            context: "batched decode query count",
            expected: queries.len(),
            got: page_ids_by_query.len(),
        });
    }

    let head_dim = queries[0].len();
    let mut outputs = queries
        .iter()
        .map(|query| vec![0.0; query.len()])
        .collect::<Vec<_>>();

    if backend.decode_batch_fused(store, page_ids_by_query, queries, &mut outputs)? {
        return Ok(outputs);
    }

    let score_scale = attention_score_scale(head_dim);
    let max_page_count = page_ids_by_query
        .iter()
        .map(|page_ids| page_ids.len())
        .max()
        .unwrap_or(0);

    let mut logits_by_query = vec![Vec::new(); queries.len()];
    let mut spans_by_query = vec![Vec::new(); queries.len()];

    for (query_index, (query, page_ids)) in queries.iter().zip(page_ids_by_query.iter()).enumerate()
    {
        if page_ids.is_empty() {
            return Err(RuntimeError::EmptyDecode);
        }
        if query.len() != head_dim {
            return Err(RuntimeError::DimensionMismatch {
                context: "batched decode query",
                expected: head_dim,
                got: query.len(),
            });
        }
        spans_by_query[query_index].reserve(page_ids.len());
    }

    for page_slot in 0..max_page_count {
        for (page_id, query_indices) in grouped_page_queries(page_ids_by_query, page_slot) {
            let prepared = prepare_page(backend, store, page_id, head_dim)?;
            let query_batch = query_indices
                .iter()
                .map(|&query_index| queries[query_index])
                .collect::<Vec<_>>();
            let mut page_logits = vec![Vec::new(); query_indices.len()];
            backend.score_batch(&query_batch, &prepared, &mut page_logits)?;

            for (batch_index, logits) in page_logits.into_iter().enumerate() {
                let query_index = query_indices[batch_index];
                let start = logits_by_query[query_index].len();
                logits_by_query[query_index]
                    .extend(logits.into_iter().map(|logit| logit * score_scale));
                spans_by_query[query_index].push(PageSpan {
                    page_id,
                    start,
                    end: logits_by_query[query_index].len(),
                });
            }
        }
    }

    for logits in &mut logits_by_query {
        softmax_in_place(logits);
    }

    for page_slot in 0..max_page_count {
        for (page_id, query_indices) in grouped_page_queries(page_ids_by_query, page_slot) {
            let prepared = prepare_page(backend, store, page_id, head_dim)?;
            let weights_batch = query_indices
                .iter()
                .map(|&query_index| {
                    let span = spans_by_query[query_index].get(page_slot).ok_or(
                        RuntimeError::External {
                            context: "batched decode",
                            message: format!(
                            "missing score span for query {query_index} at page slot {page_slot}"
                        ),
                        },
                    )?;
                    debug_assert_eq!(span.page_id, page_id);
                    Ok(&logits_by_query[query_index][span.start..span.end])
                })
                .collect::<Result<Vec<_>>>()?;
            let mut output_batch = select_output_slices(&mut outputs, &query_indices);
            backend.mix_batch(&weights_batch, &prepared, &mut output_batch)?;
        }
    }

    Ok(outputs)
}

fn prepare_page<'a, B: PageBackend>(
    backend: &'a B,
    store: &'a PageStore,
    page_id: PageId,
    expected_head_dim: usize,
) -> Result<B::Prepared<'a>> {
    if let Some(prepared) = backend.prepare_cached(page_id, expected_head_dim)? {
        return Ok(prepared);
    }

    let page = store.page(page_id)?;
    validate_page(page_id, page, expected_head_dim)?;
    backend.prepare(page_id, page)
}

fn validate_page(page_id: PageId, page: &KvPage, expected_head_dim: usize) -> Result<()> {
    page.validate_layout(page_id)?;

    let head_dim = page.head_dim_usize();
    if head_dim != expected_head_dim {
        return Err(RuntimeError::DimensionMismatch {
            context: "query",
            expected: head_dim,
            got: expected_head_dim,
        });
    }

    Ok(())
}

fn grouped_page_queries(
    page_ids_by_query: &[&[PageId]],
    page_slot: usize,
) -> BTreeMap<PageId, Vec<usize>> {
    let mut grouped = BTreeMap::new();
    for (query_index, page_ids) in page_ids_by_query.iter().enumerate() {
        if let Some(&page_id) = page_ids.get(page_slot) {
            grouped
                .entry(page_id)
                .or_insert_with(Vec::new)
                .push(query_index);
        }
    }
    grouped
}

fn select_output_slices<'a>(
    outputs: &'a mut [Vec<f32>],
    query_indices: &[usize],
) -> Vec<&'a mut [f32]> {
    let mut selected = Vec::with_capacity(query_indices.len());
    let mut remaining = outputs;
    let mut base_index = 0usize;

    for &query_index in query_indices {
        let relative_index = query_index - base_index;
        let (_, tail) = remaining.split_at_mut(relative_index);
        let (output, rest) = tail
            .split_first_mut()
            .expect("query indices should always point to an existing output");
        selected.push(output.as_mut_slice());
        remaining = rest;
        base_index = query_index + 1;
    }

    selected
}
