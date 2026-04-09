use crate::cache::PagedKvCache;
use crate::page::{KvPage, PageId};
use crate::{Result, RuntimeError};
use std::cmp::Reverse;
use std::collections::HashMap;

pub type VirtualPageId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppendPageResult {
    pub physical_page_id: PageId,
    pub virtual_page_id: VirtualPageId,
    pub sealed_now: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualPage {
    pub physical_page_id: PageId,
    pub layer: u16,
    pub kv_head: u16,
    pub token_start: u32,
    pub token_count: u16,
    pub sealed: bool,
}

impl VirtualPage {
    fn from_physical(physical_page_id: PageId, page: &KvPage) -> Self {
        Self {
            physical_page_id,
            layer: page.layer,
            kv_head: page.kv_head,
            token_start: page.token_start,
            token_count: page.token_count,
            sealed: page.sealed,
        }
    }

    pub fn token_end(&self) -> u32 {
        self.token_start + u32::from(self.token_count)
    }
}

#[derive(Clone, Debug, Default)]
pub struct VirtualPageTable {
    pages: Vec<Option<VirtualPage>>,
    physical_to_virtual: Vec<Vec<VirtualPageId>>,
}

impl VirtualPageTable {
    pub fn len(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn virtual_page(&self, virtual_page_id: VirtualPageId) -> Result<&VirtualPage> {
        self.pages
            .get(virtual_page_id)
            .and_then(|page| page.as_ref())
            .ok_or(RuntimeError::InvalidVirtualPageId {
                virtual_page_id,
                page_count: self.pages.len(),
            })
    }

    pub fn physical_page_id(&self, virtual_page_id: VirtualPageId) -> Result<PageId> {
        Ok(self.virtual_page(virtual_page_id)?.physical_page_id)
    }

    pub fn map_physical(&mut self, physical_page_id: PageId, page: &KvPage) -> VirtualPageId {
        self.ensure_physical_slot(physical_page_id);
        let virtual_page_id = self.pages.len();
        self.pages
            .push(Some(VirtualPage::from_physical(physical_page_id, page)));
        self.physical_to_virtual[physical_page_id].push(virtual_page_id);
        virtual_page_id
    }

    pub fn alias(&mut self, virtual_page_id: VirtualPageId) -> Result<VirtualPageId> {
        let page = self.virtual_page(virtual_page_id)?.clone();
        self.ensure_physical_slot(page.physical_page_id);
        let alias_id = self.pages.len();
        self.pages.push(Some(page.clone()));
        self.physical_to_virtual[page.physical_page_id].push(alias_id);
        Ok(alias_id)
    }

    pub fn remap(
        &mut self,
        virtual_page_id: VirtualPageId,
        physical_page_id: PageId,
        page: &KvPage,
    ) -> Result<()> {
        let previous_physical_page_id = self.virtual_page(virtual_page_id)?.physical_page_id;
        self.ensure_physical_slot(physical_page_id);

        if previous_physical_page_id != physical_page_id {
            if let Some(mapped_virtual_pages) =
                self.physical_to_virtual.get_mut(previous_physical_page_id)
            {
                if let Some(index) = mapped_virtual_pages
                    .iter()
                    .position(|&mapped_virtual_page_id| mapped_virtual_page_id == virtual_page_id)
                {
                    mapped_virtual_pages.remove(index);
                }
            }
            self.physical_to_virtual[physical_page_id].push(virtual_page_id);
        }

        self.pages[virtual_page_id] = Some(VirtualPage::from_physical(physical_page_id, page));
        Ok(())
    }

    pub fn resolve(&self, virtual_page_ids: &[VirtualPageId]) -> Result<Vec<PageId>> {
        virtual_page_ids
            .iter()
            .map(|&virtual_page_id| self.physical_page_id(virtual_page_id))
            .collect()
    }

    pub fn ref_count(&self, physical_page_id: PageId) -> usize {
        self.physical_to_virtual
            .get(physical_page_id)
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    pub fn sync_physical(&mut self, physical_page_id: PageId, page: &KvPage) {
        let Some(mapped_virtual_pages) = self.physical_to_virtual.get(physical_page_id) else {
            return;
        };
        for &virtual_page_id in mapped_virtual_pages {
            self.pages[virtual_page_id] = Some(VirtualPage::from_physical(physical_page_id, page));
        }
    }

    pub fn release(&mut self, virtual_page_id: VirtualPageId) -> Result<PageId> {
        let page_count = self.pages.len();
        let page = self
            .pages
            .get_mut(virtual_page_id)
            .ok_or(RuntimeError::InvalidVirtualPageId {
                virtual_page_id,
                page_count,
            })?
            .take()
            .ok_or(RuntimeError::InvalidVirtualPageId {
                virtual_page_id,
                page_count,
            })?;

        if let Some(mapped_virtual_pages) = self.physical_to_virtual.get_mut(page.physical_page_id)
        {
            if let Some(index) = mapped_virtual_pages
                .iter()
                .position(|&mapped_virtual_page_id| mapped_virtual_page_id == virtual_page_id)
            {
                mapped_virtual_pages.remove(index);
            }
        }

        Ok(page.physical_page_id)
    }

    fn ensure_physical_slot(&mut self, physical_page_id: PageId) {
        if self.physical_to_virtual.len() <= physical_page_id {
            self.physical_to_virtual
                .resize_with(physical_page_id + 1, Vec::new);
        }
    }
}

#[derive(Clone, Debug)]
pub struct VirtualLayerCache {
    pub virtual_pages_by_kv_head: Vec<Vec<VirtualPageId>>,
    pub live_by_kv_head: Vec<Option<VirtualPageId>>,
}

impl VirtualLayerCache {
    pub fn new(kv_head_count: usize) -> Self {
        Self {
            virtual_pages_by_kv_head: vec![Vec::new(); kv_head_count],
            live_by_kv_head: vec![None; kv_head_count],
        }
    }
}

#[derive(Clone, Debug)]
pub struct VirtualSeqCache {
    pub layers: Vec<VirtualLayerCache>,
}

impl VirtualSeqCache {
    pub fn new(layer_count: usize, kv_head_count: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| VirtualLayerCache::new(kv_head_count))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VirtualPagedKvCache {
    physical: PagedKvCache,
    virtual_seq: VirtualSeqCache,
    virtual_table: VirtualPageTable,
    pinned_physical_pages: HashMap<PageId, usize>,
    physical_page_last_access: HashMap<PageId, u64>,
    physical_page_last_restore: HashMap<PageId, u64>,
    access_clock: u64,
    resident_page_budget: Option<usize>,
    resident_byte_budget: Option<usize>,
    restore_cooldown_window: u64,
    metrics: VirtualCacheMetrics,
}

#[cfg_attr(feature = "hf", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtualCacheMetrics {
    pub spill_count: usize,
    pub restore_count: usize,
    pub spilled_bytes: usize,
    pub restored_bytes: usize,
    pub cooldown_hit_count: usize,
}

impl VirtualCacheMetrics {
    pub fn delta_since(&self, before: &Self) -> Self {
        Self {
            spill_count: self.spill_count.saturating_sub(before.spill_count),
            restore_count: self.restore_count.saturating_sub(before.restore_count),
            spilled_bytes: self.spilled_bytes.saturating_sub(before.spilled_bytes),
            restored_bytes: self.restored_bytes.saturating_sub(before.restored_bytes),
            cooldown_hit_count: self
                .cooldown_hit_count
                .saturating_sub(before.cooldown_hit_count),
        }
    }
}

impl VirtualPagedKvCache {
    pub fn new(
        layer_count: usize,
        kv_head_count: usize,
        tokens_per_page: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            physical: PagedKvCache::new(layer_count, kv_head_count, tokens_per_page, head_dim),
            virtual_seq: VirtualSeqCache::new(layer_count, kv_head_count),
            virtual_table: VirtualPageTable::default(),
            pinned_physical_pages: HashMap::new(),
            physical_page_last_access: HashMap::new(),
            physical_page_last_restore: HashMap::new(),
            access_clock: 0,
            resident_page_budget: None,
            resident_byte_budget: None,
            restore_cooldown_window: 8,
            metrics: VirtualCacheMetrics::default(),
        }
    }

    pub fn physical(&self) -> &PagedKvCache {
        &self.physical
    }

    pub fn layer_count(&self) -> usize {
        self.virtual_seq.layers.len()
    }

    pub fn kv_head_count(&self) -> usize {
        self.virtual_seq
            .layers
            .first()
            .map(|layer| layer.virtual_pages_by_kv_head.len())
            .unwrap_or(0)
    }

    pub fn head_dim(&self) -> usize {
        self.physical.head_dim()
    }

    pub fn physical_page_count(&self) -> usize {
        self.physical.store().len()
    }

    pub fn resident_physical_page_count(&self) -> usize {
        self.physical.resident_page_count()
    }

    pub fn spilled_physical_page_count(&self) -> usize {
        self.physical.spilled_page_count()
    }

    pub fn device_only_physical_page_count(&self) -> usize {
        self.physical.device_only_page_count()
    }

    pub fn resident_physical_byte_count(&self) -> usize {
        self.physical.resident_byte_count()
    }

    pub fn spilled_physical_byte_count(&self) -> usize {
        self.physical.spilled_byte_count()
    }

    pub fn pinned_physical_page_count(&self) -> usize {
        self.pinned_physical_pages.len()
    }

    pub fn virtual_page_count(&self) -> usize {
        self.virtual_table.len()
    }

    pub fn resident_page_budget(&self) -> Option<usize> {
        self.resident_page_budget
    }

    pub fn set_resident_page_budget(&mut self, budget: Option<usize>) -> Result<()> {
        self.resident_page_budget = budget;
        self.spill_to_budget()?;
        Ok(())
    }

    pub fn resident_byte_budget(&self) -> Option<usize> {
        self.resident_byte_budget
    }

    pub fn set_resident_byte_budget(&mut self, budget: Option<usize>) -> Result<()> {
        self.resident_byte_budget = budget;
        self.spill_to_budget()?;
        Ok(())
    }

    pub fn restore_cooldown_window(&self) -> u64 {
        self.restore_cooldown_window
    }

    pub fn set_restore_cooldown_window(&mut self, window: u64) {
        self.restore_cooldown_window = window;
    }

    pub fn metrics(&self) -> &VirtualCacheMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = VirtualCacheMetrics::default();
    }

    pub fn total_token_count(&self) -> usize {
        self.physical.store().total_token_count()
    }

    pub fn tokens_per_page(&self) -> usize {
        self.physical.tokens_per_page()
    }

    pub fn virtual_table(&self) -> &VirtualPageTable {
        &self.virtual_table
    }

    pub fn virtual_seq(&self) -> &VirtualSeqCache {
        &self.virtual_seq
    }

    pub fn fork_seq(&mut self, source: &VirtualSeqCache) -> Result<VirtualSeqCache> {
        validate_seq_shape(source, self.layer_count(), self.kv_head_count())?;

        let mut fork = VirtualSeqCache::new(self.layer_count(), self.kv_head_count());
        for layer in 0..self.layer_count() {
            for kv_head in 0..self.kv_head_count() {
                for &virtual_page_id in &source.layers[layer].virtual_pages_by_kv_head[kv_head] {
                    let alias_id = self.virtual_table.alias(virtual_page_id)?;
                    fork.layers[layer].virtual_pages_by_kv_head[kv_head].push(alias_id);
                    if source.layers[layer].live_by_kv_head[kv_head] == Some(virtual_page_id) {
                        fork.layers[layer].live_by_kv_head[kv_head] = Some(alias_id);
                    }
                }
            }
        }

        Ok(fork)
    }

    pub fn virtual_page_ids(&self, layer: usize, kv_head: usize) -> Result<&[VirtualPageId]> {
        self.validate_slot(layer, kv_head)?;
        Ok(self.virtual_seq.layers[layer].virtual_pages_by_kv_head[kv_head].as_slice())
    }

    pub fn live_virtual_page_id(
        &self,
        layer: usize,
        kv_head: usize,
    ) -> Result<Option<VirtualPageId>> {
        self.validate_slot(layer, kv_head)?;
        Ok(self.virtual_seq.layers[layer].live_by_kv_head[kv_head])
    }

    pub fn resolve_physical_page_ids(&self, layer: usize, kv_head: usize) -> Result<Vec<PageId>> {
        let virtual_pages = self.virtual_page_ids(layer, kv_head)?;
        self.virtual_table.resolve(virtual_pages)
    }

    pub fn alias_virtual_page(&mut self, virtual_page_id: VirtualPageId) -> Result<VirtualPageId> {
        self.virtual_table.alias(virtual_page_id)
    }

    pub fn append_token(
        &mut self,
        layer: usize,
        kv_head: usize,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Result<AppendPageResult> {
        let append_result = append_token_to_seq(
            &mut self.physical,
            &mut self.virtual_table,
            &mut self.virtual_seq,
            layer,
            kv_head,
            pos,
            k_row,
            v_row,
        )?;
        self.touch_physical_page(append_result.physical_page_id)?;
        Ok(append_result)
    }

    pub fn append_token_to(
        &mut self,
        seq: &mut VirtualSeqCache,
        layer: usize,
        kv_head: usize,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Result<AppendPageResult> {
        validate_seq_shape(seq, self.layer_count(), self.kv_head_count())?;
        let append_result = append_token_to_seq(
            &mut self.physical,
            &mut self.virtual_table,
            seq,
            layer,
            kv_head,
            pos,
            k_row,
            v_row,
        )?;
        self.touch_physical_page(append_result.physical_page_id)?;
        Ok(append_result)
    }

    pub fn release_seq(&mut self, seq: &VirtualSeqCache) -> Result<Vec<PageId>> {
        validate_seq_shape(seq, self.layer_count(), self.kv_head_count())?;

        let mut released_physical_page_ids = std::collections::BTreeSet::new();
        let mut released_virtual_page_ids = std::collections::BTreeSet::new();
        for layer in &seq.layers {
            for virtual_page_ids in &layer.virtual_pages_by_kv_head {
                for &virtual_page_id in virtual_page_ids {
                    if released_virtual_page_ids.insert(virtual_page_id) {
                        let physical_page_id = self.virtual_table.release(virtual_page_id)?;
                        if self.virtual_table.ref_count(physical_page_id) == 0 {
                            self.physical.reclaim_slot(physical_page_id)?;
                            self.pinned_physical_pages.remove(&physical_page_id);
                            self.physical_page_last_access.remove(&physical_page_id);
                            self.physical_page_last_restore.remove(&physical_page_id);
                            released_physical_page_ids.insert(physical_page_id);
                        }
                    }
                }
            }
        }

        Ok(released_physical_page_ids.into_iter().collect())
    }

    pub fn spill_physical_page(&mut self, page_id: PageId) -> Result<bool> {
        if self.is_physical_page_pinned(page_id) {
            return Ok(false);
        }
        let page = match self.physical.page(page_id) {
            Ok(page) => page,
            Err(_) => return Ok(false),
        };
        if !page.sealed {
            return Ok(false);
        }
        let spilled_bytes = page.kv_byte_len();
        let spilled = self.physical.spill_page(page_id)?;
        if spilled {
            self.physical_page_last_restore.remove(&page_id);
            self.metrics.spill_count += 1;
            self.metrics.spilled_bytes += spilled_bytes;
        }
        Ok(spilled)
    }

    pub fn spill_physical_pages(&mut self, page_ids: &[PageId]) -> Result<Vec<PageId>> {
        let mut spilled_page_ids = Vec::new();
        for &page_id in page_ids {
            if self.spill_physical_page(page_id)? {
                spilled_page_ids.push(page_id);
            }
        }
        Ok(spilled_page_ids)
    }

    pub fn promote_physical_page_device_only(&mut self, page_id: PageId) -> Result<bool> {
        self.physical.promote_device_only_page(page_id)
    }

    pub fn restore_physical_page(&mut self, page_id: PageId) -> Result<bool> {
        let restored = self.physical.restore_page(page_id)?;
        if restored {
            let restored_bytes = self.physical.page(page_id)?.kv_byte_len();
            self.mark_restored(page_id)?;
            self.metrics.restore_count += 1;
            self.metrics.restored_bytes += restored_bytes;
        }
        Ok(restored)
    }

    pub fn restore_physical_pages(&mut self, page_ids: &[PageId]) -> Result<Vec<PageId>> {
        let mut restored_page_ids = Vec::new();
        for &page_id in page_ids {
            if self.restore_physical_page(page_id)? {
                restored_page_ids.push(page_id);
            }
        }
        Ok(restored_page_ids)
    }

    pub fn pin_physical_page(&mut self, page_id: PageId) {
        *self.pinned_physical_pages.entry(page_id).or_insert(0) += 1;
    }

    pub fn pin_physical_pages(&mut self, page_ids: &[PageId]) {
        for &page_id in page_ids {
            self.pin_physical_page(page_id);
        }
    }

    pub fn unpin_physical_page(&mut self, page_id: PageId) -> Result<()> {
        match self.pinned_physical_pages.get_mut(&page_id) {
            Some(pin_count) if *pin_count > 1 => {
                *pin_count -= 1;
            }
            Some(_) => {
                self.pinned_physical_pages.remove(&page_id);
            }
            None => {
                return Err(RuntimeError::External {
                    context: "virtual_cache",
                    message: format!("physical page {page_id} is not pinned"),
                });
            }
        }
        self.spill_to_budget()?;
        Ok(())
    }

    pub fn unpin_physical_pages(&mut self, page_ids: &[PageId]) -> Result<()> {
        for &page_id in page_ids {
            self.unpin_physical_page(page_id)?;
        }
        Ok(())
    }

    pub fn is_physical_page_pinned(&self, page_id: PageId) -> bool {
        self.pinned_physical_pages.contains_key(&page_id)
    }

    pub fn touch_physical_page(&mut self, page_id: PageId) -> Result<()> {
        if self.physical.page(page_id).is_err()
            && !self.physical.store().is_spilled(page_id)
            && !self.physical.store().is_device_only(page_id)
        {
            return Err(RuntimeError::InvalidPageId {
                page_id,
                page_count: self.physical.store().len(),
            });
        }
        let now = self.advance_access_clock();
        self.physical_page_last_access.insert(page_id, now);
        Ok(())
    }

    pub fn touch_physical_pages(&mut self, page_ids: &[PageId]) -> Result<()> {
        for &page_id in page_ids {
            self.touch_physical_page(page_id)?;
        }
        Ok(())
    }

    pub fn spill_to_budget(&mut self) -> Result<Vec<PageId>> {
        if self.resident_page_budget.is_none() && self.resident_byte_budget.is_none() {
            return Ok(Vec::new());
        }

        let mut spilled_page_ids = Vec::new();
        while self.is_over_resident_budget() {
            let byte_pressure = self
                .resident_byte_budget
                .is_some_and(|budget| self.resident_physical_byte_count() > budget);
            let candidate_without_cooldown = self
                .physical
                .store()
                .iter_with_ids()
                .filter(|(_, page)| page.sealed)
                .filter(|(page_id, _)| !self.is_physical_page_pinned(*page_id))
                .min_by_key(|(page_id, page)| {
                    self.spill_candidate_cost(*page_id, page, byte_pressure, false)
                })
                .map(|(page_id, _)| page_id);
            let candidate = self
                .physical
                .store()
                .iter_with_ids()
                .filter(|(_, page)| page.sealed)
                .filter(|(page_id, _)| !self.is_physical_page_pinned(*page_id))
                .min_by_key(|(page_id, page)| {
                    self.spill_candidate_cost(*page_id, page, byte_pressure, true)
                })
                .map(|(page_id, _)| page_id);

            if let (Some(with_cooldown), Some(without_cooldown)) =
                (candidate, candidate_without_cooldown)
            {
                if with_cooldown != without_cooldown
                    && self
                        .physical_page_last_restore
                        .get(&without_cooldown)
                        .copied()
                        .is_some_and(|last_restore| self.restore_penalty(Some(last_restore)) > 0)
                {
                    self.metrics.cooldown_hit_count += 1;
                }
            }

            let Some(page_id) = candidate else {
                break;
            };
            if self.spill_physical_page(page_id)? {
                spilled_page_ids.push(page_id);
            } else {
                break;
            }
        }

        Ok(spilled_page_ids)
    }

    fn is_over_resident_budget(&self) -> bool {
        self.resident_page_budget
            .is_some_and(|budget| self.resident_physical_page_count() > budget)
            || self
                .resident_byte_budget
                .is_some_and(|budget| self.resident_physical_byte_count() > budget)
    }

    fn spill_candidate_cost(
        &self,
        page_id: PageId,
        page: &KvPage,
        byte_pressure: bool,
        apply_restore_penalty: bool,
    ) -> SpillCandidateCost {
        let last_restore = self.physical_page_last_restore.get(&page_id).copied();
        SpillCandidateCost {
            restore_penalty: if apply_restore_penalty {
                self.restore_penalty(last_restore)
            } else {
                0
            },
            sharing_cost: self.virtual_table.ref_count(page_id),
            priority_size_credit: if byte_pressure {
                Reverse(page.kv_byte_len())
            } else {
                Reverse(0)
            },
            recency_cost: self
                .physical_page_last_access
                .get(&page_id)
                .copied()
                .unwrap_or(0),
            size_credit: Reverse(page.kv_byte_len()),
            page_id,
        }
    }

    fn mark_restored(&mut self, page_id: PageId) -> Result<()> {
        if self.physical.page(page_id).is_err() {
            return Err(RuntimeError::InvalidPageId {
                page_id,
                page_count: self.physical.store().len(),
            });
        }
        let now = self.advance_access_clock();
        self.physical_page_last_restore.insert(page_id, now);
        self.physical_page_last_access.insert(page_id, now);
        Ok(())
    }

    fn advance_access_clock(&mut self) -> u64 {
        self.access_clock += 1;
        self.access_clock
    }

    fn restore_penalty(&self, last_restore: Option<u64>) -> u64 {
        let Some(last_restore) = last_restore else {
            return 0;
        };
        let age = self.access_clock.saturating_sub(last_restore);
        self.restore_cooldown_window.saturating_sub(age)
    }

    fn validate_slot(&self, layer: usize, kv_head: usize) -> Result<()> {
        if layer >= self.virtual_seq.layers.len() {
            return Err(RuntimeError::InvalidLayer {
                layer,
                layer_count: self.virtual_seq.layers.len(),
            });
        }
        let kv_head_count = self.virtual_seq.layers[layer]
            .virtual_pages_by_kv_head
            .len();
        if kv_head >= kv_head_count {
            return Err(RuntimeError::InvalidKvHead {
                kv_head,
                kv_head_count,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SpillCandidateCost {
    restore_penalty: u64,
    sharing_cost: usize,
    priority_size_credit: Reverse<usize>,
    recency_cost: u64,
    size_credit: Reverse<usize>,
    page_id: PageId,
}

fn append_token_to_seq(
    physical: &mut PagedKvCache,
    virtual_table: &mut VirtualPageTable,
    seq: &mut VirtualSeqCache,
    layer: usize,
    kv_head: usize,
    pos: u32,
    k_row: &[f32],
    v_row: &[f32],
) -> Result<AppendPageResult> {
    validate_seq_slot(seq, layer, kv_head)?;

    if let Some(expected_pos) = expected_next_pos(seq, virtual_table, layer, kv_head)? {
        if pos != expected_pos {
            return Err(RuntimeError::PositionMismatch {
                expected: expected_pos,
                got: pos,
            });
        }
    }

    let live_virtual_page_id = seq.layers[layer].live_by_kv_head[kv_head];
    let (physical_page_id, virtual_page_id, allocated_new_virtual_page) = match live_virtual_page_id
    {
        Some(virtual_page_id) => {
            let live_page = virtual_table.virtual_page(virtual_page_id)?.clone();
            if live_page.sealed {
                allocate_virtual_page(physical, virtual_table, seq, layer, kv_head, pos)?
            } else if virtual_table.ref_count(live_page.physical_page_id) > 1 {
                let cloned_physical_page_id =
                    clone_physical_page(physical, live_page.physical_page_id)?;
                let cloned_page = physical.page(cloned_physical_page_id)?.clone();
                virtual_table.remap(virtual_page_id, cloned_physical_page_id, &cloned_page)?;
                (cloned_physical_page_id, virtual_page_id, false)
            } else {
                (live_page.physical_page_id, virtual_page_id, false)
            }
        }
        None => allocate_virtual_page(physical, virtual_table, seq, layer, kv_head, pos)?,
    };

    let mut sealed = false;
    let tokens_per_page = physical.tokens_per_page();
    {
        let page = physical.page_mut(physical_page_id)?;
        page.push_token(k_row, v_row)?;
        if page.is_full(tokens_per_page) {
            page.seal();
            sealed = true;
        }
    }

    let physical_page = physical.page(physical_page_id)?.clone();
    if allocated_new_virtual_page {
        virtual_table.sync_physical(physical_page_id, &physical_page);
    } else {
        virtual_table.remap(virtual_page_id, physical_page_id, &physical_page)?;
    }
    if sealed {
        seq.layers[layer].live_by_kv_head[kv_head] = None;
    }

    Ok(AppendPageResult {
        physical_page_id,
        virtual_page_id,
        sealed_now: sealed,
    })
}

fn allocate_virtual_page(
    physical: &mut PagedKvCache,
    virtual_table: &mut VirtualPageTable,
    seq: &mut VirtualSeqCache,
    layer: usize,
    kv_head: usize,
    pos: u32,
) -> Result<(PageId, VirtualPageId, bool)> {
    let page_id =
        physical.push_detached_page(KvPage::new(layer, kv_head, pos, physical.head_dim())?);
    let page = physical.page(page_id)?.clone();
    let virtual_page_id = virtual_table.map_physical(page_id, &page);
    seq.layers[layer].virtual_pages_by_kv_head[kv_head].push(virtual_page_id);
    seq.layers[layer].live_by_kv_head[kv_head] = Some(virtual_page_id);
    Ok((page_id, virtual_page_id, true))
}

fn clone_physical_page(physical: &mut PagedKvCache, physical_page_id: PageId) -> Result<PageId> {
    let page = physical.page(physical_page_id)?.clone();
    Ok(physical.push_detached_page(page))
}

fn validate_seq_shape(
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

fn validate_seq_slot(seq: &VirtualSeqCache, layer: usize, kv_head: usize) -> Result<()> {
    if layer >= seq.layers.len() {
        return Err(RuntimeError::InvalidLayer {
            layer,
            layer_count: seq.layers.len(),
        });
    }
    let kv_head_count = seq.layers[layer].virtual_pages_by_kv_head.len();
    if kv_head >= kv_head_count {
        return Err(RuntimeError::InvalidKvHead {
            kv_head,
            kv_head_count,
        });
    }
    Ok(())
}

fn expected_next_pos(
    seq: &VirtualSeqCache,
    virtual_table: &VirtualPageTable,
    layer: usize,
    kv_head: usize,
) -> Result<Option<u32>> {
    let Some(&last_virtual_page_id) = seq.layers[layer].virtual_pages_by_kv_head[kv_head].last()
    else {
        return Ok(None);
    };
    Ok(Some(
        virtual_table
            .virtual_page(last_virtual_page_id)?
            .token_end(),
    ))
}
