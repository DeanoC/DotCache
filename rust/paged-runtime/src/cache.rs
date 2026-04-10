use crate::page::{KvPage, PageId};
use crate::page_mode::{PageModePolicy, PageSideKind};
use crate::{Result, RuntimeError};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LayerCache {
    pub pages_by_kv_head: Vec<Vec<PageId>>,
    pub live_by_kv_head: Vec<Option<PageId>>,
}

impl LayerCache {
    pub fn new(kv_head_count: usize) -> Self {
        Self {
            pages_by_kv_head: vec![Vec::new(); kv_head_count],
            live_by_kv_head: vec![None; kv_head_count],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SeqCache {
    pub layers: Vec<LayerCache>,
}

impl SeqCache {
    pub fn new(layer_count: usize, kv_head_count: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| LayerCache::new(kv_head_count))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PageStore {
    pages: Vec<Option<KvPage>>,
    free_list: Vec<PageId>,
    spilled_pages: HashMap<PageId, SpilledKvPage>,
    device_only_pages: HashMap<PageId, DeviceOnlyKvPage>,
}

#[derive(Clone, Debug)]
struct SpilledKvPage {
    page: KvPage,
}

impl SpilledKvPage {
    fn from_page(page: KvPage) -> Self {
        Self { page }
    }

    fn restore(self) -> KvPage {
        self.page
    }

    fn token_len(&self) -> usize {
        self.page.token_len()
    }

    fn kv_byte_len(&self) -> usize {
        self.page.kv_byte_len()
    }
}

#[derive(Clone, Debug)]
struct DeviceOnlyKvPage {
    token_count: u16,
}

impl DeviceOnlyKvPage {
    fn from_page(page: &KvPage) -> Self {
        Self {
            token_count: page.token_count,
        }
    }

    fn token_len(&self) -> usize {
        usize::from(self.token_count)
    }
}

impl PageStore {
    pub fn len(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
            + self.spilled_pages.len()
            + self.device_only_pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &KvPage> {
        self.pages.iter().filter_map(|page| page.as_ref())
    }

    pub fn iter_with_ids(&self) -> impl Iterator<Item = (PageId, &KvPage)> {
        self.pages
            .iter()
            .enumerate()
            .filter_map(|(page_id, page)| page.as_ref().map(|page| (page_id, page)))
    }

    pub fn resident_page_count(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
    }

    pub fn spilled_page_count(&self) -> usize {
        self.spilled_pages.len()
    }

    pub fn device_only_page_count(&self) -> usize {
        self.device_only_pages.len()
    }

    pub fn total_token_count(&self) -> usize {
        self.pages
            .iter()
            .filter_map(|page| page.as_ref())
            .map(KvPage::token_len)
            .sum::<usize>()
            + self
                .spilled_pages
                .values()
                .map(SpilledKvPage::token_len)
                .sum::<usize>()
            + self
                .device_only_pages
                .values()
                .map(DeviceOnlyKvPage::token_len)
                .sum::<usize>()
    }

    pub fn resident_byte_count(&self) -> usize {
        self.pages
            .iter()
            .filter_map(|page| page.as_ref())
            .map(KvPage::kv_byte_len)
            .sum()
    }

    pub fn spilled_byte_count(&self) -> usize {
        self.spilled_pages
            .values()
            .map(SpilledKvPage::kv_byte_len)
            .sum()
    }

    pub fn total_byte_count(&self) -> usize {
        self.resident_byte_count() + self.spilled_byte_count()
    }

    pub fn is_spilled(&self, page_id: PageId) -> bool {
        self.spilled_pages.contains_key(&page_id)
    }

    pub fn is_device_only(&self, page_id: PageId) -> bool {
        self.device_only_pages.contains_key(&page_id)
    }

    pub fn page(&self, page_id: PageId) -> Result<&KvPage> {
        self.pages
            .get(page_id)
            .and_then(|page| page.as_ref())
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count: self.pages.len(),
            })
    }

    pub(crate) fn page_mut(&mut self, page_id: PageId) -> Result<&mut KvPage> {
        if self.device_only_pages.contains_key(&page_id) {
            return Err(RuntimeError::External {
                context: "page_store",
                message: format!("page {page_id} is device-only and has no mutable host payload"),
            });
        }
        if self.spilled_pages.contains_key(&page_id) {
            self.restore(page_id)?;
        }
        let page_count = self.pages.len();
        self.pages
            .get_mut(page_id)
            .and_then(|page| page.as_mut())
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count,
            })
    }

    pub(crate) fn push(&mut self, page: KvPage) -> PageId {
        if let Some(page_id) = self.free_list.pop() {
            self.pages[page_id] = Some(page);
            page_id
        } else {
            let page_id = self.pages.len();
            self.pages.push(Some(page));
            page_id
        }
    }

    pub(crate) fn reclaim(&mut self, page_id: PageId) -> Result<KvPage> {
        if let Some(page) = self.spilled_pages.remove(&page_id) {
            self.free_list.push(page_id);
            return Ok(page.restore());
        }
        let page_count = self.pages.len();
        let slot = self
            .pages
            .get_mut(page_id)
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count,
            })?;
        let page = slot.take().ok_or(RuntimeError::InvalidPageId {
            page_id,
            page_count,
        })?;
        self.free_list.push(page_id);
        Ok(page)
    }

    pub(crate) fn reclaim_slot(&mut self, page_id: PageId) -> Result<()> {
        if self.spilled_pages.remove(&page_id).is_some()
            || self.device_only_pages.remove(&page_id).is_some()
        {
            self.free_list.push(page_id);
            return Ok(());
        }

        let page_count = self.pages.len();
        let slot = self
            .pages
            .get_mut(page_id)
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count,
            })?;
        slot.take().ok_or(RuntimeError::InvalidPageId {
            page_id,
            page_count,
        })?;
        self.free_list.push(page_id);
        Ok(())
    }

    pub(crate) fn spill(&mut self, page_id: PageId) -> Result<bool> {
        if self.spilled_pages.contains_key(&page_id) {
            return Ok(false);
        }

        let page = self.reclaim(page_id)?;
        if !page.sealed {
            self.pages[page_id] = Some(page);
            if let Some(index) = self
                .free_list
                .iter()
                .position(|&free_page_id| free_page_id == page_id)
            {
                self.free_list.remove(index);
            }
            return Err(RuntimeError::External {
                context: "page_store",
                message: format!("cannot spill live page {page_id}"),
            });
        }
        if let Some(index) = self
            .free_list
            .iter()
            .position(|&free_page_id| free_page_id == page_id)
        {
            self.free_list.remove(index);
        }
        self.spilled_pages
            .insert(page_id, SpilledKvPage::from_page(page));
        Ok(true)
    }

    pub(crate) fn restore(&mut self, page_id: PageId) -> Result<bool> {
        let Some(page) = self.spilled_pages.remove(&page_id) else {
            return Ok(false);
        };
        if self.pages.len() <= page_id {
            self.pages.resize_with(page_id + 1, || None);
        }
        self.pages[page_id] = Some(page.restore());
        if let Some(index) = self
            .free_list
            .iter()
            .position(|&free_page_id| free_page_id == page_id)
        {
            self.free_list.remove(index);
        }
        Ok(true)
    }

    pub(crate) fn promote_device_only(&mut self, page_id: PageId) -> Result<bool> {
        if self.device_only_pages.contains_key(&page_id)
            || self.spilled_pages.contains_key(&page_id)
        {
            return Ok(false);
        }

        let page_count = self.pages.len();
        let page = self
            .pages
            .get_mut(page_id)
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count,
            })?
            .take()
            .ok_or(RuntimeError::InvalidPageId {
                page_id,
                page_count,
            })?;

        if !page.sealed {
            self.pages[page_id] = Some(page);
            return Err(RuntimeError::External {
                context: "page_store",
                message: format!("cannot promote live page {page_id} to device-only"),
            });
        }

        self.device_only_pages
            .insert(page_id, DeviceOnlyKvPage::from_page(&page));
        Ok(true)
    }
}

#[derive(Clone, Debug)]
pub struct PagedKvCache {
    tokens_per_page: usize,
    head_dim: usize,
    page_mode_policy: PageModePolicy,
    seq: SeqCache,
    store: PageStore,
}

impl PagedKvCache {
    pub fn new(
        layer_count: usize,
        kv_head_count: usize,
        tokens_per_page: usize,
        head_dim: usize,
    ) -> Self {
        Self::new_with_page_mode_policy(
            layer_count,
            kv_head_count,
            tokens_per_page,
            head_dim,
            PageModePolicy::default(),
        )
    }

    pub fn new_with_page_mode_policy(
        layer_count: usize,
        kv_head_count: usize,
        tokens_per_page: usize,
        head_dim: usize,
        page_mode_policy: PageModePolicy,
    ) -> Self {
        assert!(tokens_per_page > 0, "tokens_per_page must be positive");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            tokens_per_page,
            head_dim,
            page_mode_policy,
            seq: SeqCache::new(layer_count, kv_head_count),
            store: PageStore::default(),
        }
    }

    pub fn tokens_per_page(&self) -> usize {
        self.tokens_per_page
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn seq(&self) -> &SeqCache {
        &self.seq
    }

    pub fn page_mode_policy(&self) -> &PageModePolicy {
        &self.page_mode_policy
    }

    pub fn set_page_mode_policy(&mut self, policy: PageModePolicy) {
        self.page_mode_policy = policy;
    }

    pub fn store(&self) -> &PageStore {
        &self.store
    }

    pub fn resident_page_count(&self) -> usize {
        self.store.resident_page_count()
    }

    pub fn spilled_page_count(&self) -> usize {
        self.store.spilled_page_count()
    }

    pub fn device_only_page_count(&self) -> usize {
        self.store.device_only_page_count()
    }

    pub fn resident_byte_count(&self) -> usize {
        self.store.resident_byte_count()
    }

    pub fn spilled_byte_count(&self) -> usize {
        self.store.spilled_byte_count()
    }

    pub fn total_byte_count(&self) -> usize {
        self.store.total_byte_count()
    }

    pub fn page(&self, page_id: PageId) -> Result<&KvPage> {
        self.store.page(page_id)
    }

    pub fn live_page_id(&self, layer: usize, kv_head: usize) -> Result<Option<PageId>> {
        self.validate_slot(layer, kv_head)?;
        Ok(self.seq.layers[layer].live_by_kv_head[kv_head])
    }

    pub fn page_ids(&self, layer: usize, kv_head: usize) -> Result<&[PageId]> {
        self.validate_slot(layer, kv_head)?;
        Ok(self.seq.layers[layer].pages_by_kv_head[kv_head].as_slice())
    }

    pub fn append_token(
        &mut self,
        layer: usize,
        kv_head: usize,
        pos: u32,
        k_row: &[f32],
        v_row: &[f32],
    ) -> Result<PageId> {
        self.validate_slot(layer, kv_head)?;

        if let Some(expected_pos) = self.expected_next_pos(layer, kv_head)? {
            if pos != expected_pos {
                return Err(RuntimeError::PositionMismatch {
                    expected: expected_pos,
                    got: pos,
                });
            }
        }

        let live_page_id = self.seq.layers[layer].live_by_kv_head[kv_head];
        let page_id = match live_page_id {
            Some(page_id) if !self.store.page(page_id)?.sealed => page_id,
            _ => self.allocate_page(layer, kv_head, pos)?,
        };

        let mut sealed = false;
        {
            let page = self.store.page_mut(page_id)?;
            page.push_token(k_row, v_row)?;
            if page.is_full(self.tokens_per_page) {
                page.seal()?;
                sealed = true;
            }
        }

        if sealed {
            self.seq.layers[layer].live_by_kv_head[kv_head] = None;
        }

        Ok(page_id)
    }

    pub fn into_parts(self) -> (PageStore, SeqCache) {
        (self.store, self.seq)
    }

    pub(crate) fn page_mut(&mut self, page_id: PageId) -> Result<&mut KvPage> {
        self.store.page_mut(page_id)
    }

    pub(crate) fn push_detached_page(&mut self, page: KvPage) -> PageId {
        self.store.push(page)
    }

    pub(crate) fn reclaim_slot(&mut self, page_id: PageId) -> Result<()> {
        self.store.reclaim_slot(page_id)
    }

    pub(crate) fn spill_page(&mut self, page_id: PageId) -> Result<bool> {
        self.store.spill(page_id)
    }

    pub(crate) fn restore_page(&mut self, page_id: PageId) -> Result<bool> {
        self.store.restore(page_id)
    }

    pub(crate) fn promote_device_only_page(&mut self, page_id: PageId) -> Result<bool> {
        self.store.promote_device_only(page_id)
    }

    fn validate_slot(&self, layer: usize, kv_head: usize) -> Result<()> {
        if layer >= self.seq.layers.len() {
            return Err(RuntimeError::InvalidLayer {
                layer,
                layer_count: self.seq.layers.len(),
            });
        }
        let kv_head_count = self.seq.layers[layer].pages_by_kv_head.len();
        if kv_head >= kv_head_count {
            return Err(RuntimeError::InvalidKvHead {
                kv_head,
                kv_head_count,
            });
        }
        Ok(())
    }

    fn expected_next_pos(&self, layer: usize, kv_head: usize) -> Result<Option<u32>> {
        let page_ids = self.page_ids(layer, kv_head)?;
        let Some(&last_page_id) = page_ids.last() else {
            return Ok(None);
        };
        Ok(Some(self.store.page(last_page_id)?.token_end()))
    }

    fn allocate_page(&mut self, layer: usize, kv_head: usize, pos: u32) -> Result<PageId> {
        let page = KvPage::new_with_modes(
            layer,
            kv_head,
            pos,
            self.head_dim,
            self.page_mode_policy.resolve(PageSideKind::Key, layer),
            self.page_mode_policy.resolve(PageSideKind::Value, layer),
        )?;
        let page_id = self.store.push(page);
        self.seq.layers[layer].pages_by_kv_head[kv_head].push(page_id);
        self.seq.layers[layer].live_by_kv_head[kv_head] = Some(page_id);
        Ok(page_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spilled_page_ids_are_not_reused_for_new_pages() {
        let mut store = PageStore::default();
        let mut spilled_page = KvPage::new(0, 0, 0, 4).expect("page");
        spilled_page.seal().expect("seal");
        let spilled_page_id = store.push(spilled_page);

        assert!(store.spill(spilled_page_id).expect("spill page"));
        assert!(store.is_spilled(spilled_page_id));

        let replacement_page_id = store.push(KvPage::new(0, 0, 16, 4).expect("replacement page"));

        assert_ne!(replacement_page_id, spilled_page_id);
        assert!(store.is_spilled(spilled_page_id));
        assert_eq!(
            store
                .page(replacement_page_id)
                .expect("replacement")
                .token_start,
            16
        );
    }

    #[test]
    fn device_only_page_ids_are_not_reused_for_new_pages() {
        let mut store = PageStore::default();
        let mut sealed_page = KvPage::new(0, 0, 0, 4).expect("page");
        sealed_page.seal().expect("seal");
        let page_id = store.push(sealed_page);

        assert!(store.promote_device_only(page_id).expect("promote page"));
        assert!(store.is_device_only(page_id));

        let replacement_page_id = store.push(KvPage::new(0, 0, 16, 4).expect("replacement page"));

        assert_ne!(replacement_page_id, page_id);
        assert!(store.is_device_only(page_id));
        assert_eq!(
            store
                .page(replacement_page_id)
                .expect("replacement")
                .token_start,
            16
        );
    }
}
