use half::f16;

use crate::{Result, RuntimeError};

pub type PageId = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct KvPage {
    pub layer: u16,
    pub kv_head: u16,
    pub token_start: u32,
    pub token_count: u16,
    pub sealed: bool,
    pub head_dim: u16,
    pub k: Vec<f16>,
    pub v: Vec<f16>,
}

impl KvPage {
    pub fn new(layer: usize, kv_head: usize, token_start: u32, head_dim: usize) -> Result<Self> {
        Ok(Self {
            layer: u16::try_from(layer).map_err(|_| RuntimeError::ConversionOverflow {
                field: "layer",
                value: layer,
            })?,
            kv_head: u16::try_from(kv_head).map_err(|_| RuntimeError::ConversionOverflow {
                field: "kv_head",
                value: kv_head,
            })?,
            token_start,
            token_count: 0,
            sealed: false,
            head_dim: u16::try_from(head_dim).map_err(|_| RuntimeError::ConversionOverflow {
                field: "head_dim",
                value: head_dim,
            })?,
            k: Vec::new(),
            v: Vec::new(),
        })
    }

    pub fn head_dim_usize(&self) -> usize {
        usize::from(self.head_dim)
    }

    pub fn token_len(&self) -> usize {
        usize::from(self.token_count)
    }

    pub fn token_end(&self) -> u32 {
        self.token_start + u32::from(self.token_count)
    }

    pub fn expected_buffer_len(&self) -> usize {
        self.token_len() * self.head_dim_usize()
    }

    pub fn kv_byte_len(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f16>()
    }

    pub fn is_full(&self, tokens_per_page: usize) -> bool {
        self.token_len() >= tokens_per_page
    }

    pub fn seal(&mut self) {
        self.sealed = true;
    }

    pub fn push_token(&mut self, k_row: &[f32], v_row: &[f32]) -> Result<()> {
        if self.sealed {
            return Err(RuntimeError::SealedPage {
                layer: self.layer,
                kv_head: self.kv_head,
            });
        }

        let head_dim = self.head_dim_usize();
        if k_row.len() != head_dim {
            return Err(RuntimeError::DimensionMismatch {
                context: "key row",
                expected: head_dim,
                got: k_row.len(),
            });
        }
        if v_row.len() != head_dim {
            return Err(RuntimeError::DimensionMismatch {
                context: "value row",
                expected: head_dim,
                got: v_row.len(),
            });
        }

        self.k.extend(k_row.iter().copied().map(f16::from_f32));
        self.v.extend(v_row.iter().copied().map(f16::from_f32));

        let next_token_count = self.token_len() + 1;
        self.token_count =
            u16::try_from(next_token_count).map_err(|_| RuntimeError::ConversionOverflow {
                field: "token_count",
                value: next_token_count,
            })?;
        Ok(())
    }

    pub fn validate_layout(&self, page_id: PageId) -> Result<()> {
        let expected = self.expected_buffer_len();
        if self.k.len() != expected {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "k",
                expected,
                got: self.k.len(),
            });
        }
        if self.v.len() != expected {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "v",
                expected,
                got: self.v.len(),
            });
        }
        Ok(())
    }

    pub fn key_row(&self, token_index: usize) -> &[f16] {
        let width = self.head_dim_usize();
        let start = token_index * width;
        &self.k[start..start + width]
    }

    pub fn value_row(&self, token_index: usize) -> &[f16] {
        let width = self.head_dim_usize();
        let start = token_index * width;
        &self.v[start..start + width]
    }
}
