use half::f16;

use crate::page_mode::{PageModeSpec, PageModeTag, PageQuantScheme, PageSideKind};
use crate::{Result, RuntimeError};

pub type PageId = usize;

#[derive(Clone, Debug, PartialEq)]
struct DensePageData {
    values: Vec<f16>,
}

impl DensePageData {
    fn dense_storage_f32(&self) -> Vec<f32> {
        self.values.iter().map(|value| value.to_f32()).collect()
    }
}

fn lower_bound(values: &[f32], target: f32) -> usize {
    values.partition_point(|value| *value < target)
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped = q.clamp(0.0, 1.0);
    let position = clamped * (sorted.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = position - lower as f32;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn normalize_in_place(values: &mut [f32]) {
    let norm = l2_norm(values).max(1e-6);
    for value in values {
        *value /= norm;
    }
}

fn fwht_in_place(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let width = values.len();
    if width & (width - 1) != 0 {
        return Err(RuntimeError::External {
            context: "page",
            message: format!("FWHT requires a power-of-two width, got {width}"),
        });
    }
    let mut step = 1;
    while step < width {
        let block = step * 2;
        for base in (0..width).step_by(block) {
            for offset in 0..step {
                let left = values[base + offset];
                let right = values[base + step + offset];
                values[base + offset] = left + right;
                values[base + step + offset] = left - right;
            }
        }
        step = block;
    }
    let norm = (width as f32).sqrt();
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn mat_vec_mul(matrix: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row_index in 0..rows {
        let row_start = row_index * cols;
        out[row_index] = matrix[row_start..row_start + cols]
            .iter()
            .zip(vector.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum();
    }
    out
}

fn top_basis_from_covariance(
    centered: &[f32],
    token_count: usize,
    group_size: usize,
    rank: usize,
) -> Vec<f32> {
    let mut covariance = vec![0.0f32; group_size * group_size];
    for row_index in 0..token_count {
        let row = &centered[row_index * group_size..(row_index + 1) * group_size];
        for lhs in 0..group_size {
            for rhs in 0..group_size {
                covariance[lhs * group_size + rhs] += row[lhs] * row[rhs];
            }
        }
    }

    let mut basis = vec![0.0f32; rank * group_size];
    for basis_index in 0..rank {
        let mut vector = vec![0.0f32; group_size];
        vector[basis_index % group_size] = 1.0;
        for _ in 0..24 {
            let mut next = mat_vec_mul(&covariance, group_size, group_size, &vector);
            for prev_basis_index in 0..basis_index {
                let prev =
                    &basis[prev_basis_index * group_size..(prev_basis_index + 1) * group_size];
                let projection: f32 = next
                    .iter()
                    .zip(prev.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum();
                for (next_value, prev_value) in next.iter_mut().zip(prev.iter()) {
                    *next_value -= projection * prev_value;
                }
            }
            if next.iter().all(|value| value.abs() < 1e-9) {
                next[basis_index % group_size] = 1.0;
            }
            normalize_in_place(&mut next);
            vector = next;
        }
        basis[basis_index * group_size..(basis_index + 1) * group_size].copy_from_slice(&vector);
    }
    basis
}

fn hadamard_basis(group_size: usize, rank: usize) -> Result<Vec<f32>> {
    if group_size & (group_size - 1) != 0 {
        return Err(RuntimeError::External {
            context: "page",
            message: format!("Hadamard basis requires a power-of-two group size, got {group_size}"),
        });
    }
    let mut rows = Vec::with_capacity(rank * group_size);
    for row_index in 1..=rank {
        let mut row = vec![0.0f32; group_size];
        row[row_index % group_size] = 1.0;
        fwht_in_place(&mut row)?;
        rows.extend(row);
    }
    Ok(rows)
}

#[derive(Clone, Debug, PartialEq)]
struct M0PageData {
    padded_head_dim: usize,
    num_groups: usize,
    group_size: usize,
    bits: u8,
    quant_scheme: PageQuantScheme,
    codes: Vec<u8>,
    scales: Vec<f32>,
    bias: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
struct M1PageData {
    padded_head_dim: usize,
    num_groups: usize,
    group_size: usize,
    bits: u8,
    levels: usize,
    codes: Vec<u8>,
    codebooks: Vec<f32>,
}

impl M1PageData {
    fn encode(
        values: &[f16],
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        let bits = spec.bits();
        if bits == 0 || bits > 8 {
            return Err(RuntimeError::UnsupportedPageMode {
                mode: spec.describe(),
                context: "M1 encoding bit width",
            });
        }
        let group_size = spec.group_size().max(1);
        let num_groups = head_dim.div_ceil(group_size);
        let padded_head_dim = num_groups * group_size;
        let levels = 1usize << bits;
        let mut codes = vec![0u8; token_count * padded_head_dim];
        let mut codebooks = vec![0.0f32; num_groups * levels];
        let quantiles = (0..levels)
            .map(|index| index as f32 / (levels.saturating_sub(1).max(1)) as f32)
            .collect::<Vec<_>>();

        for group_index in 0..num_groups {
            let group_start = group_index * group_size;
            let mut flat_values = Vec::with_capacity(token_count * group_size);
            for token_index in 0..token_count {
                let row_start = token_index * head_dim;
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    flat_values.push(if dim_index < head_dim {
                        values[row_start + dim_index].to_f32()
                    } else {
                        0.0
                    });
                }
            }
            flat_values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
            let lut_start = group_index * levels;
            for (level_index, q) in quantiles.iter().copied().enumerate() {
                codebooks[lut_start + level_index] = quantile(&flat_values, q);
            }

            for _ in 0..6 {
                let current_lut = &codebooks[lut_start..lut_start + levels];
                let boundaries = current_lut
                    .windows(2)
                    .map(|window| (window[0] + window[1]) * 0.5)
                    .collect::<Vec<_>>();
                let mut sums = vec![0.0f32; levels];
                let mut counts = vec![0usize; levels];
                for &value in &flat_values {
                    let code = lower_bound(&boundaries, value).min(levels - 1);
                    sums[code] += value;
                    counts[code] += 1;
                }
                let mut changed = false;
                for level_index in 0..levels {
                    if counts[level_index] > 0 {
                        let updated = sums[level_index] / counts[level_index] as f32;
                        if (updated - codebooks[lut_start + level_index]).abs() > 1e-6 {
                            changed = true;
                        }
                        codebooks[lut_start + level_index] = updated;
                    }
                }
                if !changed {
                    break;
                }
            }

            let boundaries = codebooks[lut_start..lut_start + levels]
                .windows(2)
                .map(|window| (window[0] + window[1]) * 0.5)
                .collect::<Vec<_>>();
            for token_index in 0..token_count {
                let row_start = token_index * head_dim;
                let code_row_start = token_index * padded_head_dim + group_start;
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    let value = if dim_index < head_dim {
                        values[row_start + dim_index].to_f32()
                    } else {
                        0.0
                    };
                    let code = lower_bound(&boundaries, value).min(levels - 1);
                    codes[code_row_start + offset] = code as u8;
                }
            }
        }

        Ok(Self {
            padded_head_dim,
            num_groups,
            group_size,
            bits,
            levels,
            codes,
            codebooks,
        })
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected_codes = token_count * self.padded_head_dim;
        if self.codes.len() != expected_codes {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "lut_codes",
                expected: expected_codes,
                got: self.codes.len(),
            });
        }
        let expected_codebooks = self.num_groups * self.levels;
        if self.codebooks.len() != expected_codebooks {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "lut_codebooks",
                expected: expected_codebooks,
                got: self.codebooks.len(),
            });
        }
        if self.padded_head_dim < head_dim {
            return Err(RuntimeError::External {
                context: "page",
                message: format!(
                    "page {page_id} padded_head_dim {} is smaller than head_dim {head_dim}",
                    self.padded_head_dim
                ),
            });
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.codes.len() * std::mem::size_of::<u8>()
            + self.codebooks.len() * std::mem::size_of::<f32>()
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            let lut_start = group_index * self.levels;
            for dim_index in group_start..row_end {
                let code = self.codes[row_code_start + (dim_index - group_start)] as usize;
                out.push(self.codebooks[lut_start + code]);
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            let lut_start = group_index * self.levels;
            for dim_index in group_start..row_end {
                let code = self.codes[row_code_start + (dim_index - group_start)] as usize;
                total += self.codebooks[lut_start + code] * query[dim_index];
            }
        }
        total
    }

    fn mix_row(&self, token_index: usize, head_dim: usize, weight: f32, out: &mut [f32]) {
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            let lut_start = group_index * self.levels;
            for dim_index in group_start..row_end {
                let code = self.codes[row_code_start + (dim_index - group_start)] as usize;
                out[dim_index] += weight * self.codebooks[lut_start + code];
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum M3Payload {
    F16(Vec<f16>),
    I8 { values: Vec<i8>, scales: Vec<f16> },
}

#[derive(Clone, Debug, PartialEq)]
struct M3PageData {
    payload: M3Payload,
}

impl M3PageData {
    fn encode(
        values: Vec<f16>,
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        match spec
            .escape_dtype()
            .unwrap_or(crate::page_mode::PageEscapeDType::Float16)
        {
            crate::page_mode::PageEscapeDType::Float16 => Ok(Self {
                payload: M3Payload::F16(values),
            }),
            crate::page_mode::PageEscapeDType::Int8 => {
                let mut quantized = vec![0i8; values.len()];
                let mut scales = Vec::with_capacity(token_count);
                let eps = 1e-8f32;
                for (token_index, row) in
                    values.chunks_exact(head_dim).enumerate().take(token_count)
                {
                    let max_abs = row
                        .iter()
                        .map(|value| value.to_f32().abs())
                        .fold(0.0f32, f32::max);
                    let scale = (max_abs / 127.0).max(eps);
                    scales.push(f16::from_f32(scale));
                    let row_start = token_index * head_dim;
                    for (out, value) in quantized[row_start..row_start + head_dim]
                        .iter_mut()
                        .zip(row.iter())
                    {
                        let scaled = (value.to_f32() / scale).round().clamp(-127.0, 127.0);
                        *out = scaled as i8;
                    }
                }
                Ok(Self {
                    payload: M3Payload::I8 {
                        values: quantized,
                        scales,
                    },
                })
            }
        }
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected = token_count * head_dim;
        match &self.payload {
            M3Payload::F16(values) => {
                if values.len() != expected {
                    return Err(RuntimeError::PageBufferMismatch {
                        page_id,
                        buffer: "escape_values",
                        expected,
                        got: values.len(),
                    });
                }
            }
            M3Payload::I8 { values, scales } => {
                if values.len() != expected {
                    return Err(RuntimeError::PageBufferMismatch {
                        page_id,
                        buffer: "escape_values",
                        expected,
                        got: values.len(),
                    });
                }
                if scales.len() != token_count {
                    return Err(RuntimeError::PageBufferMismatch {
                        page_id,
                        buffer: "escape_scales",
                        expected: token_count,
                        got: scales.len(),
                    });
                }
            }
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        match &self.payload {
            M3Payload::F16(values) => values.len() * std::mem::size_of::<f16>(),
            M3Payload::I8 { values, scales } => {
                values.len() * std::mem::size_of::<i8>() + scales.len() * std::mem::size_of::<f16>()
            }
        }
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        match &self.payload {
            M3Payload::F16(values) => {
                let start = token_index * head_dim;
                out.extend(
                    values[start..start + head_dim]
                        .iter()
                        .map(|value| value.to_f32()),
                );
            }
            M3Payload::I8 { values, scales } => {
                let scale = scales[token_index].to_f32();
                let start = token_index * head_dim;
                out.extend(
                    values[start..start + head_dim]
                        .iter()
                        .map(|value| *value as f32 * scale),
                );
            }
        }
    }

    fn dense_storage_f32(&self, token_count: usize, head_dim: usize) -> Vec<f32> {
        let total = token_count * head_dim;
        match &self.payload {
            M3Payload::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
            M3Payload::I8 { values, scales } => {
                let mut out = vec![0.0f32; total];
                for token_index in 0..token_count {
                    let scale = scales[token_index].to_f32();
                    let start = token_index * head_dim;
                    let end = start + head_dim;
                    for (out_value, value) in out[start..end].iter_mut().zip(values[start..end].iter())
                    {
                        *out_value = *value as f32 * scale;
                    }
                }
                out
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        match &self.payload {
            M3Payload::F16(values) => {
                let start = token_index * head_dim;
                query
                    .iter()
                    .zip(values[start..start + head_dim].iter())
                    .map(|(lhs, rhs)| lhs * rhs.to_f32())
                    .sum()
            }
            M3Payload::I8 { values, scales } => {
                let scale = scales[token_index].to_f32();
                let start = token_index * head_dim;
                query
                    .iter()
                    .zip(values[start..start + head_dim].iter())
                    .map(|(lhs, rhs)| lhs * (*rhs as f32 * scale))
                    .sum()
            }
        }
    }

    fn mix_row(&self, token_index: usize, head_dim: usize, weight: f32, out: &mut [f32]) {
        match &self.payload {
            M3Payload::F16(values) => {
                let start = token_index * head_dim;
                for (out_value, value) in out.iter_mut().zip(values[start..start + head_dim].iter())
                {
                    *out_value += weight * value.to_f32();
                }
            }
            M3Payload::I8 { values, scales } => {
                let scale = scales[token_index].to_f32();
                let start = token_index * head_dim;
                for (out_value, value) in out.iter_mut().zip(values[start..start + head_dim].iter())
                {
                    *out_value += weight * (*value as f32 * scale);
                }
            }
        }
    }
}

const TURBO3_CENTROIDS: [f32; 8] = [-1.863, -1.318, -0.912, -0.522, 0.185, 0.603, 1.016, 1.594];

#[derive(Clone, Debug, PartialEq)]
struct T3PageData {
    padded_head_dim: usize,
    num_groups: usize,
    group_size: usize,
    codes: Vec<u8>,
    corrections: Vec<f16>,
}

impl T3PageData {
    fn encode(
        values: &[f16],
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        let group_size = spec.group_size().max(1);
        if group_size & (group_size - 1) != 0 {
            return Err(RuntimeError::External {
                context: "page sealing",
                message: format!("T3 requires a power-of-two group size, got {group_size}"),
            });
        }
        let num_groups = head_dim.div_ceil(group_size);
        let padded_head_dim = num_groups * group_size;
        let mut codes = vec![0u8; token_count * padded_head_dim];
        let mut corrections = vec![f16::from_f32(1.0); token_count * num_groups];

        for token_index in 0..token_count {
            let row_start = token_index * head_dim;
            for group_index in 0..num_groups {
                let group_start = group_index * group_size;
                let mut group = vec![0.0f32; group_size];
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    if dim_index < head_dim {
                        group[offset] = values[row_start + dim_index].to_f32();
                    }
                }
                fwht_in_place(&mut group)?;
                let norm = l2_norm(&group);
                let denom = norm.max(1e-6);
                let mut reconstructed_norm_sq = 0.0f32;
                let code_row_start = token_index * padded_head_dim + group_start;
                for offset in 0..group_size {
                    let normalized = group[offset] / denom;
                    let mut best_index = 0usize;
                    let mut best_delta = f32::INFINITY;
                    for (centroid_index, centroid) in TURBO3_CENTROIDS.iter().copied().enumerate() {
                        let delta = (normalized - centroid).abs();
                        if delta < best_delta {
                            best_delta = delta;
                            best_index = centroid_index;
                        }
                    }
                    codes[code_row_start + offset] = best_index as u8;
                    let centroid = TURBO3_CENTROIDS[best_index];
                    reconstructed_norm_sq += centroid * centroid;
                }
                let recon_norm = reconstructed_norm_sq.sqrt().max(1e-6);
                corrections[token_index * num_groups + group_index] =
                    f16::from_f32(norm / recon_norm);
            }
        }

        Ok(Self {
            padded_head_dim,
            num_groups,
            group_size,
            codes,
            corrections,
        })
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected_codes = token_count * self.padded_head_dim;
        if self.codes.len() != expected_codes {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "turbo3_codes",
                expected: expected_codes,
                got: self.codes.len(),
            });
        }
        let expected_corrections = token_count * self.num_groups;
        if self.corrections.len() != expected_corrections {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "turbo3_corrections",
                expected: expected_corrections,
                got: self.corrections.len(),
            });
        }
        if self.padded_head_dim < head_dim {
            return Err(RuntimeError::External {
                context: "page",
                message: format!(
                    "page {page_id} padded_head_dim {} is smaller than head_dim {head_dim}",
                    self.padded_head_dim
                ),
            });
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.codes.len() * std::mem::size_of::<u8>()
            + self.corrections.len() * std::mem::size_of::<f16>()
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let code_row_start = token_index * self.padded_head_dim + group_start;
            let correction = self.corrections[token_index * self.num_groups + group_index].to_f32();
            let mut rotated = vec![0.0f32; self.group_size];
            for offset in 0..self.group_size {
                rotated[offset] =
                    TURBO3_CENTROIDS[self.codes[code_row_start + offset] as usize] * correction;
            }
            let _ = fwht_in_place(&mut rotated);
            let row_end = (group_start + self.group_size).min(head_dim);
            for dim_index in group_start..row_end {
                out.push(rotated[dim_index - group_start]);
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_end = (group_start + self.group_size).min(head_dim);
            let mut padded_query = vec![0.0f32; self.group_size];
            for dim_index in group_start..row_end {
                padded_query[dim_index - group_start] = query[dim_index];
            }
            let _ = fwht_in_place(&mut padded_query);
            let correction = self.corrections[token_index * self.num_groups + group_index].to_f32();
            let code_row_start = token_index * self.padded_head_dim + group_start;
            for offset in 0..self.group_size {
                total += TURBO3_CENTROIDS[self.codes[code_row_start + offset] as usize]
                    * correction
                    * padded_query[offset];
            }
        }
        total
    }

    fn mix_row(&self, token_index: usize, head_dim: usize, weight: f32, out: &mut [f32]) {
        let mut dense = Vec::with_capacity(head_dim);
        self.row_to_f32(token_index, head_dim, &mut dense);
        for (out_value, value) in out.iter_mut().zip(dense.into_iter()) {
            *out_value += weight * value;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct M2PageData {
    padded_head_dim: usize,
    num_groups: usize,
    group_size: usize,
    rank: usize,
    coeffs: Vec<f16>,
    basis: Vec<f32>,
    mean: Vec<f32>,
}

impl M2PageData {
    fn encode(
        values: &[f16],
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        let group_size = spec.group_size().max(1);
        let num_groups = head_dim.div_ceil(group_size);
        let padded_head_dim = num_groups * group_size;
        let rank = 8usize.min(group_size).min(token_count.max(1));
        let mut coeffs = vec![f16::from_f32(0.0); token_count * num_groups * rank];
        let mut basis = vec![0.0f32; num_groups * rank * group_size];
        let mut mean = vec![0.0f32; num_groups * group_size];

        for group_index in 0..num_groups {
            let group_start = group_index * group_size;
            let mut matrix = vec![0.0f32; token_count * group_size];
            for token_index in 0..token_count {
                let row_start = token_index * head_dim;
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    matrix[token_index * group_size + offset] = if dim_index < head_dim {
                        values[row_start + dim_index].to_f32()
                    } else {
                        0.0
                    };
                }
            }
            for offset in 0..group_size {
                let mut sum = 0.0f32;
                for token_index in 0..token_count {
                    sum += matrix[token_index * group_size + offset];
                }
                mean[group_index * group_size + offset] = sum / token_count.max(1) as f32;
            }
            let mut centered = matrix.clone();
            for token_index in 0..token_count {
                for offset in 0..group_size {
                    centered[token_index * group_size + offset] -=
                        mean[group_index * group_size + offset];
                }
            }

            let group_basis = top_basis_from_covariance(&centered, token_count, group_size, rank);
            basis[group_index * rank * group_size..(group_index + 1) * rank * group_size]
                .copy_from_slice(&group_basis);

            for token_index in 0..token_count {
                let row = &centered[token_index * group_size..(token_index + 1) * group_size];
                for basis_index in 0..rank {
                    let basis_row =
                        &group_basis[basis_index * group_size..(basis_index + 1) * group_size];
                    let coefficient: f32 = row
                        .iter()
                        .zip(basis_row.iter())
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum();
                    coeffs[(token_index * num_groups + group_index) * rank + basis_index] =
                        f16::from_f32(coefficient);
                }
            }
        }

        Ok(Self {
            padded_head_dim,
            num_groups,
            group_size,
            rank,
            coeffs,
            basis,
            mean,
        })
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected_coeffs = token_count * self.num_groups * self.rank;
        if self.coeffs.len() != expected_coeffs {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m2_coeffs",
                expected: expected_coeffs,
                got: self.coeffs.len(),
            });
        }
        let expected_basis = self.num_groups * self.rank * self.group_size;
        if self.basis.len() != expected_basis {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m2_basis",
                expected: expected_basis,
                got: self.basis.len(),
            });
        }
        let expected_mean = self.num_groups * self.group_size;
        if self.mean.len() != expected_mean {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m2_mean",
                expected: expected_mean,
                got: self.mean.len(),
            });
        }
        if self.padded_head_dim < head_dim {
            return Err(RuntimeError::External {
                context: "page",
                message: format!(
                    "page {page_id} padded_head_dim {} is smaller than head_dim {head_dim}",
                    self.padded_head_dim
                ),
            });
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.coeffs.len() * std::mem::size_of::<f16>()
            + self.basis.len() * std::mem::size_of::<f32>()
            + self.mean.len() * std::mem::size_of::<f32>()
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        for group_index in 0..self.num_groups {
            let mean =
                &self.mean[group_index * self.group_size..(group_index + 1) * self.group_size];
            let basis = &self.basis[group_index * self.rank * self.group_size
                ..(group_index + 1) * self.rank * self.group_size];
            let coeffs = &self.coeffs[(token_index * self.num_groups + group_index) * self.rank
                ..(token_index * self.num_groups + group_index + 1) * self.rank];
            let mut group = mean.to_vec();
            for basis_index in 0..self.rank {
                let coefficient = coeffs[basis_index].to_f32();
                let basis_row =
                    &basis[basis_index * self.group_size..(basis_index + 1) * self.group_size];
                for (group_value, basis_value) in group.iter_mut().zip(basis_row.iter()) {
                    *group_value += coefficient * basis_value;
                }
            }
            let group_start = group_index * self.group_size;
            let row_end = (group_start + self.group_size).min(head_dim);
            for dim_index in group_start..row_end {
                out.push(group[dim_index - group_start]);
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_end = (group_start + self.group_size).min(head_dim);
            let q_group = &query[group_start..row_end];
            let mean = &self.mean
                [group_index * self.group_size..group_index * self.group_size + q_group.len()];
            total += mean
                .iter()
                .zip(q_group.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let basis = &self.basis[group_index * self.rank * self.group_size
                ..(group_index + 1) * self.rank * self.group_size];
            let coeffs = &self.coeffs[(token_index * self.num_groups + group_index) * self.rank
                ..(token_index * self.num_groups + group_index + 1) * self.rank];
            for basis_index in 0..self.rank {
                let basis_row = &basis
                    [basis_index * self.group_size..basis_index * self.group_size + q_group.len()];
                let q_proj: f32 = basis_row
                    .iter()
                    .zip(q_group.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum();
                total += coeffs[basis_index].to_f32() * q_proj;
            }
        }
        total
    }
}

#[derive(Clone, Debug, PartialEq)]
struct M4PageData {
    padded_head_dim: usize,
    num_groups: usize,
    group_size: usize,
    rank: usize,
    coeffs: Vec<f16>,
    basis: Vec<f32>,
    mean: Vec<f32>,
}

impl M4PageData {
    fn encode(
        values: &[f16],
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        let group_size = spec.group_size().max(1);
        let num_groups = head_dim.div_ceil(group_size);
        let padded_head_dim = num_groups * group_size;
        let rank = 8usize.min(group_size.saturating_sub(1).max(1));
        let fixed_basis = hadamard_basis(group_size, rank)?;
        let mut coeffs = vec![f16::from_f32(0.0); token_count * num_groups * rank];
        let mut mean = vec![0.0f32; num_groups * group_size];

        for group_index in 0..num_groups {
            let group_start = group_index * group_size;
            let mut matrix = vec![0.0f32; token_count * group_size];
            for token_index in 0..token_count {
                let row_start = token_index * head_dim;
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    matrix[token_index * group_size + offset] = if dim_index < head_dim {
                        values[row_start + dim_index].to_f32()
                    } else {
                        0.0
                    };
                }
            }
            for offset in 0..group_size {
                let mut sum = 0.0f32;
                for token_index in 0..token_count {
                    sum += matrix[token_index * group_size + offset];
                }
                mean[group_index * group_size + offset] = sum / token_count.max(1) as f32;
            }
            for token_index in 0..token_count {
                for basis_index in 0..rank {
                    let basis_row =
                        &fixed_basis[basis_index * group_size..(basis_index + 1) * group_size];
                    let row = &matrix[token_index * group_size..(token_index + 1) * group_size];
                    let coefficient: f32 = row
                        .iter()
                        .zip(mean[group_index * group_size..(group_index + 1) * group_size].iter())
                        .zip(basis_row.iter())
                        .map(|((value, mean), basis)| (value - mean) * basis)
                        .sum();
                    coeffs[(token_index * num_groups + group_index) * rank + basis_index] =
                        f16::from_f32(coefficient);
                }
            }
        }

        Ok(Self {
            padded_head_dim,
            num_groups,
            group_size,
            rank,
            coeffs,
            basis: fixed_basis,
            mean,
        })
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected_coeffs = token_count * self.num_groups * self.rank;
        if self.coeffs.len() != expected_coeffs {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m4_coeffs",
                expected: expected_coeffs,
                got: self.coeffs.len(),
            });
        }
        let expected_basis = self.rank * self.group_size;
        if self.basis.len() != expected_basis {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m4_basis",
                expected: expected_basis,
                got: self.basis.len(),
            });
        }
        let expected_mean = self.num_groups * self.group_size;
        if self.mean.len() != expected_mean {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "m4_mean",
                expected: expected_mean,
                got: self.mean.len(),
            });
        }
        if self.padded_head_dim < head_dim {
            return Err(RuntimeError::External {
                context: "page",
                message: format!(
                    "page {page_id} padded_head_dim {} is smaller than head_dim {head_dim}",
                    self.padded_head_dim
                ),
            });
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.coeffs.len() * std::mem::size_of::<f16>()
            + self.basis.len() * std::mem::size_of::<f32>()
            + self.mean.len() * std::mem::size_of::<f32>()
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        for group_index in 0..self.num_groups {
            let mean =
                &self.mean[group_index * self.group_size..(group_index + 1) * self.group_size];
            let coeffs = &self.coeffs[(token_index * self.num_groups + group_index) * self.rank
                ..(token_index * self.num_groups + group_index + 1) * self.rank];
            let mut group = mean.to_vec();
            for basis_index in 0..self.rank {
                let coefficient = coeffs[basis_index].to_f32();
                let basis_row =
                    &self.basis[basis_index * self.group_size..(basis_index + 1) * self.group_size];
                for (group_value, basis_value) in group.iter_mut().zip(basis_row.iter()) {
                    *group_value += coefficient * basis_value;
                }
            }
            let group_start = group_index * self.group_size;
            let row_end = (group_start + self.group_size).min(head_dim);
            for dim_index in group_start..row_end {
                out.push(group[dim_index - group_start]);
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_end = (group_start + self.group_size).min(head_dim);
            let q_group = &query[group_start..row_end];
            let mean = &self.mean
                [group_index * self.group_size..group_index * self.group_size + q_group.len()];
            total += mean
                .iter()
                .zip(q_group.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let coeffs = &self.coeffs[(token_index * self.num_groups + group_index) * self.rank
                ..(token_index * self.num_groups + group_index + 1) * self.rank];
            for basis_index in 0..self.rank {
                let basis_row = &self.basis
                    [basis_index * self.group_size..basis_index * self.group_size + q_group.len()];
                let q_proj: f32 = basis_row
                    .iter()
                    .zip(q_group.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum();
                total += coeffs[basis_index].to_f32() * q_proj;
            }
        }
        total
    }
}

impl M0PageData {
    fn encode(
        values: &[f16],
        token_count: usize,
        head_dim: usize,
        spec: &PageModeSpec,
    ) -> Result<Self> {
        let bits = spec.bits();
        if bits == 0 || bits > 8 {
            return Err(RuntimeError::UnsupportedPageMode {
                mode: spec.describe(),
                context: "M0 encoding bit width",
            });
        }
        let group_size = spec.group_size().max(1);
        let num_groups = head_dim.div_ceil(group_size);
        let padded_head_dim = num_groups * group_size;
        let mut codes = vec![0u8; token_count * padded_head_dim];
        let mut scales = vec![0.0f32; token_count * num_groups];
        let mut bias = match spec.quant_scheme() {
            PageQuantScheme::Affine => Some(vec![0.0f32; token_count * num_groups]),
            PageQuantScheme::Symmetric => None,
        };
        let qmax_affine = ((1u32 << bits) - 1) as f32;
        let qmax_symmetric = ((1u32 << (bits - 1)) - 1) as f32;
        let zero_point = qmax_symmetric;
        let eps = 1e-8f32;

        for token_index in 0..token_count {
            let row_start = token_index * head_dim;
            for group_index in 0..num_groups {
                let group_start = group_index * group_size;
                let mut group_values = vec![0.0f32; group_size];
                for offset in 0..group_size {
                    let dim_index = group_start + offset;
                    if dim_index < head_dim {
                        group_values[offset] = values[row_start + dim_index].to_f32();
                    }
                }

                let (scale, group_bias) = match spec.quant_scheme() {
                    PageQuantScheme::Affine => {
                        let x_min = group_values.iter().copied().fold(f32::INFINITY, f32::min);
                        let x_max = group_values
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        (
                            ((x_max - x_min) / qmax_affine.max(1.0)).max(eps),
                            Some(x_min),
                        )
                    }
                    PageQuantScheme::Symmetric => {
                        let max_abs = group_values
                            .iter()
                            .copied()
                            .map(f32::abs)
                            .fold(0.0, f32::max);
                        ((max_abs / qmax_symmetric.max(1.0)).max(eps), None)
                    }
                };

                let scale_index = token_index * num_groups + group_index;
                scales[scale_index] = scale;
                if let Some(group_bias) = group_bias {
                    bias.as_mut().expect("affine bias")[scale_index] = group_bias;
                }

                let code_row_start = token_index * padded_head_dim + group_start;
                for (offset, value) in group_values.into_iter().enumerate() {
                    let code = match spec.quant_scheme() {
                        PageQuantScheme::Affine => {
                            let shifted = (value - group_bias.expect("affine bias")) / scale;
                            shifted.round().clamp(0.0, qmax_affine) as u8
                        }
                        PageQuantScheme::Symmetric => {
                            let signed = (value / scale)
                                .round()
                                .clamp(-qmax_symmetric, qmax_symmetric);
                            (signed + zero_point).clamp(0.0, ((1u32 << bits) - 1) as f32) as u8
                        }
                    };
                    codes[code_row_start + offset] = code;
                }
            }
        }

        Ok(Self {
            padded_head_dim,
            num_groups,
            group_size,
            bits,
            quant_scheme: spec.quant_scheme(),
            codes,
            scales,
            bias,
        })
    }

    fn validate_layout(&self, page_id: PageId, token_count: usize, head_dim: usize) -> Result<()> {
        let expected_codes = token_count * self.padded_head_dim;
        if self.codes.len() != expected_codes {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "codes",
                expected: expected_codes,
                got: self.codes.len(),
            });
        }
        let expected_scales = token_count * self.num_groups;
        if self.scales.len() != expected_scales {
            return Err(RuntimeError::PageBufferMismatch {
                page_id,
                buffer: "scales",
                expected: expected_scales,
                got: self.scales.len(),
            });
        }
        if let Some(bias) = &self.bias {
            if bias.len() != expected_scales {
                return Err(RuntimeError::PageBufferMismatch {
                    page_id,
                    buffer: "bias",
                    expected: expected_scales,
                    got: bias.len(),
                });
            }
        }
        if self.padded_head_dim < head_dim {
            return Err(RuntimeError::External {
                context: "page",
                message: format!(
                    "page {page_id} padded_head_dim {} is smaller than head_dim {head_dim}",
                    self.padded_head_dim
                ),
            });
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        let mut total = self.codes.len() * std::mem::size_of::<u8>();
        total += self.scales.len() * std::mem::size_of::<f32>();
        if let Some(bias) = &self.bias {
            total += bias.len() * std::mem::size_of::<f32>();
        }
        total
    }

    fn row_to_f32(&self, token_index: usize, head_dim: usize, out: &mut Vec<f32>) {
        out.reserve(head_dim);
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let scale = self.scales[token_index * self.num_groups + group_index];
            let bias = self
                .bias
                .as_ref()
                .map(|bias| bias[token_index * self.num_groups + group_index])
                .unwrap_or(0.0);
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            for dim_index in group_start..row_end {
                let code = self.codes[row_code_start + (dim_index - group_start)] as f32;
                let value = match self.quant_scheme {
                    PageQuantScheme::Affine => scale * code + bias,
                    PageQuantScheme::Symmetric => {
                        let zero_point = ((1u32 << (self.bits - 1)) - 1) as f32;
                        scale * (code - zero_point)
                    }
                };
                out.push(value);
            }
        }
    }

    fn score_row(&self, token_index: usize, head_dim: usize, query: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            let mut int_dot = 0.0f32;
            let mut query_sum = 0.0f32;
            for dim_index in group_start..row_end {
                let q = query[dim_index];
                int_dot += self.codes[row_code_start + (dim_index - group_start)] as f32 * q;
                query_sum += q;
            }
            let scale = self.scales[token_index * self.num_groups + group_index];
            match self.quant_scheme {
                PageQuantScheme::Affine => {
                    let bias = self.bias.as_ref().expect("affine bias")
                        [token_index * self.num_groups + group_index];
                    total += scale * int_dot + bias * query_sum;
                }
                PageQuantScheme::Symmetric => {
                    let zero_point = ((1u32 << (self.bits - 1)) - 1) as f32;
                    total += scale * (int_dot - zero_point * query_sum);
                }
            }
        }
        total
    }

    fn mix_row(&self, token_index: usize, head_dim: usize, weight: f32, out: &mut [f32]) {
        for group_index in 0..self.num_groups {
            let group_start = group_index * self.group_size;
            let row_code_start = token_index * self.padded_head_dim + group_start;
            let row_end = (group_start + self.group_size).min(head_dim);
            let scale = self.scales[token_index * self.num_groups + group_index];
            let bias = self
                .bias
                .as_ref()
                .map(|bias| bias[token_index * self.num_groups + group_index])
                .unwrap_or(0.0);
            for dim_index in group_start..row_end {
                let code = self.codes[row_code_start + (dim_index - group_start)] as f32;
                let value = match self.quant_scheme {
                    PageQuantScheme::Affine => scale * code + bias,
                    PageQuantScheme::Symmetric => {
                        let zero_point = ((1u32 << (self.bits - 1)) - 1) as f32;
                        scale * (code - zero_point)
                    }
                };
                out[dim_index] += weight * value;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum PageSideStorage {
    LiveDense(DensePageData),
    Exact(DensePageData),
    M0(M0PageData),
    M1(M1PageData),
    M3(M3PageData),
    T3(T3PageData),
    M2(M2PageData),
    M4(M4PageData),
}

impl PageSideStorage {
    fn is_exact_fused_compatible(&self) -> bool {
        matches!(self, Self::Exact(_) | Self::LiveDense(_))
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PageSide {
    mode: PageModeSpec,
    storage: PageSideStorage,
}

impl PageSide {
    fn new(mode: PageModeSpec) -> Self {
        Self {
            mode,
            storage: PageSideStorage::LiveDense(DensePageData { values: Vec::new() }),
        }
    }

    fn live_values_mut(&mut self) -> Result<&mut Vec<f16>> {
        match &mut self.storage {
            PageSideStorage::LiveDense(data) => Ok(&mut data.values),
            _ => Err(RuntimeError::External {
                context: "page",
                message: format!("cannot append to sealed {} page side", self.mode.describe()),
            }),
        }
    }

    fn validate_layout(
        &self,
        page_id: PageId,
        side: PageSideKind,
        token_count: usize,
        head_dim: usize,
    ) -> Result<()> {
        let expected = token_count * head_dim;
        match &self.storage {
            PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => {
                if data.values.len() != expected {
                    return Err(RuntimeError::PageBufferMismatch {
                        page_id,
                        buffer: side.as_str(),
                        expected,
                        got: data.values.len(),
                    });
                }
            }
            PageSideStorage::M0(data) => data.validate_layout(page_id, token_count, head_dim)?,
            PageSideStorage::M1(data) => data.validate_layout(page_id, token_count, head_dim)?,
            PageSideStorage::M3(data) => data.validate_layout(page_id, token_count, head_dim)?,
            PageSideStorage::T3(data) => data.validate_layout(page_id, token_count, head_dim)?,
            PageSideStorage::M2(data) => data.validate_layout(page_id, token_count, head_dim)?,
            PageSideStorage::M4(data) => data.validate_layout(page_id, token_count, head_dim)?,
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        match &self.storage {
            PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => {
                data.values.len() * std::mem::size_of::<f16>()
            }
            PageSideStorage::M0(data) => data.byte_len(),
            PageSideStorage::M1(data) => data.byte_len(),
            PageSideStorage::M3(data) => data.byte_len(),
            PageSideStorage::T3(data) => data.byte_len(),
            PageSideStorage::M2(data) => data.byte_len(),
            PageSideStorage::M4(data) => data.byte_len(),
        }
    }

    fn seal(&mut self, token_count: usize, head_dim: usize, side: PageSideKind) -> Result<()> {
        self.mode.validate_for_side(side)?;
        let storage = std::mem::replace(
            &mut self.storage,
            PageSideStorage::LiveDense(DensePageData { values: Vec::new() }),
        );
        let PageSideStorage::LiveDense(dense) = storage else {
            self.storage = storage;
            return Ok(());
        };
        if self.mode.tag() == PageModeTag::Exact {
            self.storage = PageSideStorage::Exact(dense);
            return Ok(());
        }
        let sealed_storage = match self.mode.tag() {
            PageModeTag::M0 => M0PageData::encode(&dense.values, token_count, head_dim, &self.mode)
                .map(PageSideStorage::M0),
            PageModeTag::M1 => M1PageData::encode(&dense.values, token_count, head_dim, &self.mode)
                .map(PageSideStorage::M1),
            PageModeTag::M3 => M3PageData::encode(dense.values.clone(), token_count, head_dim, &self.mode)
                .map(PageSideStorage::M3),
            PageModeTag::T3 => T3PageData::encode(&dense.values, token_count, head_dim, &self.mode)
                .map(PageSideStorage::T3),
            PageModeTag::M2 => M2PageData::encode(&dense.values, token_count, head_dim, &self.mode)
                .map(PageSideStorage::M2),
            PageModeTag::M4 => M4PageData::encode(&dense.values, token_count, head_dim, &self.mode)
                .map(PageSideStorage::M4),
            PageModeTag::Exact => unreachable!("exact pages return early before encoding"),
        };
        self.storage = match sealed_storage {
            Ok(storage) => storage,
            Err(err) => {
                self.storage = PageSideStorage::LiveDense(dense);
                return Err(err);
            }
        };
        Ok(())
    }

    fn row_f32(&self, token_index: usize, head_dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(head_dim);
        match &self.storage {
            PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => {
                let start = token_index * head_dim;
                out.extend(
                    data.values[start..start + head_dim]
                        .iter()
                        .map(|value| value.to_f32()),
                );
            }
            PageSideStorage::M0(data) => data.row_to_f32(token_index, head_dim, &mut out),
            PageSideStorage::M1(data) => data.row_to_f32(token_index, head_dim, &mut out),
            PageSideStorage::M3(data) => data.row_to_f32(token_index, head_dim, &mut out),
            PageSideStorage::T3(data) => data.row_to_f32(token_index, head_dim, &mut out),
            PageSideStorage::M2(data) => data.row_to_f32(token_index, head_dim, &mut out),
            PageSideStorage::M4(data) => data.row_to_f32(token_index, head_dim, &mut out),
        }
        out
    }

    fn dense_storage_f32(&self, token_count: usize, head_dim: usize) -> Vec<f32> {
        match &self.storage {
            PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => data.dense_storage_f32(),
            PageSideStorage::M3(data) => data.dense_storage_f32(token_count, head_dim),
            _ => {
                let mut out = Vec::with_capacity(token_count * head_dim);
                for token_index in 0..token_count {
                    out.extend(self.row_f32(token_index, head_dim));
                }
                out
            }
        }
    }

    fn score_rows(
        &self,
        token_count: usize,
        head_dim: usize,
        query: &[f32],
        logits_out: &mut Vec<f32>,
    ) {
        for token_index in 0..token_count {
            let score = match &self.storage {
                PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => {
                    let start = token_index * head_dim;
                    query
                        .iter()
                        .zip(data.values[start..start + head_dim].iter())
                        .map(|(lhs, rhs)| lhs * rhs.to_f32())
                        .sum()
                }
                PageSideStorage::M0(data) => data.score_row(token_index, head_dim, query),
                PageSideStorage::M1(data) => data.score_row(token_index, head_dim, query),
                PageSideStorage::M3(data) => data.score_row(token_index, head_dim, query),
                PageSideStorage::T3(data) => data.score_row(token_index, head_dim, query),
                PageSideStorage::M2(data) => data.score_row(token_index, head_dim, query),
                PageSideStorage::M4(data) => data.score_row(token_index, head_dim, query),
            };
            logits_out.push(score);
        }
    }

    fn mix_rows(&self, token_count: usize, head_dim: usize, weights: &[f32], out: &mut [f32]) {
        for (token_index, weight) in weights.iter().copied().enumerate().take(token_count) {
            match &self.storage {
                PageSideStorage::LiveDense(data) | PageSideStorage::Exact(data) => {
                    let start = token_index * head_dim;
                    for (out_value, value) in out
                        .iter_mut()
                        .zip(data.values[start..start + head_dim].iter())
                    {
                        *out_value += weight * value.to_f32();
                    }
                }
                PageSideStorage::M0(data) => data.mix_row(token_index, head_dim, weight, out),
                PageSideStorage::M1(data) => data.mix_row(token_index, head_dim, weight, out),
                PageSideStorage::M3(data) => data.mix_row(token_index, head_dim, weight, out),
                PageSideStorage::T3(data) => data.mix_row(token_index, head_dim, weight, out),
                PageSideStorage::M2(_) | PageSideStorage::M4(_) => {
                    unreachable!("value pages cannot use M2/M4 due to policy validation")
                }
            }
        }
    }

    fn mode(&self) -> &PageModeSpec {
        &self.mode
    }

    fn is_exact_fused_compatible(&self) -> bool {
        self.storage.is_exact_fused_compatible()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvPage {
    pub layer: u16,
    pub kv_head: u16,
    pub token_start: u32,
    pub token_count: u16,
    pub sealed: bool,
    pub head_dim: u16,
    key: PageSide,
    value: PageSide,
}

impl KvPage {
    pub fn new(layer: usize, kv_head: usize, token_start: u32, head_dim: usize) -> Result<Self> {
        Self::new_with_modes(
            layer,
            kv_head,
            token_start,
            head_dim,
            PageModeSpec::exact(),
            PageModeSpec::exact(),
        )
    }

    pub fn new_with_modes(
        layer: usize,
        kv_head: usize,
        token_start: u32,
        head_dim: usize,
        key_mode: PageModeSpec,
        value_mode: PageModeSpec,
    ) -> Result<Self> {
        key_mode.validate_for_side(PageSideKind::Key)?;
        value_mode.validate_for_side(PageSideKind::Value)?;
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
            key: PageSide::new(key_mode),
            value: PageSide::new(value_mode),
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
        self.key.byte_len() + self.value.byte_len()
    }

    pub fn is_full(&self, tokens_per_page: usize) -> bool {
        self.token_len() >= tokens_per_page
    }

    pub fn key_mode(&self) -> &PageModeSpec {
        self.key.mode()
    }

    pub fn value_mode(&self) -> &PageModeSpec {
        self.value.mode()
    }

    pub fn is_exact_fused_compatible(&self) -> bool {
        self.key.is_exact_fused_compatible() && self.value.is_exact_fused_compatible()
    }

    pub fn seal(&mut self) -> Result<()> {
        let token_count = self.token_len();
        let head_dim = self.head_dim_usize();
        self.key.seal(token_count, head_dim, PageSideKind::Key)?;
        self.value
            .seal(token_count, head_dim, PageSideKind::Value)?;
        self.sealed = true;
        Ok(())
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

        self.key
            .live_values_mut()?
            .extend(k_row.iter().copied().map(f16::from_f32));
        self.value
            .live_values_mut()?
            .extend(v_row.iter().copied().map(f16::from_f32));

        let next_token_count = self.token_len() + 1;
        self.token_count =
            u16::try_from(next_token_count).map_err(|_| RuntimeError::ConversionOverflow {
                field: "token_count",
                value: next_token_count,
            })?;
        Ok(())
    }

    pub fn validate_layout(&self, page_id: PageId) -> Result<()> {
        let token_count = self.token_len();
        let head_dim = self.head_dim_usize();
        self.key
            .validate_layout(page_id, PageSideKind::Key, token_count, head_dim)?;
        self.value
            .validate_layout(page_id, PageSideKind::Value, token_count, head_dim)?;
        Ok(())
    }

    pub fn key_row_f32(&self, token_index: usize) -> Vec<f32> {
        self.key.row_f32(token_index, self.head_dim_usize())
    }

    pub fn value_row_f32(&self, token_index: usize) -> Vec<f32> {
        self.value.row_f32(token_index, self.head_dim_usize())
    }

    pub fn dense_key_storage_f32(&self) -> Vec<f32> {
        self.key
            .dense_storage_f32(self.token_len(), self.head_dim_usize())
    }

    pub fn dense_value_storage_f32(&self) -> Vec<f32> {
        self.value
            .dense_storage_f32(self.token_len(), self.head_dim_usize())
    }

    pub fn score_keys(&self, query: &[f32], logits_out: &mut Vec<f32>) -> Result<()> {
        if query.len() != self.head_dim_usize() {
            return Err(RuntimeError::DimensionMismatch {
                context: "key score query",
                expected: self.head_dim_usize(),
                got: query.len(),
            });
        }
        self.key
            .score_rows(self.token_len(), self.head_dim_usize(), query, logits_out);
        Ok(())
    }

    pub fn mix_values(&self, weights: &[f32], out: &mut [f32]) -> Result<()> {
        if out.len() != self.head_dim_usize() {
            return Err(RuntimeError::DimensionMismatch {
                context: "value mix output",
                expected: self.head_dim_usize(),
                got: out.len(),
            });
        }
        if weights.len() != self.token_len() {
            return Err(RuntimeError::DimensionMismatch {
                context: "value mix weights",
                expected: self.token_len(),
                got: weights.len(),
            });
        }
        self.value
            .mix_rows(self.token_len(), self.head_dim_usize(), weights, out);
        Ok(())
    }
}
