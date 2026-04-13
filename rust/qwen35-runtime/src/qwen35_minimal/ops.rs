use candle_core::shape::Dim;
use candle_core::{Result, Tensor, D};

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    let dim = D::Minus1.to_index(xs.shape(), "softmax-last-dim")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    _softcapping: f32,
) -> Result<Tensor> {
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let mut attn = (q.matmul(&k_t)? * scale as f64)?;
    if let Some(mask) = mask {
        attn = attn.broadcast_add(mask)?;
    }
    if do_causal {
        let (.., q_len, kv_len) = attn.dims4()?;
        let device = attn.device();
        let mut data = vec![0f32; q_len * kv_len];
        for q_idx in 0..q_len {
            for k_idx in 0..kv_len {
                if k_idx > q_idx {
                    data[q_idx * kv_len + k_idx] = f32::NEG_INFINITY;
                }
            }
        }
        let causal = Tensor::from_vec(data, (1, 1, q_len, kv_len), device)?;
        attn = attn.broadcast_add(&causal)?;
    }
    let attn = softmax_last_dim(&attn)?;
    attn.matmul(v)
}
