use candle_core::{Result, Tensor, D};

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle_core::bail!("inconsistent batch size in rope {b_sz} {cs:?}")
            }
            Ok((t, d))
        }
        _ => candle_core::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle_core::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }

    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = if cos.rank() == 3 {
        cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?
    } else {
        cos
    };
    let sin = if sin.rank() == 3 {
        sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?
    } else {
        sin
    };
    let x = xs.contiguous()?.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    Tensor::cat(&[y0, y1], D::Minus1)?.flatten_from(D::Minus2)
}
