use candle_core::Result;
use candle_nn::{linear, linear_no_bias as candle_linear_no_bias, VarBuilder};

pub use candle_nn::Linear;

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    candle_linear_no_bias(in_dim, out_dim, vb)
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        candle_linear_no_bias(in_dim, out_dim, vb)
    }
}
