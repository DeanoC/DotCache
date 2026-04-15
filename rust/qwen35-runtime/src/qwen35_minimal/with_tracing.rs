use candle_core::{Module, Result, Tensor};
use super::backend_buffer_api;
use super::backend_ops;
use super::types::StateBuffer;
#[cfg(any(feature = "hf", test))]
use super::builder::WeightBuilder;
#[cfg(any(feature = "hf", test))]
use candle_nn::Init;

#[derive(Clone, Debug)]
pub struct Linear {
    pub(super) weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        backend_ops::linear_forward(x, &self.weight, self.bias.as_ref())
    }
}

impl Linear {
    pub fn forward_buffer(&self, x: &StateBuffer) -> Result<StateBuffer> {
        backend_buffer_api::for_device(x.device()).linear_forward(
            x,
            &self.weight,
            self.bias.as_ref(),
        )
    }

    pub fn forward_buffer_into_scratch(
        &self,
        x: &StateBuffer,
        scratch: &StateBuffer,
    ) -> Result<StateBuffer> {
        backend_buffer_api::for_device(x.device()).linear_forward_into_scratch(
            x,
            &self.weight,
            self.bias.as_ref(),
            scratch,
        )
    }
}

#[cfg(any(feature = "hf", test))]
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: WeightBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

#[cfg(any(feature = "hf", test))]
pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: WeightBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    if bias {
        let bound = 1. / (in_dim as f64).sqrt();
        let init_bs = Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
        Ok(Linear::new(ws, Some(bs)))
    } else {
        Ok(Linear::new(ws, None))
    }
}
