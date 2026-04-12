use candle_core::{Module, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => super::ops::sigmoid(xs),
            Self::HardSigmoid => ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32),
            Self::Swiglu => {
                let xs = xs.chunk(2, candle_core::D::Minus1)?;
                &xs[0].silu()? * &xs[1]
            }
            Self::Swish => xs * super::ops::sigmoid(xs)?,
            Self::HardSwish => xs * (((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32))?,
            Self::Mish => xs * (1.0 + xs.exp()?)?.log()?.tanh()?,
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => {
                let zeros = xs.zeros_like()?;
                xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
            }
            Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}
