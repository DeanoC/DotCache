use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::{Init, VarBuilder};
use std::path::Path;

#[derive(Clone)]
pub(crate) struct WeightBuilder {
    inner: VarBuilder<'static>,
}

impl WeightBuilder {
    pub(crate) unsafe fn from_mmaped_safetensors<P: AsRef<Path>>(
        paths: &[P],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: VarBuilder::from_mmaped_safetensors(paths, dtype, device)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_varmap(
        varmap: &candle_nn::VarMap,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self {
            inner: VarBuilder::from_varmap(varmap, dtype, device),
        }
    }

    pub(crate) fn pp<S: ToString>(&self, s: S) -> Self {
        Self {
            inner: self.inner.pp(s),
        }
    }

    pub(crate) fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        self.inner.get(s, name)
    }

    pub(crate) fn get_with_hints<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        init: Init,
    ) -> Result<Tensor> {
        self.inner.get_with_hints(s, name, init)
    }

    pub(crate) fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }

    pub(crate) fn device(&self) -> &Device {
        self.inner.device()
    }

    pub(crate) fn dtype(&self) -> DType {
        self.inner.dtype()
    }
}
