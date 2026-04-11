use std::fmt::Display;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::PreparedModelPackage;

#[derive(Debug, Clone)]
pub(crate) struct PreparedTensorSource {
    package: Arc<PreparedModelPackage>,
    device: Device,
    prefix: String,
}

impl PreparedTensorSource {
    pub(crate) fn new(package: Arc<PreparedModelPackage>, device: Device) -> Self {
        Self {
            package,
            device,
            prefix: String::new(),
        }
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn pp<T: Display>(&self, component: T) -> Self {
        let component = component.to_string();
        let prefix = if self.prefix.is_empty() {
            component
        } else {
            format!("{}.{}", self.prefix, component)
        };
        Self {
            package: self.package.clone(),
            device: self.device.clone(),
            prefix,
        }
    }

    pub(crate) fn get(&self, name: &str) -> Result<Tensor> {
        self.package
            .load_tensor(&self.full_name(name), &self.device)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))
    }

    pub(crate) fn contains_tensor(&self, name: &str) -> bool {
        self.package.contains_tensor(&self.full_name(name))
    }

    fn full_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else if name.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}
