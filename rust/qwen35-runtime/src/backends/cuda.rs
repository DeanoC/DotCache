use crate::{Qwen35Backend, Qwen35BackendDescriptor};
use dotcache_runtime_core::{BackendKind, TargetSpec};

pub fn descriptor(target: TargetSpec) -> Qwen35BackendDescriptor {
    debug_assert!(matches!(target.backend, BackendKind::Cuda));
    Qwen35BackendDescriptor {
        target,
        optimized: true,
    }
}

pub fn backend(target: TargetSpec) -> Qwen35Backend {
    Qwen35Backend {
        descriptor: descriptor(target),
    }
}

