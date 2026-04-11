#[cfg(feature = "qwen35-minimal")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candle_core::Device;
    use dotcache_paged_runtime::{
        MinimalQwen35LoadMode, MinimalQwen35Runner, Result, RuntimeError,
    };
    use serde::Serialize;
    use std::path::Path;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    #[derive(Clone, Debug)]
    enum DeviceSelector {
        Cpu,
        Cuda(usize),
        Hip(usize),
    }

    impl std::fmt::Display for DeviceSelector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Cpu => f.write_str("cpu"),
                Self::Cuda(ordinal) => write!(f, "cuda:{ordinal}"),
                Self::Hip(ordinal) => write!(f, "hip:{ordinal}"),
            }
        }
    }

    impl std::str::FromStr for DeviceSelector {
        type Err = RuntimeError;

        fn from_str(value: &str) -> Result<Self> {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized == "cpu" {
                return Ok(Self::Cpu);
            }
            if let Some(rest) = normalized.strip_prefix("cuda") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| RuntimeError::External {
                        context: "device",
                        message: format!("invalid cuda device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Cuda(ordinal));
            }
            if let Some(rest) = normalized.strip_prefix("hip") {
                let ordinal = rest
                    .strip_prefix(':')
                    .map(|ordinal| ordinal.parse::<usize>())
                    .transpose()
                    .map_err(|err| RuntimeError::External {
                        context: "device",
                        message: format!("invalid hip device ordinal in `{value}`: {err}"),
                    })?
                    .unwrap_or(0);
                return Ok(Self::Hip(ordinal));
            }
            Err(RuntimeError::External {
                context: "device",
                message: format!(
                    "unsupported device `{value}`, expected cpu, cuda[:ordinal], or hip[:ordinal]"
                ),
            })
        }
    }

    impl DeviceSelector {
        fn resolve(&self) -> Result<Device> {
            match self {
                Self::Cpu => Ok(Device::Cpu),
                Self::Cuda(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-cuda")]
                    {
                        Ok(Device::new_cuda(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-cuda"))]
                    {
                        Err(RuntimeError::BackendUnavailable {
                            backend: "cuda",
                            device: format!("cuda:{ordinal}"),
                        })
                    }
                }
                Self::Hip(ordinal) => {
                    #[cfg(feature = "qwen35-minimal-hip")]
                    {
                        Ok(Device::new_hip(*ordinal)?)
                    }
                    #[cfg(not(feature = "qwen35-minimal-hip"))]
                    {
                        Err(RuntimeError::BackendUnavailable {
                            backend: "hip",
                            device: format!("hip:{ordinal}"),
                        })
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum LoadMode {
        Native,
        Direct,
    }

    impl std::fmt::Display for LoadMode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Native => f.write_str("native"),
                Self::Direct => f.write_str("direct"),
            }
        }
    }

    impl std::str::FromStr for LoadMode {
        type Err = RuntimeError;

        fn from_str(value: &str) -> Result<Self> {
            match value.trim().to_ascii_lowercase().as_str() {
                "native" => Ok(Self::Native),
                "direct" => Ok(Self::Direct),
                other => Err(RuntimeError::External {
                    context: "load-mode",
                    message: format!("unsupported load mode `{other}`, expected native or direct"),
                }),
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        model_id: String,
        device: DeviceSelector,
        mode: LoadMode,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        model_id: String,
        device: String,
        load_mode: String,
        load_millis: f64,
        tokenizer_load_millis: f64,
        peak_rss_kib: u64,
        current_rss_kib: u64,
        package_root: Option<String>,
        tokenizer_path: String,
        revision: String,
    }

    fn parse_args() -> Result<Args> {
        let mut args = std::env::args().skip(1);
        let model_id = args.next().ok_or_else(|| RuntimeError::External {
            context: "hf_qwen35_minimal_load_bench",
            message:
                "usage: hf_qwen35_minimal_load_bench <model_id> [--device cpu|cuda[:n]|hip[:n]] [--mode native|direct]"
                    .to_string(),
        })?;
        let mut device = DeviceSelector::Cpu;
        let mut mode = LoadMode::Native;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--device" => {
                    let value = args.next().ok_or_else(|| RuntimeError::External {
                        context: "hf_qwen35_minimal_load_bench",
                        message: "missing value for --device".to_string(),
                    })?;
                    device = value.parse()?;
                }
                "--mode" => {
                    let value = args.next().ok_or_else(|| RuntimeError::External {
                        context: "hf_qwen35_minimal_load_bench",
                        message: "missing value for --mode".to_string(),
                    })?;
                    mode = value.parse()?;
                }
                other => {
                    return Err(RuntimeError::External {
                        context: "hf_qwen35_minimal_load_bench",
                        message: format!("unexpected argument `{other}`"),
                    });
                }
            }
        }
        Ok(Args {
            model_id,
            device,
            mode,
        })
    }

    fn read_proc_status_value_kib(key: &str) -> Result<u64> {
        let status = std::fs::read_to_string("/proc/self/status")?;
        let line = status
            .lines()
            .find(|line| line.starts_with(key))
            .ok_or_else(|| RuntimeError::External {
                context: "proc-status",
                message: format!("missing {key} in /proc/self/status"),
            })?;
        let value = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| RuntimeError::External {
                context: "proc-status",
                message: format!("malformed {key} line: {line}"),
            })?
            .parse::<u64>()
            .map_err(|err| RuntimeError::External {
                context: "proc-status",
                message: format!("invalid {key} value: {err}"),
            })?;
        Ok(value)
    }

    let args = parse_args()?;
    let device = args.device.resolve()?;
    let load_started = Instant::now();
    let runner = match args.mode {
        LoadMode::Native => MinimalQwen35Runner::load_with_mode(
            &args.model_id,
            &device,
            MinimalQwen35LoadMode::NativeStore,
        )?,
        LoadMode::Direct => MinimalQwen35Runner::load_from_hf_direct_f16(&args.model_id, &device)?,
    };
    let load_millis = load_started.elapsed().as_secs_f64() * 1000.0;
    let tokenizer_started = Instant::now();
    let _tokenizer = Tokenizer::from_file(&runner.weights.tokenizer_path)?;
    let tokenizer_load_millis = tokenizer_started.elapsed().as_secs_f64() * 1000.0;
    let peak_rss_kib = read_proc_status_value_kib("VmHWM:")?;
    let current_rss_kib = read_proc_status_value_kib("VmRSS:")?;

    let package_root = if runner.weights.package_root.as_os_str().is_empty() {
        None
    } else {
        Some(
            Path::new(&runner.weights.package_root)
                .display()
                .to_string(),
        )
    };

    let summary = Summary {
        model_id: runner.weights.model_id.clone(),
        device: args.device.to_string(),
        load_mode: args.mode.to_string(),
        load_millis,
        tokenizer_load_millis,
        peak_rss_kib,
        current_rss_kib,
        package_root,
        tokenizer_path: runner.weights.tokenizer_path.display().to_string(),
        revision: runner.weights.revision.clone(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(not(feature = "qwen35-minimal"))]
fn main() {
    eprintln!("hf_qwen35_minimal_load_bench requires --features qwen35-minimal");
    std::process::exit(1);
}
