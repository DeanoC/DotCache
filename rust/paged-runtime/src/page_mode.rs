use std::collections::BTreeMap;

use crate::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageSideKind {
    Key,
    Value,
}

impl PageSideKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Value => "value",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageQuantScheme {
    Affine,
    Symmetric,
}

impl PageQuantScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Symmetric => "symmetric",
        }
    }
}

impl std::str::FromStr for PageQuantScheme {
    type Err = RuntimeError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "affine" => Ok(Self::Affine),
            "symmetric" => Ok(Self::Symmetric),
            other => Err(RuntimeError::External {
                context: "page_mode",
                message: format!(
                    "unsupported quant scheme `{other}`, expected `affine` or `symmetric`"
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageEscapeDType {
    Float16,
    Int8,
}

impl PageEscapeDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Int8 => "int8",
        }
    }
}

impl std::str::FromStr for PageEscapeDType {
    type Err = RuntimeError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "float16" => Ok(Self::Float16),
            "int8" => Ok(Self::Int8),
            other => Err(RuntimeError::External {
                context: "page_mode",
                message: format!(
                    "unsupported escape dtype `{other}`, expected `float16` or `int8`"
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageModeTag {
    Exact,
    M0,
    M1,
    M2,
    M3,
    M4,
    T3,
}

impl PageModeTag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::M0 => "M0",
            Self::M1 => "M1",
            Self::M2 => "M2",
            Self::M3 => "M3",
            Self::M4 => "M4",
            Self::T3 => "T3",
        }
    }

    pub fn supports_mix(self) -> bool {
        !matches!(self, Self::M2 | Self::M4)
    }
}

impl std::str::FromStr for PageModeTag {
    type Err = RuntimeError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "exact" => Ok(Self::Exact),
            "m0" => Ok(Self::M0),
            "m1" => Ok(Self::M1),
            "m2" => Ok(Self::M2),
            "m3" => Ok(Self::M3),
            "m4" => Ok(Self::M4),
            "t3" => Ok(Self::T3),
            other => Err(RuntimeError::External {
                context: "page_mode",
                message: format!(
                    "unsupported page mode `{other}`, expected exact, M0, M1, M2, M3, M4, or T3"
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PageModeSpec {
    tag: PageModeTag,
    quant_scheme: PageQuantScheme,
    bits: u8,
    group_size: usize,
    escape_dtype: Option<PageEscapeDType>,
}

impl PageModeSpec {
    pub const DEFAULT_GROUP_SIZE: usize = 32;

    pub fn exact() -> Self {
        Self {
            tag: PageModeTag::Exact,
            quant_scheme: PageQuantScheme::Affine,
            bits: 16,
            group_size: Self::DEFAULT_GROUP_SIZE,
            escape_dtype: None,
        }
    }

    pub fn m0_affine(bits: u8) -> Self {
        Self {
            tag: PageModeTag::M0,
            quant_scheme: PageQuantScheme::Affine,
            bits,
            group_size: Self::DEFAULT_GROUP_SIZE,
            escape_dtype: None,
        }
    }

    pub fn tag(&self) -> PageModeTag {
        self.tag
    }

    pub fn quant_scheme(&self) -> PageQuantScheme {
        self.quant_scheme
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn escape_dtype(&self) -> Option<PageEscapeDType> {
        self.escape_dtype
    }

    pub fn is_exact(&self) -> bool {
        self.tag == PageModeTag::Exact
    }

    pub fn describe(&self) -> String {
        match self.tag {
            PageModeTag::Exact => "exact".to_string(),
            PageModeTag::M3 => format!(
                "{}/{}/{}/{}",
                self.tag.as_str(),
                self.quant_scheme.as_str(),
                self.bits,
                self.escape_dtype
                    .unwrap_or(PageEscapeDType::Float16)
                    .as_str()
            ),
            _ => format!(
                "{}/{}/{}",
                self.tag.as_str(),
                self.quant_scheme.as_str(),
                self.bits
            ),
        }
    }

    pub fn validate_for_side(&self, side: PageSideKind) -> Result<()> {
        if side == PageSideKind::Value && matches!(self.tag, PageModeTag::M2 | PageModeTag::M4) {
            return Err(RuntimeError::UnsupportedPageModeForValue {
                mode: self.tag.as_str().to_string(),
            });
        }
        Ok(())
    }
}

impl Default for PageModeSpec {
    fn default() -> Self {
        Self::exact()
    }
}

impl std::str::FromStr for PageModeSpec {
    type Err = RuntimeError;

    fn from_str(value: &str) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("exact") {
            return Ok(Self::exact());
        }

        let mut parts = trimmed.split('/');
        let tag = parts
            .next()
            .ok_or(RuntimeError::External {
                context: "page_mode",
                message: "missing page mode".to_string(),
            })?
            .parse::<PageModeTag>()?;
        let quant_scheme = parts
            .next()
            .ok_or(RuntimeError::External {
                context: "page_mode",
                message: format!("missing quant scheme in page mode `{trimmed}`"),
            })?
            .parse::<PageQuantScheme>()?;
        let bits = parts
            .next()
            .ok_or(RuntimeError::External {
                context: "page_mode",
                message: format!("missing bit width in page mode `{trimmed}`"),
            })?
            .parse::<u8>()
            .map_err(|err| RuntimeError::External {
                context: "page_mode",
                message: format!("invalid bit width in page mode `{trimmed}`: {err}"),
            })?;

        let escape_dtype = if tag == PageModeTag::M3 {
            Some(
                parts
                    .next()
                    .unwrap_or("float16")
                    .parse::<PageEscapeDType>()?,
            )
        } else {
            None
        };
        if parts.next().is_some() {
            return Err(RuntimeError::External {
                context: "page_mode",
                message: format!("unexpected extra components in page mode `{trimmed}`"),
            });
        }
        Ok(Self {
            tag,
            quant_scheme,
            bits,
            group_size: Self::DEFAULT_GROUP_SIZE,
            escape_dtype,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PageModePolicy {
    default_key: PageModeSpec,
    default_value: PageModeSpec,
    key_overrides: BTreeMap<usize, PageModeSpec>,
    value_overrides: BTreeMap<usize, PageModeSpec>,
}

impl PageModePolicy {
    pub fn exact() -> Self {
        Self::default()
    }

    pub fn with_defaults(default_key: PageModeSpec, default_value: PageModeSpec) -> Result<Self> {
        default_key.validate_for_side(PageSideKind::Key)?;
        default_value.validate_for_side(PageSideKind::Value)?;
        Ok(Self {
            default_key,
            default_value,
            key_overrides: BTreeMap::new(),
            value_overrides: BTreeMap::new(),
        })
    }

    pub fn default_key(&self) -> &PageModeSpec {
        &self.default_key
    }

    pub fn default_value(&self) -> &PageModeSpec {
        &self.default_value
    }

    pub fn key_overrides(&self) -> &BTreeMap<usize, PageModeSpec> {
        &self.key_overrides
    }

    pub fn value_overrides(&self) -> &BTreeMap<usize, PageModeSpec> {
        &self.value_overrides
    }

    pub fn resolve(&self, side: PageSideKind, layer: usize) -> PageModeSpec {
        match side {
            PageSideKind::Key => self
                .key_overrides
                .get(&layer)
                .cloned()
                .unwrap_or_else(|| self.default_key.clone()),
            PageSideKind::Value => self
                .value_overrides
                .get(&layer)
                .cloned()
                .unwrap_or_else(|| self.default_value.clone()),
        }
    }

    pub fn set_default_key(&mut self, mode: PageModeSpec) -> Result<()> {
        mode.validate_for_side(PageSideKind::Key)?;
        self.default_key = mode;
        Ok(())
    }

    pub fn set_default_value(&mut self, mode: PageModeSpec) -> Result<()> {
        mode.validate_for_side(PageSideKind::Value)?;
        self.default_value = mode;
        Ok(())
    }

    pub fn set_override(
        &mut self,
        side: PageSideKind,
        layer: usize,
        mode: PageModeSpec,
    ) -> Result<()> {
        mode.validate_for_side(side)?;
        match side {
            PageSideKind::Key => {
                self.key_overrides.insert(layer, mode);
            }
            PageSideKind::Value => {
                self.value_overrides.insert(layer, mode);
            }
        }
        Ok(())
    }

    pub fn parse_overrides(input: &str) -> Result<Vec<(usize, PageModeSpec)>> {
        let mut parsed = Vec::new();
        for entry in input
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (lhs, rhs) = entry.split_once('=').ok_or(RuntimeError::External {
                context: "page_mode",
                message: format!("invalid page mode override `{entry}`, expected `layer=<mode>`"),
            })?;
            let layer = lhs.parse::<usize>().map_err(|err| RuntimeError::External {
                context: "page_mode",
                message: format!("invalid page mode override layer `{lhs}`: {err}"),
            })?;
            parsed.push((layer, rhs.parse::<PageModeSpec>()?));
        }
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::{PageModePolicy, PageModeSpec, PageModeTag, PageQuantScheme};

    #[test]
    fn parses_exact_and_m0_specs() {
        assert_eq!(
            PageModeSpec::exact(),
            "exact".parse::<PageModeSpec>().unwrap()
        );

        let parsed = "M0/affine/4".parse::<PageModeSpec>().unwrap();
        assert_eq!(parsed.tag(), PageModeTag::M0);
        assert_eq!(parsed.quant_scheme(), PageQuantScheme::Affine);
        assert_eq!(parsed.bits(), 4);
    }

    #[test]
    fn parses_m3_spec_with_escape_dtype() {
        let parsed = "M3/affine/4/float16".parse::<PageModeSpec>().unwrap();
        assert_eq!(parsed.tag(), PageModeTag::M3);
        assert_eq!(parsed.describe(), "M3/affine/4/float16");
    }

    #[test]
    fn parses_layer_override_map() {
        let overrides =
            PageModePolicy::parse_overrides("3=M0/affine/4,7=M3/affine/4/float16").unwrap();
        assert_eq!(overrides.len(), 2);
        assert_eq!(overrides[0].0, 3);
        assert_eq!(overrides[0].1.tag(), PageModeTag::M0);
        assert_eq!(overrides[1].0, 7);
        assert_eq!(overrides[1].1.tag(), PageModeTag::M3);
    }
}
