//! UI-only Render dock request and echo contract.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderDock {
    pub panel: RenderPanel,
    pub side: DockSide,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum RenderPanel {
    Inspector,
    Browser,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DockSide {
    Left,
    Right,
}
