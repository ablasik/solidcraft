//! Persistable opaque optical parameters, independent of engineering material data.

use serde::{Deserialize, Serialize};
use solidcore_record::Vec3;
use solidcraft_viewport::surface::{OpaqueSurface, SurfaceError, srgb8_to_linear};

/// Explicitly selected opaque response. Existing scalar looks have no such record.
/// Construction and deserialization share the renderer's finite admission bounds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Parameters", into = "Parameters")]
pub struct PbrParameters {
    metallic: f64,
    roughness: f64,
    dielectric_ior: f64,
    anisotropy: f64,
    anisotropy_angle: f64,
}

impl Default for PbrParameters {
    fn default() -> Self {
        Self { metallic: 0.0, roughness: 0.5, dielectric_ior: 1.5, anisotropy: 0.0, anisotropy_angle: 0.0 }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    metallic: f64,
    roughness: f64,
    dielectric_ior: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    anisotropy: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    anisotropy_angle: f64,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

impl PbrParameters {
    pub fn new(metallic: f64, roughness: f64, dielectric_ior: f64) -> Result<Self, SurfaceError> {
        OpaqueSurface::new(Vec3::ZERO, metallic, roughness, dielectric_ior)?;
        Ok(Self { metallic, roughness, dielectric_ior, anisotropy: 0.0, anisotropy_angle: 0.0 })
    }

    /// Strength of brushed grooves, 0 (none) to 1; the viewport picks their direction.
    pub fn with_anisotropy(mut self, anisotropy: f64) -> Result<Self, SurfaceError> {
        if !(0.0..=1.0).contains(&anisotropy) {
            return Err(SurfaceError::Anisotropy);
        }
        self.anisotropy = anisotropy;
        Ok(self)
    }

    /// The direction of straight brushed grooves, in degrees (-180 to 180) about the face normal from
    /// the direction the viewport picks; turned faces keep their rings.
    pub fn with_anisotropy_angle(mut self, degrees: f64) -> Result<Self, SurfaceError> {
        if !(-180.0..=180.0).contains(&degrees) {
            return Err(SurfaceError::AnisotropyAngle);
        }
        self.anisotropy_angle = degrees;
        Ok(self)
    }

    /// Decode the containing look's sRGB8 base color once, for the actual consumer.
    pub fn surface(self, color: [u8; 3]) -> Result<OpaqueSurface, SurfaceError> {
        let [r, g, b] = color.map(srgb8_to_linear);
        OpaqueSurface::new(Vec3::new(r, g, b), self.metallic, self.roughness, self.dielectric_ior)
    }

    pub fn metallic(self) -> f64 {
        self.metallic
    }
    pub fn roughness(self) -> f64 {
        self.roughness
    }
    pub fn dielectric_ior(self) -> f64 {
        self.dielectric_ior
    }
    pub fn anisotropy(self) -> f64 {
        self.anisotropy
    }
    pub fn anisotropy_angle(self) -> f64 {
        self.anisotropy_angle
    }
}

impl TryFrom<Parameters> for PbrParameters {
    type Error = SurfaceError;
    fn try_from(p: Parameters) -> Result<Self, Self::Error> {
        Self::new(p.metallic, p.roughness, p.dielectric_ior)?.with_anisotropy(p.anisotropy)?.with_anisotropy_angle(p.anisotropy_angle)
    }
}
impl From<PbrParameters> for Parameters {
    fn from(p: PbrParameters) -> Self {
        Self {
            metallic: p.metallic,
            roughness: p.roughness,
            dielectric_ior: p.dielectric_ior,
            anisotropy: p.anisotropy,
            anisotropy_angle: p.anisotropy_angle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn optical_parameters_share_strict_admission_and_round_trip() {
        let p = PbrParameters::new(1.0, 0.3, 1.5).unwrap();
        let saved = serde_json::to_value(p).unwrap();
        assert_eq!(serde_json::from_value::<PbrParameters>(saved.clone()).unwrap(), p);
        for (key, value) in [
            ("metallic", json!(1.1)),
            ("roughness", json!(0)),
            ("dielectric_ior", json!(0.9)),
            ("extra", json!(true)),
            ("anisotropy", json!(1.5)),
            ("anisotropy", json!(-0.1)),
            ("anisotropy_angle", json!(180.5)),
            ("anisotropy_angle", json!("45")),
        ] {
            let mut bad = saved.clone();
            bad[key] = value;
            assert!(serde_json::from_value::<PbrParameters>(bad).is_err());
        }
        assert!(saved.get("anisotropy").is_none() && saved.get("anisotropy_angle").is_none());
        let brushed = p.with_anisotropy(0.6).unwrap().with_anisotropy_angle(-45.0).unwrap();
        let saved = serde_json::to_value(brushed).unwrap();
        assert_eq!((&saved["anisotropy"], &saved["anisotropy_angle"]), (&json!(0.6), &json!(-45.0)));
        assert_eq!(serde_json::from_value::<PbrParameters>(saved).unwrap(), brushed);
        assert!(p.with_anisotropy(f64::NAN).is_err());
        assert!(p.with_anisotropy(1.01).is_err());
        assert!(p.with_anisotropy_angle(f64::NAN).is_err() && p.with_anisotropy_angle(-181.0).is_err());
        assert!(PbrParameters::new(f64::NAN, 0.3, 1.5).is_err());
        assert!(PbrParameters::new(0.0, f64::INFINITY, 1.5).is_err());
        let surface = p.surface([128, 0, 255]).unwrap();
        assert!((surface.base_color().x - 0.21586050011389926).abs() < 1e-14);
        assert_eq!(surface.base_color().z, 1.0);
    }
}
