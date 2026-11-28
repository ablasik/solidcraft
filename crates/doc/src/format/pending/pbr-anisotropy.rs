//! A look's `pbr` record can carry `anisotropy` (0 to 1, the strength of brushed grooves in the
//! viewport) and `anisotropy_angle` (degrees, -180 to 180, the direction of straight grooves about
//! the face normal). Designs without the fields have both at 0, and a look with both at 0 saves
//! without them, byte for byte as before. The last release warns about the newer design format,
//! cannot read a `pbr` record with either field, so it leaves that look's response out, and keeps
//! it as it came when saving.
