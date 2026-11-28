//! A look's `pbr` record can carry `anisotropy` (0 to 1, the strength of brushed grooves in the
//! viewport). Designs without the field have anisotropy 0, and a look with anisotropy 0 saves
//! without the field, byte for byte as before. The last release warns about the newer design
//! format, cannot read a `pbr` record with the field, so it leaves that look's response out, and
//! keeps it as it came when saving.
