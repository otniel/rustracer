extern crate core;

pub mod canvas;
pub mod color;
pub mod matrix;
pub mod projectiles;
pub mod tuple;

const EPSILON: f64 = 0.0001;

pub fn assert_float_eq(a: f64, b: f64) -> bool {
    f64::abs(a - b) < EPSILON
}
