use crate::assert_float_eq;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul, Sub};

fn translate_color(float_color: f64) -> i32 {
    // Translate from a value between 0 and 1 to 0-255
    if float_color <= 0.0 {
        return 0;
    }

    if float_color >= 1.0 {
        return 255;
    }
    (float_color * 256.0 / 1.0) as i32
}

#[derive(Debug)]
pub struct Color {
    red: f64,
    green: f64,
    blue: f64,
}

impl Color {
    pub fn new(red: f64, green: f64, blue: f64) -> Color {
        Color { red, green, blue }
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        assert_float_eq(self.red, other.red)
            && assert_float_eq(self.green, other.green)
            && assert_float_eq(self.blue, other.blue)
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {}",
            translate_color(self.red),
            translate_color(self.green),
            translate_color(self.blue)
        )
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            red: self.red - other.red,
            green: self.green - other.green,
            blue: self.blue - other.blue,
        }
    }
}

impl Mul<f64> for Color {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        Self {
            red: self.red * scalar,
            green: self.green * scalar,
            blue: self.blue * scalar,
        }
    }
}

impl Mul for Color {
    type Output = Self;

    fn mul(self, other: Color) -> Self::Output {
        Self {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colors_are_red_green_and_blue_tuples() {
        let c = Color::new(-5.0, 4.0, 7.0);

        assert_eq!(c.red, -5.0);
        assert_eq!(c.green, 4.0);
        assert_eq!(c.blue, 7.0);
    }

    #[test]
    fn test_adding_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);

        assert_eq!(c1 + c2, Color::new(1.6, 0.7, 1.0))
    }

    #[test]
    fn test_subtracting_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);

        assert_eq!(c1 - c2, Color::new(0.2, 0.5, 0.5))
    }

    // Scenario: Multiplying a color by a scalar
    // Given c ← color(0.2, 0.3, 0.4)
    // Then c * 2 = color(0.4, 0.6, 0.8)

    #[test]
    fn test_multiply_color_by_scalar() {
        let c = Color::new(0.2, 0.3, 0.4);

        assert_eq!(c * 2.0, Color::new(0.4, 0.6, 0.8))
    }

    #[test]
    fn test_multiply_color_by_color() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.9, 1.0, 0.1);

        assert_eq!(c1 * c2, Color::new(0.9, 0.2, 0.04))
    }
}
