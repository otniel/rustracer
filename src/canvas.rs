use crate::color::Color;

struct Canvas {
    width: i32,
    height: i32,
    pixels: Vec<Color>,
}
impl Canvas {}

impl Canvas {
    pub fn new(width: i32, height: i32) -> Canvas {
        let resolution = width * height;
        let mut pixels = Vec::with_capacity(resolution as usize);
        for _ in 0..resolution {
            pixels.push(Color::new(0.0, 0.0, 0.0));
        }

        Canvas {
            width,
            height,
            pixels,
        }
    }

    pub fn pixel_at(&self, x: i32, y: i32) -> &Color {
        let index = self.coords_to_pixel_index(x, y);
        &self.pixels[index]
    }

    fn coords_to_pixel_index(&self, x: i32, y: i32) -> usize {
        (y * self.width + x) as usize
    }

    pub fn write_pixel(&mut self, color: Color, x: i32, y: i32) {
        let index = self.coords_to_pixel_index(x, y);
        self.pixels[index] = color;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_creation() {
        let c = Canvas::new(10, 20);

        assert_eq!(c.width, 10);
        assert_eq!(c.height, 20);
        assert_eq!(c.pixel_at(5, 5), &Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_writing_pixels_to_canvas() {
        let mut c = Canvas::new(10, 20);
        let red = Color::new(1.0, 0.0, 0.0);

        c.write_pixel(red, 2, 3);

        assert_eq!(c.pixel_at(2, 3), &Color::new(1.0, 0.0, 0.0));
    }
}
