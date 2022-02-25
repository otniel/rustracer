use crate::color::Color;

const MAX_ROW_LENGTH: usize = 70;

struct Canvas {
    width: i32,
    height: i32,
    pixels: Vec<Color>,
}

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

    pub fn to_ppm(&self) -> String {
        let ppm_header = format!("P3\n{} {}\n255\n", self.width, self.height);

        let mut pixel_data = String::new();

        for colors in self.pixels.chunks(self.width as usize) {
            let canvas_row = self.get_canvas_row(colors);
            pixel_data = format!("{}{}\n", pixel_data, canvas_row);
        }

        format!("{}{}", ppm_header, pixel_data)
    }

    fn get_canvas_row(&self, colors: &[Color]) -> String {
        let mut canvas_row = String::new();
        self.add_colors_to_canvas_row(&mut canvas_row, colors);

        if canvas_row.len() > MAX_ROW_LENGTH {
            self.split_long_canvas_row(&mut canvas_row);
        }
        canvas_row
    }

    fn add_colors_to_canvas_row(&self, canvas_row: &mut String, colors: &[Color]) {
        for color in colors {
            let color_string = color.to_string();
            if canvas_row.is_empty() {
                *canvas_row = color_string;
                continue;
            }
            *canvas_row = format!("{} {}", canvas_row, color_string);
        }
    }

    fn split_long_canvas_row(&self, canvas_row: &mut String) {
        let chars_vector = canvas_row.chars().collect::<Vec<_>>();

        let mut break_line_index = MAX_ROW_LENGTH;
        let mut row_chart = chars_vector[break_line_index];

        while !row_chart.is_whitespace() {
            break_line_index -= 1;
            row_chart = chars_vector[break_line_index];
        }
        canvas_row.replace_range(break_line_index..break_line_index + 1, "\n");
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

    #[test]
    fn test_header_of_canvas_ppm_representation() {
        let c = Canvas::new(5, 3);
        let ppm_string = c.to_ppm();
        let ppm_lines = ppm_string.lines().collect::<Vec<_>>();

        let expected_lines = vec!["P3", "5 3", "255"];
        for line in 1..3 {
            assert_eq!(ppm_lines[line], expected_lines[line])
        }
    }

    #[test]
    fn test_constructing_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);
        let c1 = Color::new(1.5, 0.0, 0.0);
        let c2 = Color::new(0.0, 0.5, 0.0);
        let c3 = Color::new(-0.5, 0.0, 1.0);

        c.write_pixel(c1, 0, 0);
        c.write_pixel(c2, 2, 1);
        c.write_pixel(c3, 4, 2);

        let ppm_string = c.to_ppm();

        let expected_lines = vec![
            "p3",
            "5 3",
            "255",
            "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 128 0 0 0 0 0 0 0",
            "0 0 0 0 0 0 0 0 0 0 0 0 0 0 255",
        ];

        let ppm_lines = ppm_string.lines().collect::<Vec<_>>();
        for line in 4..6 {
            assert_eq!(ppm_lines[line], expected_lines[line])
        }
    }

    #[test]
    fn test_a_line_splits_if_its_too_long() {
        let width = 10;
        let height = 2;

        let mut c = Canvas::new(width, height);

        for y in 0..height {
            for x in 0..width {
                c.write_pixel(Color::new(1.0, 0.8, 0.6), x, y);
            }
        }

        let ppm_string = c.to_ppm();

        let expected_lines = vec![
            "p3",
            "5 3",
            "255",
            "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204",
            "153 255 204 153 255 204 153 255 204 153 255 204 153",
            "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204",
            "153 255 204 153 255 204 153 255 204 153 255 204 153",
        ];

        let ppm_lines = ppm_string.lines().collect::<Vec<_>>();
        for line in 4..7 {
            assert_eq!(ppm_lines[line], expected_lines[line])
        }
    }

    #[test]
    fn test_ppm_files_terminate_by_a_newline_character() {
        let c = Canvas::new(5, 3);
        let mut ppm_string = c.to_ppm();

        let last_char = ppm_string.pop();
        let expected_char = "\n".chars().next();
        assert_eq!(last_char, expected_char);
    }
}
