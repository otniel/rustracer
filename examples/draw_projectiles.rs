use std::fs;

use rustracer::canvas::Canvas;
use rustracer::color::Color;
use rustracer::projectiles::{tick, Environment, Projectile};
use rustracer::tuple::Tuple;

fn main() {
    let start = Tuple::point(0.0, 1.0, 0.0);
    let velocity = Tuple::vector(1.0, 1.8, 0.0).normalize() * 11.25;
    let p = Projectile::new(start, velocity);

    let gravity = Tuple::vector(0.0, -0.1, 0.0);
    let wind = Tuple::vector(-0.01, 0.0, 0.0);

    let e = Environment::new(gravity, wind);
    let mut c = Canvas::new(900, 550);

    let mut current = p;

    println!("Drawing a projectile trajectory into an image...");
    while current.position.y > 0.0 {
        let x = current.position.x.round() as i32;
        let y = c.height - current.position.y.round() as i32;

        if x < c.width && x > 0 && y < c.height && y > 0 {
            c.write_pixel(Color::new(1.0, 1.0, 1.0), x, y);
        }

        current = tick(&e, &current);
    }

    let ppm = c.to_ppm();
    fs::write("projectile.ppm", ppm).expect("Unable to write file");
    println!("Image ready! Open projectile.ppm in your current dir.");
}
