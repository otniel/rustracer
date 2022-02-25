use std::thread::sleep;
use std::time::Duration;

use rustracer::projectiles::{tick, Environment, Projectile};
use rustracer::tuple::Tuple;

fn main() {
    let p = Projectile::new(
        Tuple::point(0.0, 1.0, 0.0),
        Tuple::vector(1.0, 1.0, 0.0).normalize(),
    );

    let e = Environment::new(
        Tuple::vector(0.0, -0.1, 0.0),
        Tuple::vector(-0.01, 0.0, 0.0),
    );

    let mut current = p;
    while current.position.y > 0.0 {
        println!("{:?}", current);
        current = tick(&e, &current);
        sleep(Duration::from_millis(250));
    }
}
