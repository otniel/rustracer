extern crate core;

use std::thread::sleep;
use std::time::Duration;
use crate::tuple::Tuple;

mod tuple;



#[derive(Debug)]
struct Projectile {
    position: Tuple,
    velocity: Tuple,
}

impl Projectile {
    fn new(position: Tuple, velocity: Tuple) -> Projectile {
        Projectile { position, velocity }
    }
}

struct Environment {
    gravity: Tuple,
    wind: Tuple,
}

impl Environment {
    fn new(gravity: Tuple, wind: Tuple) -> Environment {
        Environment { gravity, wind }
    }
}

fn tick(environment: &Environment, projectile: &Projectile) -> Projectile {
    Projectile::new(
        projectile.position + projectile.velocity,
        projectile.velocity + environment.gravity + environment.wind
    )
}

fn main() {
    let p = Projectile::new(
        Tuple::point(0.0, 1.0, 0.0),
        Tuple::vector(1.0, 1.0, 0.0).normalize()
    );

    let e = Environment::new(
        Tuple::vector(0.0, -0.1, 0.0),
        Tuple::vector(-0.01, 0.0, 0.0)
    );

    let mut current = p;
    while current.position.y > 0.0 {
        println!("{:?}", current);
        current = tick(&e, &current);
        sleep(Duration::from_secs(1));
    }
}
