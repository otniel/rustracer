#[derive(Debug, PartialEq)]
enum TupleKinds {
    Point,
    Vector
}

struct Tuple { x: f64, y: f64, z: f64, w: f64 }

impl Tuple {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Tuple {
        Tuple {x, y, z, w}
    }

    fn point(x: f64, y: f64, z: f64) -> Tuple {
        Self::new(x, y, z, 1.0)
    }

    fn vector(x: f64, y: f64, z: f64) -> Tuple {
        Self::new(x, y, z, 0.0)
    }

    fn kind(&self) -> TupleKinds {
        match self.w {
            1.0 => TupleKinds::Point,
            0.0 => TupleKinds::Vector,
            _ => panic!("Unknown value for w.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn it_creates_a_point_tuple() {
        let t = Tuple::new(4.3, -4.2, 3.1, 1.0);

        assert_eq!(t.x, 4.3);
        assert_eq!(t.y, -4.2);
        assert_eq!(t.z, 3.1);
        assert_eq!(t.w, 1.0);

        assert_eq!(t.kind(), TupleKinds::Point);
        assert_ne!(t.kind(), TupleKinds::Vector);
    }

    #[test]
    fn it_creates_a_vector_tuple() {
        let t = Tuple::new(4.3, -4.2, 3.1, 0.0);

        assert_eq!(t.x, 4.3);
        assert_eq!(t.y, -4.2);
        assert_eq!(t.z, 3.1);
        assert_eq!(t.w, 0.0);

        assert_ne!(t.kind(), TupleKinds::Point);
        assert_eq!(t.kind(), TupleKinds::Vector);
    }

    #[test]
    fn it_creates_a_point_from_tuple_method() {
        let point = Tuple::point(4.3, -4.2, 3.1);
        assert_eq!(point.kind(), TupleKinds::Point);
    }

    #[test]
    fn it_creates_a_vector_from_tuple_method() {
        let point = Tuple::vector(4.3, -4.2, 3.1);
        assert_eq!(point.kind(), TupleKinds::Vector);
    }
}