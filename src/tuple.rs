use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, PartialEq)]
enum TupleKinds {
    Point,
    Vector,
}
#[derive(Debug, PartialEq)]
struct Tuple {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl Tuple {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Tuple {
        Tuple { x, y, z, w }
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
            _ => panic!("Unknown value for w."),
        }
    }

    fn magnitude(&self) -> f64 {
        let radicand = self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2);

        f64::sqrt(radicand)
    }
}

impl Add for Tuple {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Sub for Tuple {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Neg for Tuple {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Mul<f64> for Tuple {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl Div<f64> for Tuple {
    type Output = Self;

    fn div(self, scalar: f64) -> Self::Output {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            w: self.w / scalar,
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

    #[test]
    fn it_tests_tuples_can_be_added() {
        let a1 = Tuple::new(3.0, -2.0, 5.0, 1.0);
        let a2 = Tuple::new(-2.0, 3.0, 1.0, 0.0);

        assert_eq!(a1 + a2, Tuple::new(1.0, 1.0, 6.0, 1.0));
    }
    #[test]
    fn it_tests_tuples_can_be_subtracted() {
        let a1 = Tuple::point(3.0, 2.0, 1.0);
        let a2 = Tuple::point(5.0, 6.0, 7.0);

        let res = a1 - a2;
        assert_eq!(res, Tuple::vector(-2.0, -4.0, -6.0));
        assert_eq!(res.kind(), TupleKinds::Vector);
    }

    #[test]
    fn it_can_subtract_a_vector_from_a_point() {
        let p = Tuple::point(3.0, 2.0, 1.0);
        let v = Tuple::vector(5.0, 6.0, 7.0);

        let res = p - v;
        assert_eq!(res, Tuple::point(-2.0, -4.0, -6.0));
        assert_eq!(res.kind(), TupleKinds::Point);
    }

    #[test]
    fn it_can_subtract_two_vectors() {
        let v1 = Tuple::vector(3.0, 2.0, 1.0);
        let v2 = Tuple::vector(5.0, 6.0, 7.0);

        let res = v1 - v2;
        assert_eq!(res, Tuple::vector(-2.0, -4.0, -6.0));
        assert_eq!(res.kind(), TupleKinds::Vector);
    }
    #[test]
    fn it_can_subtract_a_vector_from_the_zero_vector() {
        let zero = Tuple::vector(0.0, 0.0, 0.0);
        let v = Tuple::vector(1.0, -2.0, 3.0);

        let res = zero - v;
        assert_eq!(res, Tuple::vector(-1.0, 2.0, -3.0));
        assert_eq!(res.kind(), TupleKinds::Vector);
    }
    #[test]
    fn it_can_negate_a_tuple() {
        let t = Tuple::new(1.0, -2.0, 3.0, -4.0);

        assert_eq!(-t, Tuple::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn it_can_multiply_a_vector_with_a_scalar() {
        let v = Tuple::new(1.0, -2.0, 3.0, -4.0);

        let res = v * 3.5;

        assert_eq!(res, Tuple::new(3.5, -7.0, 10.5, -14.0))
    }

    #[test]
    fn it_can_multiply_a_tuple_with_a_fraction() {
        let v = Tuple::new(1.0, -2.0, 3.0, -4.0);

        let res = v * 0.5;

        assert_eq!(res, Tuple::new(0.5, -1.0, 1.5, -2.0))
    }

    #[test]
    fn it_can_divide_a_tuple_with_a_scalar() {
        let v = Tuple::new(1.0, -2.0, 3.0, -4.0);

        let res = v / 2.0;

        assert_eq!(res, Tuple::new(0.5, -1.0, 1.5, -2.0))
    }
    #[test]
    fn it_can_compute_vector_magnitude_1() {
        let v = Tuple::vector(1.0, 0.0, 0.0);

        assert_eq!(v.magnitude(), 1.0)
    }

    #[test]
    fn it_can_compute_vector_magnitude_2() {
        let v = Tuple::vector(0.0, 1.0, 0.0);

        assert_eq!(v.magnitude(), 1.0)
    }

    #[test]
    fn it_can_compute_vector_magnitude_3() {
        let v = Tuple::vector(0.0, 0.0, 1.0);

        assert_eq!(v.magnitude(), 1.0)
    }

    #[test]
    fn it_can_compute_vector_magnitude_4() {
        let v = Tuple::vector(1.0, 2.0, 3.0);

        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }

    #[test]
    fn it_can_compute_vector_magnitude_5() {
        let v = Tuple::vector(-1.0, -2.0, -3.0);

        assert_eq!(v.magnitude(), f64::sqrt(14.0));
    }
}
