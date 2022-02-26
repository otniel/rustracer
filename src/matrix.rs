use std::ops::Mul;

use crate::tuple::Tuple;

#[derive(PartialEq, Debug)]
pub struct Matrix {
    width: i32,
    height: i32,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(width: i32, height: i32) -> Matrix {
        let mut data = vec![0.0; (width * height) as usize];
        Matrix {
            width,
            height,
            data,
        }
    }
    pub fn identity(width: i32, height: i32) -> Matrix {
        let capacity = (width * height) as usize;
        let mut data = vec![0.0; capacity];
        for (x, y) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
            let index = (x * width + y) as usize;
            data[index] = 1.0
        }

        Matrix {
            width,
            height,
            data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; (self.width * self.height) as usize];
        for x in 0..self.width {
            for y in 0..self.height {
                let index = (x * self.width + y) as usize;
                data[index] = self.get(y, x)
            }
        }
        Matrix {
            width: self.width,
            height: self.height,
            data,
        }
    }
    pub fn fill(&mut self, data: Vec<f64>) {
        self.data = data;
    }

    pub fn get(&self, x: i32, y: i32) -> f64 {
        let index = (x * self.width + y) as usize;
        self.data[index]
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        // for row ← 0 to 3
        //   for col ← 0 to 3
        //     M[row, col] ← A[row, 0] * B[0, col]
        //                    + A[row, 1] * B[1, col]
        //                   + A[row, 2] * B[2, col]
        //                   + A[row, 3] * B[3, col]
        //     end for
        //   end for
        let mut data = vec![0.0; (self.width * self.height) as usize];

        for row in 0..self.width {
            for col in 0..self.height {
                let element = self.get(row, 0) * other.get(0, col)
                    + self.get(row, 1) * other.get(1, col)
                    + self.get(row, 2) * other.get(2, col)
                    + self.get(row, 3) * other.get(3, col);
                let index = (row * self.width + col) as usize;
                data[index] = element;
            }
        }

        Matrix {
            width: self.width,
            height: self.height,
            data,
        }
    }
}

impl Mul<Tuple> for Matrix {
    type Output = Tuple;

    fn mul(self, tuple: Tuple) -> Self::Output {
        let mut data: Vec<f64> = Vec::with_capacity(4);
        for row in 0..self.width {
            let element = self.get(row, 0) * tuple.x
                + self.get(row, 1) * tuple.y
                + self.get(row, 2) * tuple.z
                + self.get(row, 3) * tuple.w;
            data.push(element);
        }

        Tuple::new(data[0], data[1], data[2], data[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::id;

    #[test]
    fn test_a_4x4_matrix_can_be_created() {
        // Given the following 4x4 matrix M:
        // |  1.0 |  2.0 |  3.0 |  4.0 |
        // |  5.5 |  6.5 |  7.5 |  8.5 |
        // |  9.0 | 10.0 | 11.0 | 12.0 |
        // | 13.5 | 14.5 | 15.5 | 16.5 |
        let mut matrix = Matrix::new(4, 4);
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.5, 15.5, 16.5,
        ];
        matrix.fill(data);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 3), 4.0);
        assert_eq!(matrix.get(1, 0), 5.5);
        assert_eq!(matrix.get(1, 2), 7.5);
        assert_eq!(matrix.get(2, 2), 11.0);
        assert_eq!(matrix.get(3, 0), 13.5);
        assert_eq!(matrix.get(3, 2), 15.5);
    }

    #[test]
    fn test_a_2x2_matrix_ought_to_be_representable() {
        // Scenario: A 2x2 matrix ought to be representable
        // Given the following 2x2 matrix M:
        // | -3 | 5 |
        // | 1 | -2 |
        // Then M[0,0] = -3
        // And M[0,1] = 5
        // And M[1,0] = 1
        // And M[1,1] = -2

        let mut matrix = Matrix::new(2, 2);
        let data = vec![-3.0, 5.0, 1.0, -2.0];
        matrix.fill(data);

        assert_eq!(matrix.get(0, 0), -3.0);
        assert_eq!(matrix.get(0, 1), 5.0);
        assert_eq!(matrix.get(1, 0), 1.0);
        assert_eq!(matrix.get(1, 1), -2.0);
    }

    #[test]
    fn test_a_3x3_matrix_ought_to_be_representable() {
        // Given the following 3x3 matrix M:
        // | -3 | 5 |  0 |
        // | 1 | -2 | -7 |
        // | 0 |  1 |  1 |
        // Then M[0,0] = -3
        // And M[1,1] = -2
        // And M[2,2] = 1
        let mut matrix = Matrix::new(3, 3);
        let data = vec![-3.0, 5.0, 0.0, 1.0, -2.0, -7.0, 0.0, 1.0, 1.0];
        matrix.fill(data);

        assert_eq!(matrix.get(0, 0), -3.0);
        assert_eq!(matrix.get(1, 1), -2.0);
        assert_eq!(matrix.get(2, 2), 1.0);
    }

    #[test]
    fn test_matrix_equality_with_identical_matrices() {
        // Given the following matrix A:
        // |1|2|3|4|
        // |5|6|7|8|
        // |9|8|7|6|
        // |5|4|3|2|
        // And the following matrix B:
        // |1|2|3|4|
        // |5|6|7|8|
        // |9|8|7|6|
        // |5|4|3|2|
        // Then A = B

        let mut matrix_a = Matrix::new(4, 4);
        matrix_a.fill(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
        ]);
        let mut matrix_b = Matrix::new(4, 4);
        matrix_b.fill(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
        ]);

        let mut matrix_c = Matrix::new(3, 3);
        matrix_c.fill(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mut matrix_d = Matrix::new(3, 3);
        matrix_d.fill(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let mut matrix_e = Matrix::new(2, 2);
        matrix_e.fill(vec![1.0, 2.0, 3.0, 4.0]);

        let mut matrix_f = Matrix::new(2, 2);
        matrix_f.fill(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix_a, matrix_b);
        assert_eq!(matrix_c, matrix_d);
        assert_eq!(matrix_e, matrix_f);
    }

    #[test]
    fn test_matrix_multiplication() {
        //Given the following matrix A:
        // |1|2|3|4|
        // |5|6|7|8|
        // |9|8|7|6|
        // |5|4|3|2|
        // And the following matrix B:
        // |-2|1|2| 3|
        // | 3|2|1|-1|
        // | 4|3|6| 5|
        // | 1|2|7| 8|
        // Then A * B is the following 4x4 matrix:
        // |20| 22| 50| 48|
        // |44| 54|114|108|
        // |40| 58|110|102|
        // |16| 26| 46| 42|

        let mut matrix_a = Matrix::new(4, 4);
        matrix_a.fill(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
        ]);

        let mut matrix_b = Matrix::new(4, 4);
        matrix_b.fill(vec![
            -2.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, -1.0, 4.0, 3.0, 6.0, 5.0, 1.0, 2.0, 7.0, 8.0,
        ]);

        let mut expected_matrix = Matrix::new(4, 4);
        expected_matrix.fill(vec![
            20.0, 22.0, 50.0, 48.0, 44.0, 54.0, 114.0, 108.0, 40.0, 58.0, 110.0, 102.0, 16.0, 26.0,
            46.0, 42.0,
        ]);
        assert_eq!(matrix_a * matrix_b, expected_matrix)
    }

    #[test]
    fn test_a_matrix_can_be_multiplied_by_a_tuple() {
        // Given the following matrix A:
        // |1|2|3|4|
        // |2|4|4|2|
        // |8|6|4|1|
        // |0|0|0|1|
        // And b ← tuple(1, 2, 3, 1)
        // Then A * b = tuple(18, 24, 33, 1)

        let mut matrix_a = Matrix::new(4, 4);
        matrix_a.fill(vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 4.0, 2.0, 8.0, 6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        let tuple = Tuple::new(1.0, 2.0, 3.0, 1.0);

        assert_eq!(matrix_a * tuple, Tuple::new(18.0, 24.0, 33.0, 1.0))
    }

    #[test]
    fn test_multiply_a_matrix_by_the_identity_matrix() {
        // Given the following matrix A:
        // |0|1| 2|   4|
        // |1|2| 4|   8|
        // |2|4| 8|  16|
        // |4|8| 16| 32|
        // Then A * identity_matrix = A
        let identity = Matrix::identity(4, 4);

        let mut matrix_a = Matrix::new(4, 4);
        matrix_a.fill(vec![
            0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 16.0, 4.0, 8.0, 16.0, 32.0,
        ]);

        let mut expected_matrix = Matrix::new(4, 4);
        expected_matrix.fill(vec![
            0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 16.0, 4.0, 8.0, 16.0, 32.0,
        ]);

        assert_eq!(matrix_a * identity, expected_matrix)
    }

    #[test]
    fn test_transposing_a_matrix() {
        // Given the following matrix A:
        // |0|9|3|0|
        // |9|8|0|8|
        // |1|8|5|3|
        // |0|0|5|8|
        // Then transpose(A) is the following matrix:
        // |0|9|1|0|
        // |9|8|8|0|
        // |3|0|5|5|
        // |0|8|3|8|
        // (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)

        let mut matrix_a = Matrix::new(4, 4);
        matrix_a.fill(vec![
            0.0, 9.0, 3.0, 0.0, 9.0, 8.0, 0.0, 8.0, 1.0, 8.0, 5.0, 3.0, 0.0, 0.0, 5.0, 8.0,
        ]);

        let mut transposed_matrix = Matrix::new(4, 4);
        transposed_matrix.fill(vec![
            0.0, 9.0, 1.0, 0.0, 9.0, 8.0, 8.0, 0.0, 3.0, 0.0, 5.0, 5.0, 0.0, 8.0, 3.0, 8.0,
        ]);

        assert_eq!(matrix_a.transpose(), transposed_matrix)
    }

    #[test]
    fn test_transposing_the_identity_matrix() {
        let im = Matrix::identity(4, 4);

        assert_eq!(im.transpose(), im)
    }
}
