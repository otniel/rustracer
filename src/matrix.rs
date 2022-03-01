use crate::assert_float_eq;
use std::ops::Mul;

use crate::tuple::Tuple;

fn coords_to_index(width: i32, x: i32, y: i32) -> usize {
    (x * width + y) as usize
}

#[derive(Debug)]
pub struct Matrix {
    width: i32,
    height: i32,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(width: i32, height: i32, data: Vec<f64>) -> Matrix {
        Matrix {
            width,
            height,
            data,
        }
    }

    pub fn zeros(width: i32, height: i32) -> Matrix {
        let data = vec![0.0; (width * height) as usize];
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
            let index = coords_to_index(width, x, y);
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
                let index = coords_to_index(self.width, x, y);
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
        let index = coords_to_index(self.width, x, y);
        self.data[index]
    }

    pub fn submatrix(&self, row_to_delete: i32, col_to_delete: i32) -> Matrix {
        let submatrix_width = self.width - 1;
        let submatrix_height = self.height - 1;
        let submatrix_size = (submatrix_width * submatrix_height) as usize;

        let mut data = Vec::with_capacity(submatrix_size);
        for row in 0..self.width {
            if row != row_to_delete {
                for col in 0..self.height {
                    if col != col_to_delete {
                        data.push(self.get(row, col));
                    }
                }
            }
        }
        Matrix::new(submatrix_width, submatrix_height, data)
    }

    pub fn cofactor(&self, row_to_delete: i32, col_to_delete: i32) -> f64 {
        let minor = self.minor(row_to_delete, col_to_delete);

        // if sum of coord is odd, negate minor
        if (row_to_delete + col_to_delete) & 1 != 0 {
            return -minor;
        }
        minor
    }

    pub fn determinant(&self) -> f64 {
        if self.width == 2 {
            // Get the determinant of a 2x2 matrix
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }
        let fixed_row = 0;
        let determinant: f64 = (0..self.width)
            .map(|col| self.get(fixed_row, col) * self.cofactor(fixed_row, col))
            .sum();
        determinant
    }

    fn minor(&self, row_to_delete: i32, col_to_delete: i32) -> f64 {
        self.submatrix(row_to_delete, col_to_delete).determinant()
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant() != 0.0
    }

    pub fn inverse(&self) -> Matrix {
        if !self.is_invertible() {
            panic!("Not invertible matrix!");
        }

        let mut inverse_data = vec![0.0; (self.width * self.height) as usize];
        let curr_determinant = self.determinant();

        for row in 0..self.width {
            for col in 0..self.height {
                let cofactor = self.cofactor(row, col);
                // Get index for the inverted coords (to get the transposed equivalent!)
                let index = coords_to_index(self.width, col, row);
                inverse_data[index] = cofactor / curr_determinant;
            }
        }
        Matrix::new(self.width, self.height, inverse_data)
    }

    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        let mut translation_matrix = Self::identity(4, 4);
        translation_matrix.data[coords_to_index(4, 0, 3)] = x;
        translation_matrix.data[coords_to_index(4, 1, 3)] = y;
        translation_matrix.data[coords_to_index(4, 2, 3)] = z;
        translation_matrix
    }

    pub fn scaling(x: f64, y: f64, z: f64) -> Self {
        let mut scaling_matrix = Self::identity(4, 4);
        scaling_matrix.data[coords_to_index(4, 0, 0)] = x;
        scaling_matrix.data[coords_to_index(4, 1, 1)] = y;
        scaling_matrix.data[coords_to_index(4, 2, 2)] = z;
        scaling_matrix
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
                let index = coords_to_index(self.width, row, col);
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
        let data: Vec<f64> = (0..self.width)
            .map(|row| -> f64 {
                self.get(row, 0) * tuple.x
                    + self.get(row, 1) * tuple.y
                    + self.get(row, 2) * tuple.z
                    + self.get(row, 3) * tuple.w
            })
            .collect();
        Tuple::new(data[0], data[1], data[2], data[3])
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        for (value, other) in self.data.iter().zip(other.data.iter()) {
            if !assert_float_eq(*value, *other) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::TupleKinds::Point;

    #[test]
    fn test_a_4x4_matrix_can_be_created() {
        // Given the following 4x4 matrix M:
        // |  1.0 |  2.0 |  3.0 |  4.0 |
        // |  5.5 |  6.5 |  7.5 |  8.5 |
        // |  9.0 | 10.0 | 11.0 | 12.0 |
        // | 13.5 | 14.5 | 15.5 | 16.5 |
        let matrix = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.5, 15.5,
                16.5,
            ],
        );

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

        let matrix = Matrix::new(2, 2, vec![-3.0, 5.0, 1.0, -2.0]);

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
        let matrix = Matrix::new(3, 3, vec![-3.0, 5.0, 0.0, 1.0, -2.0, -7.0, 0.0, 1.0, 1.0]);

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

        let matrix_a = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        );
        let matrix_b = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        );

        let matrix_c = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let matrix_d = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let matrix_e = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix_f = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

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

        let matrix_a = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        );

        let matrix_b = Matrix::new(
            4,
            4,
            vec![
                -2.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, -1.0, 4.0, 3.0, 6.0, 5.0, 1.0, 2.0, 7.0, 8.0,
            ],
        );
        let expected_matrix = Matrix::new(
            4,
            4,
            vec![
                20.0, 22.0, 50.0, 48.0, 44.0, 54.0, 114.0, 108.0, 40.0, 58.0, 110.0, 102.0, 16.0,
                26.0, 46.0, 42.0,
            ],
        );
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

        let matrix_a = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 4.0, 2.0, 8.0, 6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
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
        let matrix = Matrix::new(
            4,
            4,
            vec![
                0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 16.0, 4.0, 8.0, 16.0, 32.0,
            ],
        );

        assert_eq!(
            matrix * identity,
            Matrix::new(
                4,
                4,
                vec![
                    0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 16.0, 4.0, 8.0, 16.0,
                    32.0,
                ],
            )
        )
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

        let matrix_a = Matrix::new(
            4,
            4,
            vec![
                0.0, 9.0, 3.0, 0.0, 9.0, 8.0, 0.0, 8.0, 1.0, 8.0, 5.0, 3.0, 0.0, 0.0, 5.0, 8.0,
            ],
        );

        let transposed_matrix = Matrix::new(
            4,
            4,
            vec![
                0.0, 9.0, 1.0, 0.0, 9.0, 8.0, 8.0, 0.0, 3.0, 0.0, 5.0, 5.0, 0.0, 8.0, 3.0, 8.0,
            ],
        );

        assert_eq!(matrix_a.transpose(), transposed_matrix)
    }

    #[test]
    fn test_transposing_the_identity_matrix() {
        let im = Matrix::identity(4, 4);

        assert_eq!(im.transpose(), im)
    }

    #[test]
    fn test_calculate_a_2x2_matrix_determinant() {
        let matrix = Matrix::new(2, 2, vec![1.0, 5.0, -3.0, 2.0]);
        assert_eq!(matrix.determinant(), 17.0)
    }

    #[test]
    fn test_a_sub_matrix_of_3x3_is_2x2() {
        // Given the following 3x3 matrix A:
        // |  1 | 5 |  0 |
        // | -3 | 2 |  7 |
        // |  0 | 6 | -3 |
        // Then submatrix(A, 0, 2) is the following 2x2 matrix:
        // | -3 | 2 |
        // |  0 | 6 |
        let matrix = Matrix::new(3, 3, vec![1.0, 5.0, 0.0, -3.0, 2.0, 7.0, 0.0, 6.0, -3.0]);
        let submatrix = matrix.submatrix(0, 2);

        assert_eq!(submatrix, Matrix::new(2, 2, vec![-3.0, 2.0, 0.0, 6.0]))
    }

    #[test]
    fn test_a_sub_matrix_of_4x4_is_3x3() {
        // Given the following 4x4 matrix A:
        // | -6 | 1 |  1 | 6 |
        // | -8 | 5 |  8 | 6 |
        // | -1 | 0 |  8 | 2 |
        // | -7 | 1 | -1 | 1 |
        // Then submatrix(A, 2, 1) is the following 3x3 matrix:
        // | -6 |  1 | 6 |
        // | -8 |  8 | 6 |
        // | -7 | -1 | 1 |
        let matrix = Matrix::new(
            4,
            4,
            vec![
                -6.0, 1.0, 1.0, 6.0, -8.0, 5.0, 8.0, 6.0, -1.0, 0.0, 8.0, 2.0, -7.0, 1.0, -1.0, 1.0,
            ],
        );
        let submatrix = matrix.submatrix(2, 1);

        assert_eq!(
            submatrix,
            Matrix::new(3, 3, vec![-6.0, 1.0, 6.0, -8.0, 8.0, 6.0, -7.0, -1.0, 1.0])
        )
    }

    #[test]
    fn test_calculate_a_3x3_matrix_minor() {
        // Given the following 3x3 matrix A:
        // | 3 |  5 |  0 |
        // | 2 | -1 | -7 |
        // | 6 | -1 |  5 |
        //   And B ← submatrix(A, 1, 0)
        // Then determinant(B) = 25
        //   And minor(A, 1, 0) = 2
        let matrix = Matrix::new(3, 3, vec![3.0, 5.0, 0.0, 2.0, -1.0, -7.0, 6.0, -1.0, 5.0]);
        let submatrix = matrix.submatrix(1, 0);

        assert_eq!(submatrix.determinant(), 25.0);
        assert_eq!(matrix.minor(1, 0), 25.0);
    }

    #[test]
    fn test_calculate_a_3x3_matrix_cofactor() {
        // Given the following 3x3 matrix A:
        // | 3 |  5 |  0 |
        // | 2 | -1 | -7 |
        // | 6 | -1 |  5 |
        // Then minor(A, 0, 0) = -12
        //   And cofactor(A, 0, 0) = -12
        //   And minor(A, 1, 0) = 25
        //   And cofactor(A, 1, 0) = -25
        let matrix = Matrix::new(3, 3, vec![3.0, 5.0, 0.0, 2.0, -1.0, -7.0, 6.0, -1.0, 5.0]);

        assert_eq!(matrix.minor(0, 0), -12.0);
        assert_eq!(matrix.cofactor(0, 0), -12.0);
        assert_eq!(matrix.minor(1, 0), 25.0);
        assert_eq!(matrix.cofactor(1, 0), -25.0);
    }

    #[test]
    fn calculate_the_determinant_of_a_3x3_matrix() {
        // Given the following 3x3 matrix A:
        // |  1 | 2 |  6 |
        // | -5 | 8 | -4 |
        // |  2 | 6 |  4 |
        // Then cofactor(A, 0, 0) = 56
        //   And cofactor(A, 0, 1) = 12
        //   And cofactor(A, 0, 2) = -46
        //   And determinant(A) = -196

        let matrix = Matrix::new(3, 3, vec![1.0, 2.0, 6.0, -5.0, 8.0, -4.0, 2.0, 6.0, 4.0]);

        assert_eq!(matrix.cofactor(0, 0), 56.0);
        assert_eq!(matrix.cofactor(0, 1), 12.0);
        assert_eq!(matrix.cofactor(0, 2), -46.0);

        assert_eq!(matrix.determinant(), -196.0);
    }

    #[test]
    fn calculate_the_determinant_of_a_4x4_matrix() {
        // Given the following 4x4 matrix A:
        // | -2 | -8 |  3 |  5 |
        // | -3 |  1 |  7 |  3 |
        // |  1 |  2 | -9 |  6 |
        // | -6 |  7 |  7 | -9 |
        // Then cofactor(A, 0, 0) = 690
        //   And cofactor(A, 0, 1) = 447
        //   And cofactor(A, 0, 2) = 210
        //   And cofactor(A, 0, 3) = 51
        //   And determinant(A) = -4071
        let matrix = Matrix::new(
            4,
            4,
            vec![
                -2.0, -8.0, 3.0, 5.0, -3.0, 1.0, 7.0, 3.0, 1.0, 2.0, -9.0, 6.0, -6.0, 7.0, 7.0,
                -9.0,
            ],
        );

        assert_eq!(matrix.cofactor(0, 0), 690.0);
        assert_eq!(matrix.cofactor(0, 1), 447.0);
        assert_eq!(matrix.cofactor(0, 2), 210.0);
        assert_eq!(matrix.cofactor(0, 3), 51.0);

        assert_eq!(matrix.determinant(), -4071.0);
    }

    #[test]
    fn test_an_invertible_matrix_for_invertibility() {
        // Given the following 4x4 matrix A:
        // | 6 |  4 | 4 |  4 |
        // | 5 |  5 | 7 |  6 |
        // | 4 | -9 | 3 | -7 |
        // | 9 |  1 | 7 | -6 |
        // Then determinant(A) = -2120
        //   And A is invertible

        let matrix = Matrix::new(
            4,
            4,
            vec![
                6.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 6.0, 4.0, -9.0, 3.0, -7.0, 9.0, 1.0, 7.0, -6.0,
            ],
        );

        assert_eq!(matrix.determinant(), -2120.0);
        assert_eq!(matrix.is_invertible(), true)
    }

    #[test]
    fn test_an_non_invertible_matrix_for_invertibility() {
        // Given the following 4x4 matrix A:
        // | 6 |  4 | 4 |  4 |
        // | 5 |  5 | 7 |  6 |
        // | 4 | -9 | 3 | -7 |
        // | 9 |  1 | 7 | -6 |
        // Then determinant(A) = -2120
        //   And A is invertible

        let matrix = Matrix::new(
            4,
            4,
            vec![
                -4.0, 2.0, -2.0, 3.0, 9.0, 6.0, 2.0, 6.0, 0.0, -5.0, 1.0, -5.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(matrix.determinant(), 0.0);
        assert_eq!(matrix.is_invertible(), false)
    }

    #[test]
    fn test_calculate_the_inverse_of_the_matrix() {
        //Given the following 4x4 matrix A:
        // | -5 |  2 |  6 | -8|
        // |  1 | -5 |  1 |  8|
        // |  7 |  7 | -6 | -7|
        // |  1 | -3 |  7 |  4|
        //   And B ← inverse(A)
        // Then determinant(A) = 532
        //   And cofactor(A, 2, 3) = -160
        //   And B[3,2] = -160/532
        //   And cofactor(A, 3, 2) = 105
        //   And B[2,3] = 105/532
        //   And B is the following 4x4 matrix:
        // |  0.21805 |  0.45113 |  0.24060 | -0.04511 |
        // | -0.80827 | -1.45677 | -0.44361 |  0.52068 |
        // | -0.07895 | -0.22368 | -0.05263 |  0.19737 |
        // | -0.52256 | -0.81391 | -0.30075 |  0.30639 |
        // |-5| 2| 6|-8|
        // | 1|-5| 1| 8|
        // | 7| 7|-6|-7|
        // | 1|-3| 7| 4|
        let matrix = Matrix::new(
            4,
            4,
            vec![
                -5.0, 2.0, 6.0, -8.0, 1.0, -5.0, 1.0, 8.0, 7.0, 7.0, -6.0, -7.0, 1.0, -3.0, 7.0,
                4.0,
            ],
        );

        let matrix_b = matrix.inverse();

        assert_eq!(matrix.determinant(), 532.0);
        assert_eq!(matrix.cofactor(2, 3), -160.0);
        assert_eq!(matrix_b.get(3, 2), -160.0 / 532.0);
        assert_eq!(matrix.cofactor(3, 2), 105.0);
        assert_eq!(matrix_b.get(2, 3), 105.0 / 532.0);
        assert_eq!(
            matrix_b,
            Matrix::new(
                4,
                4,
                vec![
                    0.21805, 0.45113, 0.24060, -0.04511, -0.80827, -1.45677, -0.44361, 0.52068,
                    -0.07895, -0.22368, -0.05263, 0.19737, -0.52256, -0.81391, -0.30075, 0.30639,
                ],
            )
        )
    }

    #[test]
    fn test_calculate_the_inverse_of_another_matrix() {
        //Given the following 4x4 matrix A:
        // |  8 | -5 |  9 |  2 |
        // |  7 |  5 |  6 |  1 |
        // | -6 |  0 |  9 |  6 |
        // | -3 |  0 | -9 | -4 |
        // Then inverse(A) is the following 4x4 matrix:
        // | -0.15385 | -0.15385 | -0.28205 | -0.53846 |
        // | -0.07692 |  0.12308 |  0.02564 |  0.03077 |
        // |  0.35897 |  0.35897 |  0.43590 |  0.92308 |
        // | -0.69231 | -0.69231 | -0.76923 | -1.92308 |
        let matrix = Matrix::new(
            4,
            4,
            vec![
                8.0, -5.0, 9.0, 2.0, 7.0, 5.0, 6.0, 1.0, -6.0, 0.0, 9.0, 6.0, -3.0, 0.0, -9.0, -4.0,
            ],
        );

        assert_eq!(
            matrix.inverse(),
            Matrix::new(
                4,
                4,
                vec![
                    -0.15385, -0.15385, -0.28205, -0.53846, -0.07692, 0.12308, 0.02564, 0.03077,
                    0.35897, 0.35897, 0.43590, 0.92308, -0.69231, -0.69231, -0.76923, -1.92308,
                ],
            )
        )
    }

    #[test]
    fn test_calculate_the_inverse_of_a_third_matrix() {
        // Given the following 4x4 matrix A:
        // |  9 |  3 |  0 |  9 |
        // | -5 | -2 | -6 | -3 |
        // | -4 |  9 |  6 |  4 |
        // | -7 |  6 |  6 |  2 |
        // Then inverse(A) is the following 4x4 matrix:
        // | -0.04074 | -0.07778 | 0.14444 | -0.22222 |
        // | -0.07778 |  0.03333 | 0.36667 | -0.33333 |
        // | -0.02901 | -0.14630 | -0.10926 | 0.12963 |
        // |  0.17778 |  0.06667 | -0.26667 | 0.33333 |

        let matrix = Matrix::new(
            4,
            4,
            vec![
                9.0, 3.0, 0.0, 9.0, -5.0, -2.0, -6.0, -3.0, -4.0, 9.0, 6.0, 4.0, -7.0, 6.0, 6.0,
                2.0,
            ],
        );

        assert_eq!(
            matrix.inverse(),
            Matrix::new(
                4,
                4,
                vec![
                    -0.04074, -0.07778, 0.14444, -0.22222, -0.07778, 0.03333, 0.36667, -0.33333,
                    -0.02901, -0.14630, -0.10926, 0.12963, 0.17778, 0.06667, -0.26667, 0.33333,
                ],
            )
        )
    }

    #[test]
    fn test_multiply_a_product_by_its_inverse() {
        // Given the following 4x4 matrix A:
        // |  3 | -9 |  7 |  3 |
        // |  3 | -8 |  2 | -9 |
        // | -4 |  4 |  4 |  1 |
        // | -6 |  5 | -1 |  1 |
        // And the following 4x4 matrix B:
        // | 8 |  2 | 2 | 2 |
        // | 3 | -1 | 7 | 0 |
        // | 7 |  0 | 5 | 4 |
        // | 6 | -2 | 0 | 5 |
        // And C ← A * B
        // Then C * inverse(B) = A
        let matrix_a = Matrix::new(
            4,
            4,
            vec![
                3.0, -9.0, 7.0, 3.0, 3.0, -8.0, 2.0, -9.0, -4.0, 4.0, 4.0, 1.0, -6.0, 5.0, -1.0,
                1.0,
            ],
        );
        let matrix_b = Matrix::new(
            4,
            4,
            vec![
                8.0, 2.0, 2.0, 2.0, 3.0, -1.0, 7.0, 0.0, 7.0, 0.0, 5.0, 4.0, 6.0, -2.0, 0.0, 5.0,
            ],
        );
        let inverse_b = matrix_b.inverse();
        let matrix_c = matrix_a * matrix_b;

        assert_eq!(
            matrix_c * inverse_b,
            // Matrix a
            Matrix::new(
                4,
                4,
                vec![
                    3.0, -9.0, 7.0, 3.0, 3.0, -8.0, 2.0, -9.0, -4.0, 4.0, 4.0, 1.0, -6.0, 5.0,
                    -1.0, 1.0,
                ],
            )
        )
    }

    #[test]
    fn test_multiplying_by_a_translation_matrix() {
        // Given transform ← translation(5, -3, 2)
        // And p ← point(-3, 4, 5)
        // Then transform * p = point(2, 1, 7)
        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let p = Tuple::point(-3.0, 4.0, 5.0);

        assert_eq!(transform * p, Tuple::point(2.0, 1.0, 7.0));
    }

    #[test]
    fn test_multiply_by_the_inverse_of_a_translation_matrix() {
        // Given transform ← translation(5, -3, 2)
        //   And inv ← inverse(transform)
        //   And p ← point(-3, 4, 5)
        // Then inv * p = point(-8, 7, 3)
        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let inv = transform.inverse();
        let p = Tuple::point(-3.0, 4.0, 5.0);

        assert_eq!(inv * p, Tuple::point(-8.0, 7.0, 3.0));
    }

    #[test]
    fn test_translation_does_not_affect_vectors() {
        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let v = Tuple::vector(-3.0, 4.0, 5.0);

        assert_eq!(transform * v, Tuple::vector(-3.0, 4.0, 5.0));
    }

    #[test]
    fn test_a_scaling_matrix_applied_to_a_point() {
        // Given transform ← scaling(2, 3, 4)
        // And p ← point(-4, 6, 8)
        // Then transform * p = point(-8, 18, 32)
        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let p = Tuple::point(-4.0, 6.0, 8.0);

        assert_eq!(transform * p, Tuple::point(-8.0, 18.0, 32.0));
    }

    #[test]
    fn test_a_scaling_matrix_applied_to_a_vector() {
        // Given transform ← scaling(2, 3, 4)
        // And v ← vector(-4, 6, 8)
        // Then transform * v = vector(-8, 18, 32)
        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let v = Tuple::vector(-4.0, 6.0, 8.0);

        assert_eq!(transform * v, Tuple::vector(-8.0, 18.0, 32.0));
    }

    #[test]
    fn test_multiplying_by_the_inverse_of_a_scaling_matrix() {
        // Given transform ← scaling(2, 3, 4)
        // And inv ← inverse(transform)
        // And v ← vector(-4, 6, 8)
        // Then inv * v = vector(-2, 2, 2)
        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let inv = transform.inverse();
        let v = Tuple::vector(-4.0, 6.0, 8.0);

        assert_eq!(inv * v, Tuple::vector(-2.0, 2.0, 2.0));
    }

    #[test]
    fn test_reflection_is_scaling_by_a_negative_number() {
        // Given transform ← scaling(-1, 1, 1)
        // And p ← point(2, 3, 4)
        // Then transform * p = point(-2, 3, 4)
        let transform = Matrix::scaling(-1.0, 1.0, 1.0);
        let p = Tuple::point(2.0, 3.0, 4.0);

        assert_eq!(transform * p, Tuple::point(-2.0, 3.0, 4.0))
    }
}
