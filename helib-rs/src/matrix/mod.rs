pub(crate) mod bsgs;

use ark_ff::PrimeField;

pub trait SquareMatrix<F: PrimeField> {
    fn dimension(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> F;
}

impl<F: PrimeField> SquareMatrix<F> for Vec<Vec<F>> {
    fn dimension(&self) -> usize {
        self.len()
    }

    fn get(&self, row: usize, col: usize) -> F {
        self[row][col]
    }
}
