use num_traits::Num;
use nalgebra::{DMatrix, DVector};
use std::fmt::Display;
use num_traits::Float;

mod simulation;
pub use simulation::{binom_simulate, poisson_simulate};




pub fn print_matrix<T: Num>(matrix: &DMatrix<T>)
where
    T: Display,
{
    for r in 0..matrix.nrows() {
        for c in 0..matrix.ncols() {
            print!("{:.6} ", matrix[(r, c)]);
        }
        println!();
    }
}

pub fn print_vector(vector: &DVector<f64>) {
    for i in 0..vector.len() {
        println!("{:.6}", vector[i]);
    }
}

pub fn inv_logit_vec(logit:DVector<f64>) -> DVector<f64> {
    logit.map(|l| 1.0/(1.0+(-1.0*l).exp()))
}
pub fn inv_logit_fl<T: Float>(logit: T) -> T {
    T::one() / (T::one() + (-logit).exp())
}