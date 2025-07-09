use nalgebra::{DMatrix, DVector, Matrix, SMatrix, SVector, Vector3};
use rand::distributions::Distribution;
use rand::rngs::OsRng;
use statrs::distribution::{Binomial, Continuous, Discrete, Normal};
use std::error::Error;

use super::inv_logit_fl;

pub struct sim_results {
    pub N: f64,
    pub X: DMatrix<f64>,
    pub y: DVector<f64>,
    pub coef: DVector<f64>,
}

pub fn binom_simulate(
    sample_size: usize,
    coef: DVector<f64>,
    N: f64,
) -> Result<sim_results, Box<dyn Error>> {
    let n = Normal::new(0.0, 0.2).unwrap();
    let mut xVec = vec![1.0; sample_size * 5];
    let mut r = OsRng;
    for i in 0..sample_size * 4 {
        xVec[i] = n.sample(&mut r);
    }

    let X = DMatrix::from_vec(sample_size, 5, xVec);

    let logits = &X * &coef;

    let mut p: DVector<f64> = DVector::zeros(sample_size);
    let mut y = vec![-1.0; sample_size];
    for _i in 0..sample_size {
        p[_i] = inv_logit_fl(logits[_i]);
        let b = Binomial::new(p[_i], N as u64)?;
        y[_i] = b.sample(&mut r);
    }
    // print_vector(&p);

    let y_dvec = DVector::<f64>::from_vec(y);

    Ok(sim_results { X: X, y: y_dvec, N, coef})
}
