use nalgebra::{DMatrix, DVector, Matrix, SMatrix, SVector, Vector3};
use rand::distributions::Distribution;
use rand::rngs::OsRng;
use statrs::distribution::{Binomial, Poisson, Continuous, Discrete, Normal};
use std::error::Error;
use num_traits::Num;

use super::inv_logit_fl;

pub struct sim_results<T:Num> {
    pub param: Vec<T>, //
    pub X: DMatrix<f64>,
    pub y: DVector<f64>,
    pub coef: DVector<f64>,
}

fn create_X(N:&usize, p:&usize, r:&mut OsRng) -> DMatrix<f64> {
    let n = Normal::new(0.0, 0.2).unwrap();
    let mut xVec = vec![1.0; N * p];
    for i in 0..N * (p-1) { //final column is intercept
        xVec[i] = n.sample(r);
    }

    DMatrix::from_vec(*N, *p, xVec)
}

pub fn binom_simulate(
    sample_size: usize,
    coef: DVector<f64>,
    N: f64, //The binomial N parameter
) -> Result<sim_results<f64>, Box<dyn Error>> {
    let num_param: usize = coef.shape().0;
    
    let param = vec![N];
    let mut r = OsRng;
    
    let X = create_X(&sample_size, &num_param, &mut r);
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

    Ok(sim_results {X, y: y_dvec, param, coef})
}


pub fn poisson_simulate(
    sample_size: usize,
    coef: DVector<f64>
) -> Result<sim_results<f64>, Box<dyn Error>> {
    let num_param: usize = coef.shape().0;
    let mut r = OsRng;
    let X = create_X(&sample_size, &num_param, &mut r);
    let link = &X * &coef;

    let mut lam: DVector<f64> = DVector::zeros(sample_size);
    let mut y: Vec<f64> = vec![-1.0; sample_size];
    for _i in 0..sample_size{
        lam[_i] = link[_i].exp();
        let dist = Poisson::new(lam[_i])?;
        y[_i] = dist.sample(&mut r)
    }
    let y_dvec = DVector::<f64>::from_vec(y);

    Ok(sim_results {X, y: y_dvec, param:vec![], coef})

}
