use std::error::Error;


use nalgebra::{DMatrix, DVector};
use newton::step::optim;
use rand::Rng;
use statrs::distribution::{Binomial, Normal, Discrete, Continuous};
use statrs::statistics::Distribution;


fn obj(x: &DVector<f64>) -> f64 {
    let data: Vec<f64> = vec![1.0, 3.0, 3.0, 1.0];
    let A = DMatrix::from_vec(2, 2, data);
    let y = A * x;
    return x.dot(&y);
}

fn h(x: &DVector<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = vec![1.0, 3.0, 3.0, 1.0];
    let A = DMatrix::from_vec(2, 2, data);
    let S = DMatrix::repeat(2, 2, 2.0);
    return S.component_mul(&A);
}

fn grad(x: &DVector<f64>) -> DVector<f64> {
    let data: Vec<f64> = vec![1.0, 3.0, 3.0, 1.0];
    let A = DMatrix::from_vec(2, 2, data);
    let S = DMatrix::repeat(2, 2, 2.0);
    return S.component_mul(&A) * x;
}

fn print_matrix(matrix: &DMatrix<f64>) {
    for r in 0..matrix.nrows() {
        for c in 0..matrix.ncols() {
            print!("{:.6} ", matrix[(r, c)]);
        }
        println!();
    }
}

fn print_vector(vector: &DVector<f64>) {
    for i in 0..vector.len() {
        println!("{:.6}", vector[i]);
    }
}

fn inv_logit(l:f64){
    1/(1+expf64(l))
}

fn main() -> Result<(), Box<dyn Error>> {
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut xVec = Vec![0;30];
    let r = Rng;
    for i in 0..30{
        xVec[i] = n.sample(&mut r);
    }

    let X:DMatrix<f64> = DMatrix::from_vec(10, 3, xVec);
    let coef_vec = vec![2.0, 1.7, 8.8];
    let true_coef = DVector::from_vec(coef);

    let logits: DMatrix<f64> = X * true_coef;
    

    let mut x_0 = vec![-4201.0, 134534.0];
    let mut x_0 = DVector::from_vec(x_0);
    let mut f_0: f64 = obj(&x_0);
    let mut h_0 = h(&x_0);
    let mut nab_0 = grad(&x_0);
    let c_1:f64 = 0.9;
    let c_2:f64 = 0.1;
    let rho = 0.99;
    println!("{}", nab_0);
    println!("{}", h_0);
    println!("{}", f_0);
    for _i in 0..4 {
        optim(
            &grad,
            &h,
            &c_1,
            &c_2,
            &rho,
            X,
            a,
            max_iter);
        // print_vector(&x_0);
        let f_0: f64 = obj(&x_0);
        h_0 = h(&x_0);
        nab_0 = grad(&x_0);
        println!("x_0:");
        print_vector(&x_0);
        println!("nab_0:");
        print_vector(&nab_0);
        // println!("h_0:");
        // print_matrix(&h_0);
        println!("f_0: {:.20}", f_0);
    }

    Ok(())
}
