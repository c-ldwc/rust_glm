use std::error::Error;
use std::fmt::Display;
use num_traits::Num;

use nalgebra::{DMatrix, DVector, Matrix, SMatrix, SVector, Vector3};
// use newton::step::optim;
use rand::rngs::OsRng;
use statrs::distribution::{Binomial, Normal, Discrete, Continuous};
use rand::distributions::Distribution;
// mod newton;
mod families;
use families::Binomial as binomial_family;
use families::Family;

mod newton;
use newton::step;

const SAMPLE_SIZE: usize = 10;

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

fn print_matrix<T:Num>(matrix: &DMatrix<T>)
where T:Display, {
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

fn inv_logit(l:f64) -> f64{
    1.0/(1.0+(-l).exp())
}

fn main() -> Result<(), Box<dyn Error>> {
    let n = Normal::new(0.0, 0.2).unwrap();
    let mut xVec = vec![0.0;SAMPLE_SIZE*3];
    let mut r = OsRng;
    for i in 0..SAMPLE_SIZE*3{
        xVec[i] = n.sample(&mut r);
    }

    let X = DMatrix::from_vec(SAMPLE_SIZE,3, xVec);
    let c = vec![2.0, 1.7, 8.8];
    let true_coef = DVector::from_vec(c);

    let logits= &X * &true_coef;

    let mut p: DVector<f64> = DVector::zeros(SAMPLE_SIZE);
    let mut y = vec![-1.0; SAMPLE_SIZE];
    for _i in 0..SAMPLE_SIZE{
        p[_i] = inv_logit(logits[_i]);
        let b = Binomial::new(p[_i], 1)?;
        y[_i] = b.sample(&mut r);
    }
    // print_vector(&p);

    let y_dvec = DVector::<f64>::from_vec(y);


    let fam = binomial_family::new(
         (1.0,),
        X,
        y_dvec,
        DVector::from_vec(vec![0.1, 0.1, 0.1]),
    );

    print_vector(&fam.p);

    print!("\n\n\nTrue p\n");

    print_vector(&p);

    // let mut x_0 = vec![-4201.0, 134534.0];
    // let mut x_0 = DVector::from_vec(x_0);
    // let mut f_0: f64 = obj(&x_0);
    // let mut h_0 = h(&x_0);
    // let mut nab_0 = grad(&x_0);
    // let c_1:f64 = 0.9;
    // let c_2:f64 = 0.1;
    // let rho = 0.99;
    // println!("{}", nab_0);
    // println!("{}", h_0);
    // println!("{}", f_0);
    // for _i in 0..4 {
    //     optim(
    //         &grad,
    //         &h,
    //         &c_1,
    //         &c_2,
    //         &rho,
    //         X,
    //         a,
    //         max_iter);
    //     // print_vector(&x_0);
    //     let f_0: f64 = obj(&x_0);
    //     h_0 = h(&x_0);
    //     nab_0 = grad(&x_0);
    //     println!("x_0:");
    //     print_vector(&x_0);
    //     println!("nab_0:");
    //     print_vector(&nab_0);
    //     // println!("h_0:");
    //     // print_matrix(&h_0);
    //     println!("f_0: {:.20}", f_0);
    // }

    Ok(())
}
