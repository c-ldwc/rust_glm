use num_traits::Num;
use std::error::Error;

use nalgebra::{DMatrix, DVector, Matrix, SMatrix, SVector, Vector3};
// use newton::step::optim;
use rand::distributions::Distribution;
use rand::rngs::OsRng;
use statrs::distribution::{Binomial, Continuous, Discrete, Normal};
// mod newton;
mod families;
use families::Binomial as binomial_family;
use families::Family;

mod model;
use model::Model;

mod utils;
use utils::{print_matrix, print_vector, inv_logit_vec, inv_logit_fl};

const SAMPLE_SIZE: usize = 50_000;

fn main() -> Result<(), Box<dyn Error>> {
    let n = Normal::new(0.0, 0.2).unwrap();
    let mut xVec = vec![1.0; SAMPLE_SIZE * 5];
    let mut r = OsRng;
    for i in 0..SAMPLE_SIZE * 4 {
        xVec[i] = n.sample(&mut r);
    }

    let X = DMatrix::from_vec(SAMPLE_SIZE, 5, xVec);

    print_matrix(&X);
    let c = vec![2.0, 1.7, 8.8, 10.0, -2.0];
    let true_coef = DVector::from_vec(c);

    let logits = &X * &true_coef;

    let mut p: DVector<f64> = DVector::zeros(SAMPLE_SIZE);
    let mut y = vec![-1.0; SAMPLE_SIZE];
    for _i in 0..SAMPLE_SIZE {
        p[_i] = inv_logit_fl(logits[_i]);
        let b = Binomial::new(p[_i], 1)?;
        y[_i] = b.sample(&mut r);
    }
    // print_vector(&p);

    let y_dvec = DVector::<f64>::from_vec(y);

    let fam = binomial_family::new(
        (1.0,),
        X.clone(),
        y_dvec.clone(),
        DVector::from_vec(vec![0.1,0.1,0.1,0.1,0.1,]),
    );

    let mut model = Model::new(fam, X.clone(), y_dvec.clone());

    model.optim()?;

    print_vector(&model.coef);

    print!("\n\n\nTrue coef\n");

    print_vector(&true_coef);


    println!("Gradient");
    let fam = binomial_family::new(
        (1.0,),
        X.clone(),
        y_dvec.clone(),
        DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
    );

    let nab = fam.grad(&model.coef);

    print_vector(&nab);

    print!("sample size {}", SAMPLE_SIZE);

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
