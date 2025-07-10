use num_traits::Num;
use std::error::Error;

use nalgebra::{DMatrix, DVector};
// use newton::step::optim;


// mod newton;
mod families;
use families::{Binomial, Poisson};
// use families::poisson;
use families::Family;

mod model;
use model::GLM;

mod utils;
use utils::{print_matrix, print_vector, inv_logit_vec, inv_logit_fl, binom_simulate, poisson_simulate};

const SAMPLE_SIZE: usize = 100_000;

fn main() -> Result<(), Box<dyn Error>> {

    let true_coef = DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]);

    let simulated = binom_simulate(SAMPLE_SIZE, true_coef.clone(), 3.0)?;
    let N = simulated.param[0];
    let fam = Binomial::new(
        (N,),
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]),//DVector::from_vec(vec![0.1,0.1,0.1,0.1,0.1,]),
    );

    let mut model = GLM::new(fam, simulated.X.clone(), simulated.y.clone());

    model.optim()?;

    print_vector(&model.coef);

    print!("\n\n\nTrue coef\n");

    print_vector(&simulated.coef);


    println!("Gradient");
    let fam = Binomial::new(
        (N,),
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
    );

    let nab = fam.grad(&model.coef);

    print_vector(&nab);


    print!("{}", model.summary(0.05)?);


    let simulated = poisson_simulate(SAMPLE_SIZE, true_coef.clone())?;


    let fam = Poisson::new(
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]),//DVector::from_vec(vec![0.1,0.1,0.1,0.1,0.1,]),
    );

    let mut model = GLM::new(fam, simulated.X.clone(), simulated.y.clone());

    model.optim()?;

    print_vector(&model.coef);

    println!("\n\n\nTrue coef\n");

    print_vector(&simulated.coef);


    println!("Gradient");
    let fam = Poisson::new(
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
    );

    let nab = fam.grad(&model.coef);

    print_vector(&nab);

    print!("{}",model.summary(0.1)?);
    Ok(())
}
