use num_traits::Num;
use std::error::Error;

use nalgebra::{DMatrix, DVector};
// use newton::step::optim;


// mod newton;
mod families;
use families::Binomial as binomial_family;
use families::Family;

mod model;
use model::Model;

mod utils;
use utils::{print_matrix, print_vector, inv_logit_vec, inv_logit_fl, binom_simulate};

const SAMPLE_SIZE: usize = 1_000_000;

fn main() -> Result<(), Box<dyn Error>> {

    let true_coef = DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]);

    let simulated = binom_simulate(SAMPLE_SIZE, true_coef, 3.0)?;

    let fam = binomial_family::new(
        (simulated.N,),
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]),//DVector::from_vec(vec![0.1,0.1,0.1,0.1,0.1,]),
    );

    let mut model = Model::new(fam, simulated.X.clone(), simulated.y.clone());

    model.optim()?;

    print_vector(&model.coef);

    print!("\n\n\nTrue coef\n");

    print_vector(&simulated.coef);


    println!("Gradient");
    let fam = binomial_family::new(
        (simulated.N,),
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
    );

    let nab = fam.grad(&model.coef);

    print_vector(&nab);

    print!("sample size {}", SAMPLE_SIZE);
    Ok(())
}
