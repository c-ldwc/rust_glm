use nalgebra::{Const as na_const, DMatrix, DVector, Dyn as na_dyn};
use std::error::Error;

use crate::families::Family;

pub struct optimiser_args {
    c_1: f64,
    c_2: f64,
    a: f64,
    rho: f64,
    max_iter: usize,
}
pub struct Model<F: Family> {
    pub family: F,
    pub Data: DMatrix<f64>,
    pub coef: DVector<f64>,
    pub y: DVector<f64>,
    p: usize, //num parameters
    //for backtracking along descent direction
    optim_args: optimiser_args,
}

impl<F: Family> Model<F> {
    pub fn new(family: F, Data: DMatrix<f64>, y: DVector<f64>) -> Self {
        let c_1 = 1e-4;
        let c_2 = 0.9;
        let a = 1.0;
        let rho = 0.99;
        let max_iter = 1000;

        let p = Data.shape().1;
        let coef = DVector::repeat(p, 0.1);

        let optim_args = optimiser_args {
            c_1,
            c_2,
            a,
            rho,
            max_iter,
        };

        Model {
            family: family,
            Data: Data,
            coef: coef,
            y: y,
            p,
            optim_args,
        }
    }

    pub fn optim(&mut self) -> Result<(bool), Box<dyn Error>> {
        //Init the parameter vector
        let coef_dim = self.Data.shape().1;
        let mut coef_proposal = self.coef.clone();
        let mut nab: DVector<f64> = self.family.grad(&coef_proposal);
        let mut hess: DMatrix<f64> = self.family.hessian(&coef_proposal);
        let mut a = self.optim_args.a.clone();

        //Newton Step
        for _i in 0..self.optim_args.max_iter {
            // println!("{}",&_i);
            // let dir = &hess
            //     .try_inverse()
            //     .ok_or("Hessian is not invertible")
            //     .unwrap()
            //     .scale(-1.0)
            //     * &nab;

            let lu = &hess.lu();
            let dir = lu.solve(&(-&nab))
                .ok_or("Hessian is not invertible")?;
            // let dir: nalgebra::Matrix<f64, na_dyn, na_const<1>, nalgebra::VecStorage<f64, na_dyn, na_const<1>>> =
            //     DMatrix::repeat(self.Data.shape().0, self.Data.shape().1, -1.0).component_mul(
            //         &hess
            //             .clone()
            //             .try_inverse()
            //             .ok_or("Hessian is not invertible")?,
            //     ) * &nab;
            let mut wolfe = false;
            let mut wolfe_checks = 0;
            while !wolfe && wolfe_checks < 100 {
                a = self.optim_args.rho * a;
                let a_mat = DMatrix::repeat(coef_dim, 1, a).component_mul(&dir);
                coef_proposal = coef_proposal + a_mat;
                wolfe = self.check_wolfe(&dir);
                wolfe_checks += 1
            }

            if wolfe {
                self.coef = coef_proposal.clone();
            } else {
                return Err(
                    "Descent direction has no points that meet the Wolfe conditions".into(),
                );
            }

            nab = self.family.grad(&self.coef);

            if nab.norm().lt(&1e-5){
                return Ok(true)
            }
            hess = self.family.hessian(&self.coef);

            //If this while step completes
        }
        Ok(true)
    }

    fn armijo(&self, dir: &DVector<f64>) -> bool {
        // let nab = self.family.grad(&self.coef);
        // let a_mat = DVector::repeat(self.p, self.optim_args.a);
        // let rhs = self.family.log_lik(&self.coef)
        //     + self.optim_args.c_1 * self.optim_args.a * nab.dot(dir);
        // let step = &self.coef + a_mat.component_mul(dir);
        // self.family.log_lik(&step).le(&rhs) //The neg_log_like needs to be evalutated at a certain point, not the current point. This is the sole use case for this function
        true
    }

    fn curve(&self, dir: &DVector<f64>) -> bool {
        // let a_mat = DMatrix::repeat(self.coef.shape().0, 1, self.optim_args.a);
        // let step = &self.coef + a_mat.component_mul(dir);
        // let lhs: f64 = self.family.grad(&step).dot(&dir);
        // let rhs: f64 = self.optim_args.c_2 * self.family.grad(&self.coef).dot(dir);
        // lhs.ge(&rhs)
        true
    }

    fn check_wolfe(&self, dir: &DVector<f64>) -> bool {
        let ar = self.armijo(&dir);
        let cur = self.curve(&dir);
        cur & ar
    }
}
