extern crate nalgebra as na;
use na::{DMatrix, DVector, Matrix1x4};
use std::error::Error;
use crate::families::Family;

pub trait optimiser<F: Family> {
    fn optim(
        &self,
        grad: fn(x: &DVector<f64>) -> DVector<f64>,
        H: fn(x: &DVector<f64>) -> DMatrix<f64>,
        c_1: f64,
        c_2: f64,
        rho: f64,
        X: DMatrix<f64>,
        a: f64,
        max_iter: usize
    ) -> Result<DVector<f64>, Box<dyn Error>> {
        //Init the parameter vector
        let coef_dim = <F as Family>::Data.shape().1;
        let mut nab:DVector<f64> = grad(&coef);
        let mut hess:DMatrix<f64> = H(&coef);

        //Newton Step
        for _i in 0..max_iter{
            let mut a:f64 = 1.0;
            let dir = DMatrix::repeat(2, 2, -1.0).component_mul(
                &hess.clone()
                    .try_inverse()
                    .ok_or("Hessian is not invertible")?,
            ) * &nab;
            let mut wolfe = false;
            let mut wolfe_checks = 0; 
            while !wolfe && wolfe_checks < 100{
                wolfe = self.check_wolfe(&dir, &a, &c_1, &c_2, grad, &coef);
                a = rho * a;
                let a_mat = DMatrix::repeat(coef_dim, 1, a);
                coef = &coef + a_mat.component_mul(&dir);
                wolfe_checks += 1
            }
            nab = grad(&coef);
            hess = H(&coef);

            //If this while step completes
        }
        Ok(coef)
    }

    fn armijo(&self,dir: &DVector<f64>, a: &f64, c_1: &f64, nab: &DVector<f64>, x: &DVector<f64>) -> bool {
        let a_mat = DMatrix::repeat(2, 1, *a);
        let rhs = obj(&x) + c_1 * a * nab.dot(dir);
        let step = x + a_mat.component_mul(dir);
        obj(&step).le(&rhs)
    }

    fn curve(&self, dir: &DVector<f64>, a: &f64, c_2: &f64, grad: fn(&DVector<f64>) -> DVector<f64>, x: &DVector<f64>) -> bool {
        let a_mat = DMatrix::repeat(2, 1, *a);
        let step = x + a_mat.component_mul(dir);
        let lhs: f64 = grad(&step).dot(&dir);
        let rhs: f64 = c_2 * grad(x).dot(dir);
        lhs.ge(&rhs)
    }

    fn check_wolfe(&self, dir: &DVector<f64>, a:&f64, c_1:&f64, c_2:&f64, grad: fn(&DVector<f64>) -> DVector<f64>, x: &DVector<f64>) -> bool {
        let nab = grad(&x);
        let ar = self.armijo(&dir, &a, c_1, &nab, &x);
        let cur = self.curve(&dir, &a, c_2, grad, &x);
        cur & ar
    }
}
