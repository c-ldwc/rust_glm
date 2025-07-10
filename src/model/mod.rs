use nalgebra::{Const as na_const, DMatrix, DVector, Dyn as na_dyn};
use std::error::Error;
use statrs::distribution::{Normal, ContinuousCDF};
use crate::{families::{Family, Poisson}, utils::print_vector};

// Line search parameters for optimization
pub struct optimiser_args {
    c_1: f64,        // Armijo parameter
    c_2: f64,        // Curvature parameter  
    a: f64,          // Step size
    rho: f64,        // Backtracking factor
    max_iter: usize, // Max iterations
}

pub struct GLM<F: Family> {
    pub family: F,
    pub Data: DMatrix<f64>,
    pub coef: DVector<f64>,  // Parameter estimates
    pub y: DVector<f64>,     // Response vector
    p: usize,                // Number of parameters
    optim_args: optimiser_args,
}

// Inference outputs
struct inference_results {
    covar: DMatrix<f64>,     // Covariance matrix
    p: Vec<f64>,             // P-values
    CI: Vec<(f64,f64)>       // Confidence intervals
}

impl<F: Family> GLM<F> {
    pub fn new(family: F, Data: DMatrix<f64>, y: DVector<f64>) -> Self {
        // Default optimization parameters
        let c_1 = 1e-4;
        let c_2 = 0.9;
        let a = 1.0;
        let rho = 0.99;
        let max_iter = 1000;

        let p = Data.shape().1;
        let coef = DVector::repeat(p, 0.1);  // Start coefficients at 0.1

        let optim_args = optimiser_args {
            c_1,
            c_2,
            a,
            rho,
            max_iter,
        };

        GLM {
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

        //Newton-Raphson with fixed step size
        for _i in 0..self.optim_args.max_iter {
            let lu = &hess.lu();
            let dir = lu.solve(&(-&nab)).ok_or("Hessian is not invertible")?;

            self.coef = &self.coef + dir;  // Take full Newton step

            nab = self.family.grad(&self.coef);

            // Check convergence
            if nab.norm().lt(&1e-5) {
                println!("Converged in {} iterations", _i);
                return Ok(true);
            }
            hess = self.family.hessian(&self.coef);
        }
        Ok(true)
    }

    // Armijo condition - sufficient decrease
    // fn armijo(&self, dir: &DVector<f64>) -> bool {
    //     // let nab = self.family.grad(&self.coef);
    //     // let a_mat = DVector::repeat(self.p, self.optim_args.a);
    //     // let rhs = self.family.log_lik(&self.coef)
    //     //     + self.optim_args.c_1 * self.optim_args.a * nab.dot(dir);
    //     // let step = &self.coef + a_mat.component_mul(dir);
    //     // self.family.log_lik(&step).le(&rhs) //The neg_log_like needs to be evalutated at a certain point, not the current point. This is the sole use case for this function
    //     true
    // }

    // Curvature condition - gradient flattening
    // fn curve(&self, dir: &DVector<f64>) -> bool {
    //     // let a_mat = DMatrix::repeat(self.coef.shape().0, 1, self.optim_args.a);
    //     // let step = &self.coef + a_mat.component_mul(dir);
    //     // let lhs: f64 = self.family.grad(&step).dot(&dir);
    //     // let rhs: f64 = self.optim_args.c_2 * self.family.grad(&self.coef).dot(dir);
    //     // lhs.ge(&rhs)
    //     true
    // }

    // Combined Wolfe conditions
    // fn check_wolfe(&self, dir: &DVector<f64>) -> bool {
    //     let ar = self.armijo(&dir);
    //     let cur = self.curve(&dir);
    //     cur & ar
    // }



    fn inference(&self, alpha:f64) -> Result<inference_results, Box<dyn Error>> {
        // Fisher information matrix (inverse of negative Hessian)
        let covar = self
        .family
        .hessian(&self.coef)
        .scale(-1.0)
        .try_inverse()
        .ok_or("Hessian is not invertible")?;
        
        let q = Normal::standard().inverse_cdf(1.0 - alpha);  // Critical value
        let coef_var = covar.diagonal();  // Standard errors squared

        let mut p = vec![-1.0; self.coef.shape().0];
        let mut CI = vec![(0.0,0.0); self.coef.shape().0];

        // Calculate p-values and confidence intervals
        for i in 0..p.len() {
            let coef = self.coef[i];
            let n = Normal::new(0.0, coef_var[i].sqrt())?;
            p[i] = 2.0*(1.0-n.cdf(coef.abs()));  // Two-tailed test
            CI[i] = (coef - q * coef_var[i].sqrt(), coef + q * coef_var[i].sqrt());
        }

        Ok(inference_results{covar, p, CI})
    }

    pub fn summary(&self, alpha:f64) -> Result<String, Box<dyn Error>> {
        let mut result = String::new();
        let mu = self.family.inv_link(&(&self.Data * &self.coef));

        // Header
        result.push_str("═══════════════════════════════════════\n");
        result.push_str("    Generalized Linear Model Summary\n");
        result.push_str("═══════════════════════════════════════\n");

        // Model info
        result.push_str(&format!("Observations: {:>8}\n", self.Data.nrows()));
        result.push_str(&format!("Parameters:   {:>8}\n", self.p));
        result.push_str(&format!("Scale:        {:>8.4}\n", self.family.scale()));
        result.push_str(&format!(
            "Log-lik:      {:>8.4}\n",
            self.family.log_lik(&mu)
        ));

        // Parameter table
        let inference = self.inference(alpha.clone())?;
        result.push_str("\n┌─────────────┬────────────┬──────────┬─────────────────────────┐\n");
        result.push_str(&format!("│  Parameter  │ Estimate   │ P-value  │     {}% Conf. Int.      │\n", ((1.0-alpha) * 100.0) as usize));
        result.push_str("├─────────────┼────────────┼──────────┼─────────────────────────┤\n");

        for (i, coef) in self.coef.iter().enumerate() {
            let param_name = if i == self.coef.len() - 1 {
                "Intercept".to_string()
            } else {
                format!("β{}", i + 1)
            };
            result.push_str(&format!(
                "│ {:>11} │ {:>10.6} │ {:>8.4} │ ({:>8.4}, {:>8.4})    │\n",
                param_name, coef, inference.p[i], inference.CI[i].0, inference.CI[i].1
            ));
        }

        result.push_str("└─────────────┴────────────┴──────────┴─────────────────────────┘\n");

        Ok(result)
    }
}