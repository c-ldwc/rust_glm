use super::family::Family;
use itertools::multizip;
use nalgebra::{DMatrix, DVector, DVectorSlice};
use rug::{Float, Integer}; // For Poisson coefficient

// A member of the exponential family has its pdf structured as
// exp( [y\theta - b(theta)] / a(theta) + c(y, theta))
// Because the partial derivative of the log likelihood wrt theta has expectation 0
// b'(theta) = E[y]

// Struct representing the Poisson family for GLMs
pub struct Poisson {
    Data: DMatrix<f64>, // Design matrix (features)
    y: DVector<f64>,    // Response vector
    coef: DVector<f64>,      // Starting coefficients
    lam: DVector<f64>
}

// Implementation of the Family trait for Poisson
impl Family for Poisson {
    type Parameters = ();
    type Data = DMatrix<f64>;
    type coef = DVector<f64>;
    type y = DVector<f64>;

    fn get_y(&self) -> &DVector<f64> {
        &self.y
    }
    fn get_data(&self) -> &DMatrix<f64> {
        &self.Data
    }
    fn link(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.map(|m|{m.ln()})
    }
    
    fn link_der(&self, mu:&DVector<f64>) -> DVector<f64> {
        mu.map(|m| 1.0/m)
    }

    fn link_2der(&self, mu:&DVector<f64>) -> DVector<f64> {
        mu.map(|m| -1.0/m.powi(2))
    }

    fn inv_link(&self, l:&DVector<f64>) -> DVector<f64> {
        l.map(|L| L.exp())
    }

    fn V(&self, mu:&DVector<f64>) -> DVector<f64> {
        mu.clone()
    }

    fn V_der(&self, p: &DVector<f64>) -> DVector<f64> {
        DVector::repeat(p.shape().0, 1.0)
    }

    fn scale(&self) -> f64 {
        1.0
    }

    fn log_lik(&self, mu: &DVector<f64>) -> f64 {

        let y = &self.y;
        let theta = mu.map(|m| m.ln());
        let b_theta = mu;
        let log_y_fact = y.iter().map(|&yi| {
            if yi <= 1.0 {
                0.0
            } else {
                (1..=(yi as u64)).map(|v| (v as f64).ln()).sum()
            }
        });

        y.iter().zip(theta.iter()).zip(b_theta.iter()).zip(log_y_fact)
            .map(|(((yi, thetai), bthetai), logyfacti)| {
                yi * thetai - bthetai - logyfacti
            })
            .sum()
    }
}


// Implementation of Poisson-specific methods
impl Poisson {
    // Constructor for Poisson family
    pub fn new(
        Data: <Poisson as Family>::Data,
        y: <Poisson as Family>::y,
        coef: <Poisson as Family>::coef,
    ) -> Self {
        let l = &Data * &coef;
        let lam = l.map(|x| x.exp());

        Poisson {
            Data,
            y,
            coef,
            lam,
        }
    }
}
