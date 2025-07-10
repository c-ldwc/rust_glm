use super::family::Family;
use itertools::multizip;
use nalgebra::{DMatrix, DVector};
use rug::{Float, Integer}; // For binomial coefficient

// A member of the exponential family has its pdf structured as
// exp( [y\theta - b(theta)] / a(theta) + c(y, theta))
// Because the partial derivative of the log likelihood wrt theta has expectation 0
// b'(theta) = E[y]

// Struct representing the Binomial family for GLMs
pub struct Binomial {
    Data: DMatrix<f64>, // Design matrix (features)
    y: DVector<f64>,    // Response vector
    coef: DVector<f64>,
    n: f64,              // Number of trials
    pub p: DVector<f64>, // Probability vector
}

// Implementation of the Family trait for Binomial
impl Family for Binomial {
    type Parameters = (f64,);
    type Data = DMatrix<f64>;
    type coef = DVector<f64>;
    type y = DVector<f64>;

    fn get_y(&self) -> &DVector<f64> {
        &self.y
    }
    fn get_data(&self) -> &DMatrix<f64> {
        &self.Data
    }

    // Link function: logit (log odds)
    fn link(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| (p / (self.n - p)).ln())
    }

    // Second derivative of link function wrt p
    fn link_2der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| self.n * (self.n - 2.0 * p) / (p.powi(2) * (self.n - p).powi(2)))
    }

    // Derivative of variance function wrt p
    fn V_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| 1.0 - 2.0 * p / self.n)
    }

    // Variance function: Var[y] = n * p * (1-p)
    fn V(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| p * (1.0 - p / self.n))
    }

    // Scale parameter (constant for binomial)
    fn scale(&self) -> f64 {
        1.0
    }

    // Inverse link function: logistic sigmoid
    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64> {
        l.map(|x| self.n / (1.0 + (-x).exp()))
    }

    // Derivative of link function wrt p
    fn link_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| self.n / (p * (self.n - p)))
    }

    // Log-likelihood for the binomial model
    fn log_lik(&self, l: &DVector<f64>) -> f64 {
        let p = self.inv_link(&l);
        let theta = self.link(&p);
        let b = theta.map(|t| self.n * (1.0 + t.exp()).ln());
        let n_int = Integer::from(self.n as i32);
        // Compute log binomial coefficient for each y
        let c = self
            .y
            .map(|y| n_int.clone().binomial(y as u32).to_f64().ln());

        // Sum log-likelihood contributions for each observation
        multizip((self.y.iter(), theta.iter(), b.iter(), c.iter()))
            .map(|(y, t, b, c)| (y * t - b) + c)
            .sum::<f64>()
    }
}

// Implementation of Binomial-specific methods
impl Binomial {
    // Constructor for Binomial family
    pub fn new(
        params: <Binomial as Family>::Parameters,
        Data: <Binomial as Family>::Data,
        y: <Binomial as Family>::y,
        coef: <Binomial as Family>::coef,
    ) -> Self {
        let l = &Data * &coef;
        let p = l.map(|x| params.0 / (1.0 + (-x).exp()));

        Binomial {
            n: params.0,
            Data,
            y,
            coef,
            p,
        }
    }
}
