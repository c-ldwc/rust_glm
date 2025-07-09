use super::family::Family;
use itertools::{Zip, multizip};
use nalgebra::{DMatrix, DVector};
use rug::{Float, Integer}; // For binomial coefficient

// A member of the exponential family has its pdf structured as
// exp( [y\theta - b(theta)] / a(theta) + c(y, theta))
// Because the partial derivative of the log likelihood wrt theta has expectation 0
// b'(theta) = E[y]

// Struct representing the Binomial family for GLMs
pub struct Binomial {
    n: f64,                // Number of trials
    Data: DMatrix<f64>,    // Design matrix (features)
    y: DVector<f64>,       // Response vector
    coef: DVector<f64>,    // Coefficient vector
    pub p: DVector<f64>,   // Probability vector
}

// Implementation of the Family trait for Binomial
impl Family for Binomial {
    type Parameters = (f64,);
    type Data = DMatrix<f64>;
    type coef = DVector<f64>;
    type y = DVector<f64>;

    // Link function: logit (log odds)
    fn link(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| (p / (1.0 - p)).ln())
    }

    // Variance function: Var[y] = n * p * (1-p)
    fn V(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| self.n * p * (1.0 - p))
    }

    // Alpha as defined by Wood pp.106
    fn alpha(&self, x: &DVector<f64>) -> DVector<f64> {
        let l: DVector<f64> = &self.Data * x;
        let p: DVector<f64> = self.inv_link(&l);

        // Compute alpha for each observation using multizip for elementwise operations
        let mapped = multizip((
            self.y.iter(),
            p.iter(),
            self.V(&p).iter(),
            self.V_der(&p).iter(),
            self.link_der(&p).iter(),
            self.link_2der(&p).iter(),
        ))
        .map(|(y, p, v, v_der, link_der, link_2der)| {
            1.0 + (y - p) * (v_der / v + link_2der / link_der)
        })
        .collect::<Vec<f64>>();
        DVector::<f64>::from_vec(mapped)
    }

    // Weights as defined by Wood pp.106
    fn w(&self, x: &DVector<f64>) -> DVector<f64> {
        let l = &self.Data * x;
        let p = self.inv_link(&l);

        // Compute weights for each observation
        let mapped = multizip((
            self.alpha(x).iter(),
            self.link_der(&p).iter(),
            self.V(&p).iter(),
        ))
        .map(|(a, l, V)| a / (l.powi(2) * V))
        .collect::<Vec<f64>>();
        DVector::<f64>::from_vec(mapped)
    }

    // Scale parameter (constant for binomial)
    fn scale(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::repeat(self.n as usize, 1.0)
    }

    // Hessian matrix computation for Newton-Raphson or IRLS
    fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let X_shape = self.Data.shape();

        // Compute -X^T * W * X efficiently
        let mut neg_X_t = self.Data
        .clone()
        .scale(-1.0)
        .transpose();
        let w = self.w(&x);

        // Scale columns of -X^T by weights
        for i in 0..neg_X_t.shape().1 {
            neg_X_t.column_mut(i).scale_mut(w[i]);
        }
        neg_X_t * &self.Data
    }

    // Gradient of the log-likelihood
    fn grad(&self, x: &DVector<f64>) -> DVector<f64> {
        let N = self.Data.shape().0;
        let mut G_vec = vec![0.0; N];

        let logits = &self.Data * x;

        let p = self.inv_link(&logits);

        let a = self.alpha(&x);
        let a_iter = a.iter();

        // Compute G_vec = link_der / alpha for each observation
        let mut G_vec_storage = multizip((a_iter, self.link_der(&p).iter()))
            .map(|(a, ld)| ld / a)
            .collect::<Vec<f64>>();

        let G_vec = DVector::<f64>::from_vec(G_vec_storage);
        
        let w = self.w(&x);

        // Compute elementwise product of weights and G_vec
        let mut WG_vec = vec![1.0; w.shape().0];

        for i in 0..w.shape().1{
            WG_vec[i] = w[i] * G_vec[i];
        }

        // Multiply each row of X by corresponding WG_vec entry
        let mut X = self.Data.clone();

        // Save a bit of time here by avoiding two matmuls with diagonal matrices
        // Multiply the rows of X (the operation requires the transpose) instead
        for i in 0..X.shape().0{
            X.row_mut(i).scale_mut(WG_vec[i]);
        }

        // Return X^T * (y - p)
        X.transpose() * (&self.y - p)
    }

    // Log-likelihood for the binomial model
    fn log_lik(&self, x: &DVector<f64>) -> f64 {
        let l = &self.Data * x;
        let p = self.inv_link(&l);
        let w = self.w(&x);
        let theta = self.link(&p);
        let b = theta.map(|t| self.n * (1.0 + t.exp()).ln());
        let n_int = Integer::from(self.n as i32);
        // Compute log binomial coefficient for each y
        let c = self
            .y
            .map(|y| n_int.clone().binomial(y as u32).to_f64().ln());

        // Sum log-likelihood contributions for each observation
        multizip((w.iter(), self.y.iter(), theta.iter(), b.iter(), c.iter()))
            .map(|(w, y, t, b, c)| w * (y * t - b) + c)
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
        let p = l.map(|x| 1.0 / (1.0 + (-x).exp()));

        Binomial {
            n: params.0,
            Data,
            y,
            coef,
            p,
        }
    }

    // Inverse link function: logistic sigmoid
    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64> {
        l.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    // Derivative of link function wrt p
    fn link_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| 1.0 / (p * (1.0 - p)))
    }

    // Derivative of link function using self.p
    fn link_der_self(&self) -> DVector<f64> {
        self.link_der(&self.p)
    }

    // Second derivative of link function wrt p
    fn link_2der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| (2.0 * p - 1.0) / (p * (1.0 - p)).powi(2))
    }

    // Second derivative of link function using self.p
    fn link_2der_self(&self) -> DVector<f64> {
        self.link_2der(&self.p)
    }

    // Derivative of variance function wrt p
    fn V_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| 1.0 - 2.0 * p)
    }

    // Derivative of variance function using self.p
    fn V_der_self(&self) -> DVector<f64> {
        self.V_der(&self.p)
    }

    // Construct diagonal weight matrix from w(x)
    fn w_mat(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let N = self.Data.shape().0;
        let w_vec: nalgebra::Matrix<
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>,
        > = self.w(&x);
        DMatrix::from_diagonal(&w_vec)
    }
}
