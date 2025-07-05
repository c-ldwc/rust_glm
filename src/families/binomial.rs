use super::family::Family;
use itertools::{Zip, multizip};
use nalgebra::{DMatrix, DVector};
use rug::{Float, Integer}; //For binomial coefficient

//A member of the exponential family has its pdf structured as
//exp( [y\theta - b(theta)] / a(theta) + c(y, theta))
//Because the partial derivative of the log likelihood wrt theta has expectation 0
//b'(theta) = E[y]

pub struct Binomial {
    n: f64,
    Data: DMatrix<f64>,
    y: DVector<f64>,
    coef: DVector<f64>,
    pub p: DVector<f64>,
}

impl Family for Binomial {
    type Parameters = (f64,);
    type Data = DMatrix<f64>;
    type coef = DVector<f64>;
    type y = DVector<f64>;
    // link fn. Log odds
    fn link(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| (p / (1.0 - p)).ln())
    }

    //Var[y] = V(mu) * scale
    fn V(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| self.n * p * (1.0 - p))
    }

    //Alpha as defined by Wood pp.106
    fn alpha(&self, x: &DVector<f64>) -> DVector<f64> {
        let l: DVector<f64> = &self.Data * x;
        let p: DVector<f64> = self.inv_link(&l);

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

    //Weights as defined by wood pp.106
    fn w(&self, x: &DVector<f64>) -> DVector<f64> {
        let l = &self.Data * x;
        let p = self.inv_link(&l);

        let mapped = multizip((
            self.alpha(x).iter(),
            self.link_der(&p).iter(),
            self.V(&p).iter(),
        ))
        .map(|(a, l, V)| a / (l.powi(2) * V))
        .collect::<Vec<f64>>();
        DVector::<f64>::from_vec(mapped)
    }

    fn scale(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::repeat(self.n as usize, 1.0)
    }

    //p is updated on each Newton step
    fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let X_shape = self.Data.shape();

        let neg_X: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, _> =
            DMatrix::repeat(X_shape.0, X_shape.1, -1.0).component_mul(&self.Data);
        let w = self.w_mat(&x);
        neg_X.transpose() * w * &self.Data
    }

    fn grad(&self, x: &DVector<f64>) -> DVector<f64> {
        let N = self.Data.shape().0;
        let mut G_vec = vec![0.0; N];

        let logits = &self.Data * x;

        let p = self.inv_link(&logits);

        let a = self.alpha(&x);
        let a_iter = a.iter();

        let mut G_vec_storage = multizip((a_iter, self.link_der(&p).iter()))
            .map(|(a, ld)| ld / a)
            .collect::<Vec<f64>>();

        let G_vec = DVector::<f64>::from_vec(G_vec_storage);

        let G = DMatrix::from_diagonal(&G_vec);
        let w = self.w_mat(&x);
        self.Data.transpose() * w * G * (&self.y + p)
    }

    fn neg_log_lik(&self, x: &DVector<f64>) -> f64 {
        let l = &self.Data * x;
        let p = self.inv_link(&l);
        let w = self.w(&x);
        let theta = self.link(&p);
        let b = theta.map(|t| self.n * (1.0 + t.exp()).ln());
        let n_int = Integer::from(self.n as i32);
        let c = self
            .y
            .map(|y| n_int.clone().binomial(y as u32).to_f64().ln());

        -1.0 * multizip((w.iter(), self.y.iter(), theta.iter(), b.iter(), c.iter()))
            .map(|(w, y, t, b, c)| w * (y * t - b) + c)
            .sum::<f64>()
    }
}

impl Binomial {
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

    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64> {
        l.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    // derivative of link
    fn link_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| 1.0 / (p * (1.0 - p)))
    }

    fn link_der_self(&self) -> DVector<f64> {
        self.link_der(&self.p)
    }

    //second derivative
    fn link_2der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| (2.0 * p - 1.0) / (p * (1.0 - p)).powi(2))
    }

    fn link_2der_self(&self) -> DVector<f64> {
        self.link_2der(&self.p)
    }

    //Derivative of V
    fn V_der(&self, p: &DVector<f64>) -> DVector<f64> {
        p.map(|p| 1.0 - 2.0 * p)
    }

    fn V_der_self(&self) -> DVector<f64> {
        self.V_der(&self.p)
    }

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
