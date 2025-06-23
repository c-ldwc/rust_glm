use super::family::Family;
use itertools::{Zip, multizip};
use nalgebra::{DMatrix, DVector};

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
    fn link(&self) -> DVector<f64> {
        self.p.map(|p| (p / (1.0 - p)).ln())
    }

    //Var[y] = V(mu) * scale
    fn V(&self) -> DVector<f64> {
        self.p.map(|p| self.n * p * (1.0 - p))
    }

    //Alpha as defined by Wood pp.106
    fn alpha(&self) -> DVector<f64> {
        let mapped = multizip((
            self.y.iter(),
            self.p.iter(),
            self.V().iter(),
            self.V_der().iter(),
            self.link_der().iter(),
            self.link_2der().iter(),
        ))
        .map(|(y, p, v, v_der, link_der, link_2der)| {
            1.0 + (y - p) * (v_der / v + link_2der / link_der)
        })
        .collect::<Vec<f64>>();
        DVector::<f64>::from_vec(mapped)
    }

    //Weights as defined by wood pp.106
    fn w(&self) -> DVector<f64> {
        let mapped = multizip((
            self.alpha().iter(),
            self.link_der().iter(),
            self.V().iter(),
        ))
            .map(|(a, l, V)| a / (l.powi(2) * V))
            .collect::<Vec<f64>>();
        DVector::<f64>::from_vec(mapped)
    }

    fn scale(&self) -> DVector<f64> {
        DVector::repeat(self.n as usize, 1.0)
    }

    //p is updated on each Newton step
    fn hessian(&self) -> DMatrix<f64> {
        let X_shape = self.Data.shape();

        let neg_X: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, _> =
            DMatrix::repeat(X_shape.0, X_shape.1, -1.0).component_mul(&self.Data);
        let w = self.w_mat();
        neg_X.transpose() * w * &self.Data
    }

    fn grad(&self) -> DVector<f64> {
        let N = self.Data.shape().0;
        let mut G_vec = vec![0.0; N];

        let link_der = self.link_der();
        let link_der_iter = link_der.iter();
        let a = self.alpha();
        let a_iter = a.iter();

        let mut G_vec_storage = multizip((a_iter, link_der_iter))
            .map(|(a, ld)| ld / a)
            .collect::<Vec<f64>>();

        let G_vec = DVector::<f64>::from_vec(G_vec_storage);

        let G = DMatrix::from_diagonal(&G_vec);

        let neg_p = self.p.map(|p| -1.0 * p);
        let w = self.w_mat();
        self.Data.transpose() * w * G * (&self.y + neg_p)
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
        l.map(|x| 1.0 / (1.0 - (-x).exp()))
    }

    // derivative of link
    fn link_der(&self) -> DVector<f64> {
        self.p.map(|p| 1.0 / (p * (1.0 - p)))
    }

    //second derivative
    fn link_2der(&self) -> DVector<f64> {
        self.p.map(|p| (2.0 * p - 1.0) / (p * (1.0 - p)).powi(2))
    }

    //Derivative of V
    fn V_der(&self) -> DVector<f64> {
        self.p.map(|p| 1.0 - 2.0 * p)
    }

    fn w_mat(&self) -> DMatrix<f64> {
        let N = self.Data.shape().0;
        let w_vec: nalgebra::Matrix<
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>,
        > = self.w();
        DMatrix::from_diagonal(&w_vec)
    }
}
