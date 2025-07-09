use nalgebra::{DMatrix, DVector};
use itertools::{multizip};
pub trait Family {
    type Parameters;
    type Data;
    type y;
    type coef;
    fn get_data(&self) -> &DMatrix<f64>;
    fn get_y(&self) -> &DVector<f64>;
    fn link(&self, mu: &DVector<f64>) -> DVector<f64>;
    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64>;
    fn link_der(&self, mu: &DVector<f64>) -> DVector<f64>;
    fn V(&self, p: &DVector<f64>) -> DVector<f64>;
    fn alpha(&self, x: &DVector<f64>) -> DVector<f64>;
    fn w(&self, x: &DVector<f64>) -> DVector<f64>;
    // Hessian matrix computation for Newton-Raphson or IRLS
    fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        // Compute -X^T * W * X efficiently
        let mut neg_X_t = self.get_data()
        .clone()
        .scale(-1.0)
        .transpose();
        let w = self.w(&x);

        // Scale columns of -X^T by weights
        for i in 0..neg_X_t.shape().1 {
            neg_X_t.column_mut(i).scale_mut(w[i]);
        }
        neg_X_t * self.get_data().scale(1.0/self.scale())
    }

    // Gradient of the log-likelihood
    fn grad(&self, x: &DVector<f64>) -> DVector<f64> {
        let N = self.get_data().shape().0;

        let logits = self.get_data() * x;

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
        let mut X = self.get_data().clone();

        // Save a bit of time here by avoiding two matmuls with diagonal matrices
        // Multiply the rows of X (the operation requires the transpose) instead
        for i in 0..X.shape().0{
            X.row_mut(i).scale_mut(WG_vec[i]);
        }

        // Return X^T * (y - p)
        X.transpose() * (self.get_y() - p)
    }
    fn log_lik(&self, x: &DVector<f64>) -> f64;
    fn scale(&self) -> f64;
}
