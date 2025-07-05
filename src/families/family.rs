use nalgebra::{DMatrix, DVector};
pub trait Family {
    type Parameters;
    type Data;
    type y;
    type coef;
    fn link(&self, p: &DVector<f64>) -> DVector<f64>;
    fn V(&self, p: &DVector<f64>) -> DVector<f64>;
    fn alpha(&self, x: &DVector<f64>) -> DVector<f64>;
    fn w(&self, x: &DVector<f64>) -> DVector<f64>;
    fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64>;
    fn grad(&self, x: &DVector<f64>) -> DVector<f64>;
    fn scale(&self, x: &DVector<f64>) -> DVector<f64>;
    fn neg_log_lik(&self, x: &DVector<f64>) -> f64; //Evaluate the neg log likelihod at the point currently stored in self
}
