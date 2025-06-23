use nalgebra::{DMatrix, DVector};
pub trait Family {
    type Parameters;
    type Data;
    type y;
    type coef;
    fn link(&self) -> DVector<f64>;
    fn V(&self) -> DVector<f64>;
    fn alpha(&self) -> DVector<f64>;
    fn w(&self) -> DVector<f64>;
    fn hessian(&self) -> DMatrix<f64>;
    fn grad(&self) -> DVector<f64>;
    fn scale(&self) -> DVector<f64>;
}