pub trait Activation {
    fn sigma(z: f64) -> f64;
    fn dsigma(z: f64) -> f64;
    fn dsigma_of_y(_y: f64) -> f64 {
        unimplemented!()
    }
}

pub struct Relu;
impl Activation for Relu {
    fn sigma(n: f64) -> f64 {
        0f64.max(n)
    }
    fn dsigma(x: f64) -> f64 {
        if x > 0. {
            1.
        } else {
            0.
        }
    }
    fn dsigma_of_y(y: f64) -> f64 {
        if y > 0. {
            1.
        } else {
            0.
        }
    }
}
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn sigma(z: f64) -> f64 {
        (1.+(-z).exp()).recip()
    }
    fn dsigma(z: f64) -> f64 {
        Self::sigma(z) * (1. - Self::sigma(z))
    }
    fn dsigma_of_y(y: f64) -> f64 {
        y * (1. - y)
    }
}
