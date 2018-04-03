pub trait Activation {
    fn sigma(x: f64) -> f64;
    fn dsigma(y: f64) -> f64;
}

pub struct Relu;
impl Activation for Relu {
    fn sigma(n: f64) -> f64 {
        0f64.max(n)
    }
    fn dsigma(y: f64) -> f64 {
        if y > 0. {
            1.
        } else {
            0.
        }
    }
}
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn sigma(x: f64) -> f64 {
        (-(-x).exp_m1()).recip()
    }
    fn dsigma(y: f64) -> f64 {
        y * (1. - y)
    }
}
