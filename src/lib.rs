extern crate nalgebra;
extern crate rand;
extern crate generic_array;

use std::{
    marker::PhantomData,
    fmt::Display
};

use nalgebra::{
    allocator::Allocator,
    default_allocator::DefaultAllocator,
    dimension::{Dim, DimName},
    VectorN, MatrixMN
};

pub mod activation;
pub use nalgebra::dimension::{U1, U2, U3, U4, U5, U6};

use activation::Activation;

pub type Vect<D> = VectorN<f64, D>;
pub type Matr<R, C> = MatrixMN<f64, R, C>;

#[derive(Debug, Clone)]
pub struct MachineLearner<I: Dim + DimName, H: Dim + DimName, O: Dim + DimName, A: Activation>
where DefaultAllocator: Allocator<f64, H>
    + Allocator<f64, H, I> + Allocator<f64, O> + Allocator<f64, O, H>
    + Allocator<f64, H> + Allocator<f64, I> + Allocator<f64, U1, H> + Allocator<f64, U1, I> {
    bias_h: Vect<H>,
    weight_ih: Matr<H, I>,
    bias_o: Vect<O>,
    weight_ho: Matr<O, H>,
    activation: PhantomData<A>,
}

impl<I: Dim + DimName, H: Dim + DimName, O: Dim + DimName, A: Activation> MachineLearner<I, H, O, A>
where DefaultAllocator: Allocator<f64, H>
    + Allocator<f64, H, I> + Allocator<f64, O> + Allocator<f64, O, H>
    + Allocator<f64, H> + Allocator<f64, I> + Allocator<f64, U1, H> + Allocator<f64, U1, I> {
    pub fn new() -> Self {
        MachineLearner {
            bias_h: Vect::<H>::from_fn(|_, _| rand::random()),
            weight_ih: Matr::<H, I>::from_fn(|_, _| rand::random()),
            bias_o: Vect::<O>::from_fn(|_, _| rand::random()),
            weight_ho: Matr::<O, H>::from_fn(|_, _| rand::random()),
            activation: PhantomData,
        }
    }
    pub fn print(&self)
    where Vect<H>: Display,
    Matr<H, I>: Display,
    Vect<O>: Display,
    Matr<O, H>: Display {
        print!("Weights (i>h):{}", self.weight_ih);
        print!("Biases (h):{}", self.bias_h);
        print!("Weights (h>o):{}", self.weight_ho);
        print!("Biases (o):{}", self.bias_o);
    }
    pub fn feed_forward(&self, inputs: Vect<I>) -> (Vect<H>, Vect<O>) {
        let a_h = (self.weight_ih.clone() * inputs + self.bias_h.clone()).map(A::sigma);
        let a_o = (self.weight_ho.clone() * a_h.clone() + self.bias_o.clone()).map(A::sigma);
        (a_h, a_o)
    }
    pub fn predict(&self, inputs: Vect<I>) -> Vect<O> {
        self.feed_forward(inputs).1
    }
    pub fn train(&mut self, inputs: Vect<I>, targets: Vect<O>)
    where Vect<O>: Display {
        const LEARNING_RATE: f64 = 0.1;

        let (hidden, outputs) = self.feed_forward(inputs.clone());

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let output_errors = targets - outputs.clone();

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        let gradients = outputs.map(A::dsigma_of_y).component_mul(&output_errors) * LEARNING_RATE;

        // Calculate deltas
        let weight_ho_deltas = gradients.clone() * hidden.transpose();

        // Adjust the weights by deltas
        self.weight_ho += weight_ho_deltas;
        // Adjust the bias by its deltas (which is just the gradients)
        self.bias_o += gradients;

        // Calculate the hidden layer errors
        let hidden_errors = self.weight_ho.tr_mul(&output_errors);

        // Calculate hidden gradient
        let hidden_gradient = hidden.map(A::dsigma).component_mul(&hidden_errors) * LEARNING_RATE;

        // Calcuate input->hidden deltas
        let weight_ih_deltas = hidden_gradient.clone() * inputs.transpose();

        self.weight_ih += weight_ih_deltas;
        // Adjust the bias by its deltas (which is just the gradients)
        self.bias_h += hidden_gradient;
    }
}
