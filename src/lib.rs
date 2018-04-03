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
    + Allocator<f64, H, I> + Allocator<f64, O> + Allocator<f64, O, H> {
    biases_hidden: Vect<H>,
    weights_in: Matr<H, I>,
    biases_out: Vect<O>,
    weights_out: Matr<O, H>,
    activation: PhantomData<A>,
}

impl<I: Dim + DimName, H: Dim + DimName, O: Dim + DimName, A: Activation> MachineLearner<I, H, O, A>
where DefaultAllocator: Allocator<f64, H>
    + Allocator<f64, H, I> + Allocator<f64, O> + Allocator<f64, O, H> {
    pub fn new() -> Self {
        MachineLearner {
            biases_hidden: Vect::<H>::from_fn(|_, _| rand::random()),
            weights_in: Matr::<H, I>::from_fn(|_, _| rand::random()),
            biases_out: Vect::<O>::from_fn(|_, _| rand::random()),
            weights_out: Matr::<O, H>::from_fn(|_, _| rand::random()),
            activation: PhantomData,
        }
    }
    pub fn print(&self)
    where Vect<H>: Display,
    Matr<H, I>: Display,
    Vect<O>: Display,
    Matr<O, H>: Display {
        print!("Weights (i>h):{}", self.weights_in);
        print!("Biases (i>h):{}", self.biases_hidden);
        print!("Weights (h>o):{}", self.weights_out);
        print!("Biases (h>o):{}", self.biases_out);
    }
    pub fn predict(&self, inputs: Vect<I>) -> Vect<O>
    where DefaultAllocator: Allocator<f64, H> + Allocator<f64, I> {
        let hidden_layer_activation = (self.weights_in.clone() * inputs + self.biases_hidden.clone()).map(A::sigma);
        let output_layer_activation = (self.weights_out.clone() * hidden_layer_activation + self.biases_out.clone()).map(A::sigma);
        output_layer_activation
    }
    pub fn train(&mut self, inputs: Vect<I>, targets: Vect<O>)
        where DefaultAllocator: Allocator<f64, H> + Allocator<f64, I> + Allocator<f64, U1, H> + Allocator<f64, U1, I> {
        const LEARNING_RATE: f64 = 0.1;
        // Generating the Hidden Outputs
        let hidden = (self.weights_in.clone() * inputs.clone() + self.biases_hidden.clone()).map(A::sigma);

        // Generating the output's output!
        let outputs = (self.weights_out.clone() * hidden.clone() + self.biases_out.clone()).map(A::sigma);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let output_errors = targets - outputs.clone();

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        let gradients = outputs.map(A::dsigma).component_mul(&output_errors) * LEARNING_RATE;

        // Calculate deltas
        let weight_ho_deltas = gradients.clone() * hidden.transpose();

        // Adjust the weights by deltas
        self.weights_out += weight_ho_deltas;
        // Adjust the bias by its deltas (which is just the gradients)
        self.biases_out += gradients;

        // Calculate the hidden layer errors
        let hidden_errors = self.weights_out.tr_mul(&output_errors);

        // Calculate hidden gradient
        let hidden_gradient = hidden.map(A::dsigma).component_mul(&hidden_errors) * LEARNING_RATE;

        // Calcuate input->hidden deltas
        let weight_ih_deltas = hidden_gradient.clone() * inputs.transpose();

        self.weights_in += weight_ih_deltas;
        // Adjust the bias by its deltas (which is just the gradients)
        self.biases_hidden += hidden_gradient;

        // outputs.print();
        // targets.print();
        // error.print();
    }
}

pub fn vect_make<D: Dim + DimName>(data: &[f64]) -> Vect<D>
where DefaultAllocator: Allocator<f64, D> {
    Vect::from_row_slice(data)
}
