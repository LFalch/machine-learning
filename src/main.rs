extern crate machine_learning;

use machine_learning::{
    activation::Relu,
    MachineLearner, vect_make, U2, U1, Vect,
};

fn main() {
    let mut ml = MachineLearner::<U2, U2, U1, Relu>::new();
    ml.print();

    let result = ml.predict(vect_make(&[1., 0.]));
    println!("Result:{}", result);

    let data = [
        (Vect::<U2>::new(0., 0.), Vect::<U1>::new(0.)),
        (Vect::<U2>::new(1., 0.), Vect::<U1>::new(0.)),
        (Vect::<U2>::new(0., 1.), Vect::<U1>::new(0.)),
        (Vect::<U2>::new(1., 1.), Vect::<U1>::new(1.)),
    ];

    for _ in 0..1000 {
        for &(i, t) in data.iter() {
            ml.train(i, t);
        }
    }
    ml.print();

    let result = ml.predict(vect_make(&[1., 0.]));
    println!("Result:{}", result);
    let result = ml.predict(vect_make(&[1., 1.]));
    println!("Result:{}", result);
}
