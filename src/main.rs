extern crate machine_learning;

use std::io::{Write, stdin, stdout};

use machine_learning::{
    activation::*,
    *
};

fn main() {
    let mut ml = MachineLearner::<U2, U4, U2, Relu>::new();

    let data = [
        (Vect::<U2>::new(0., 0.), Vect::<U2>::new(0., 1.)),
        (Vect::<U2>::new(1., 0.), Vect::<U2>::new(1., 0.)),
        (Vect::<U2>::new(0., 1.), Vect::<U2>::new(1., 0.)),
        (Vect::<U2>::new(1., 1.), Vect::<U2>::new(0., 1.)),
    ];

    loop {
        let res = read_num("Input 1: ")
            .and_then(|in1| read_num("Input 2: ").map(|in2| (in1, in2)));
        let (in1, in2) = match res {
            Ok((in1, in2)) => (in1, in2),
            Err(e) => {
                match &*e {
                    "print" => ml.print(),
                    "train" => {
                        for _ in 0..10000 {
                            for &(i, t) in data.iter() {
                                ml.train(i, t);
                            }
                        }
                    }
                    "exit" => break,
                    _ => (),
                }
                continue;
            }
        };

        let result = ml.predict(Vect::<U2>::new(in1, in2));
        let (res, certainty, uncertainty);
        if result.x > result.y {
            res = 1;
            certainty = result.x * 100.;
            uncertainty = result.y * 100.;
        } else {
            res = 0;
            certainty = result.y * 100.;
            uncertainty = result.x * 100.;
        };
        println!("Result: {} ({:.2}%/{:.2})", res, certainty, uncertainty);
    }
}

fn read_num(query: &str) -> Result<f64, String> {
    print!("{}", query);
    let stdout = stdout();
    stdout.lock().flush().unwrap();

    let mut s = String::with_capacity(4);
    stdin().read_line(&mut s).unwrap();
    while let Some(c) = s.pop() {
        match c {
            '\n' | '\r' => continue,
            c => {
                s.push(c);
                break
            }
        }
    }

    match s.parse() {
        Ok(n) => Ok(n),
        Err(_) => Err(s),
    }
}
