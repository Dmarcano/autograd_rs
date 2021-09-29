use autograd_rs::{
    layer::{fully_connected::DenseLayer, Layer},
    ops::activation::ActivationFuncs,
    Tensor,
};

use mnist::{Mnist, MnistBuilder};
use rand;

struct Model {
    layers: Vec<DenseLayer<f32>>,
}

struct LayerTopology {
    pub input: usize,
    pub output: usize,
    pub activation: Option<ActivationFuncs<f32>>,
}

impl Model {
    /// creates a new model of DenseLayers each with an input
    pub fn new(topology: &[LayerTopology]) -> Self {
        let mut rng = rand::rngs::OsRng::default();
        let layers: Vec<DenseLayer<f32>> = topology
            .iter()
            .map(|top| DenseLayer::new_random(top.input, top.output, top.activation, &mut rng))
            .collect();
        Self { layers }
    }
}

fn main() {
    let (trn_size, rows, cols) = (5_000, 28, 28);

    let num_classes = 10;
    let layer_1_out = 28 * 10;
    let top = [
        LayerTopology {
            input: rows * cols,
            output: layer_1_out,
            activation: Some(ActivationFuncs::Sigmoid),
        },
        LayerTopology {
            input: layer_1_out,
            output: num_classes,
            activation: Some(ActivationFuncs::Sigmoid),
        },
    ];

    println!("Hello, world!");
}
