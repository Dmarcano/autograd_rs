use autograd_rs::{
    Tensor, 
    layer::{fully_connected::DenseLayer, Layer},
    ops::activation::ActivationFuncs,
};

use rand;

struct Model {
    layers: Vec<DenseLayer<f64>>,
}

struct LayerTopology {
    pub input: usize,
    pub output: usize,
    pub activation: Option<ActivationFuncs<f64>>,
}

impl Model {
    /// creates a new model of DenseLayers each with an input
    pub fn new(topology: &[LayerTopology]) -> Self {
        let mut rng = rand::rngs::OsRng::default();
        let layers: Vec<DenseLayer<f64>> = topology
            .iter()
            .map(|top| DenseLayer::new_random(top.input, top.output, top.activation, &mut rng))
            .collect();
        Self { 
            layers
        }
    }
}

fn main() {
    println!("Hello, world!");
}
