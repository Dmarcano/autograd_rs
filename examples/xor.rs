extern crate autograd_rs;

use std::convert::TryFrom;

use autograd_rs::{Tensor, layer::{fully_connected::DenseLayer, Layer}, ops::activation::ActivationFuncs, tensor};
use rand::rngs::OsRng;
use ndarray::array; 

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
        let mut rng = OsRng::default();
        let layers: Vec<DenseLayer<f32>> = topology
            .iter()
            .map(|top| DenseLayer::new_random(top.input, top.output, top.activation, &mut rng))
            .collect();
        Self { layers }
    }
}

impl Layer<f32> for Model {
    fn forward(&self, data: &Tensor<f32>) -> Tensor<f32> {
        self.layers.iter().fold(data.clone(), |input, layer| {
            layer.forward(&input)
        })
    }

    fn update_parameters(&mut self, rate: f32) {
        self.layers.iter_mut().for_each(|layer| layer.update_parameters(rate))
    }
}

fn main() {
    // neural networks expect column vector representation of inputs
    let inputs = array![
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ].t().to_owned();
    
    let inputs = Tensor::new_from_arr(inputs); 
    let targets = array![[-1.0], [1.0], [1.0], [-1.0]];

    let top = [
        LayerTopology{input : 2, output: 2, activation : Some(ActivationFuncs::TanH)},
        LayerTopology{input : 2, output: 1, activation : Some(ActivationFuncs::TanH)}
    ]; 

    let net = Model::new(&top);

    let num_epochs = 3;

    for _ in 0..num_epochs { 
        let output = net.forward(&inputs); 
        println!("=============================================");
        println!("{:#?}", output.data); 
        println!("=============================================");
    }
}
