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

    pub fn clear_grad(&mut self) { 
        self.layers.iter_mut().for_each(|layer|layer.clear_grad())
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
    
    let inputs = Tensor::new_from_arr(inputs).untracked(); 
    let targets = Tensor::new_from_arr(array![[-1.0], [1.0], [1.0], [-1.0]].t().to_owned()).untracked();

    println!("targets: {:#?}", targets.data);

    let top = [
        LayerTopology{input : 2, output: 4, activation : None},
        LayerTopology{input : 4, output: 1, activation : Some(ActivationFuncs::TanH)}
    ]; 

    let mut net = Model::new(&top);

    let learning_rate = -0.2;
    let num_epochs = 100;

    for _ in 0..num_epochs { 

        println!("======================= NEW EPOCH =======================");

        println!("targets {:#?}", targets.data); 
        println!("=============================================");

        let output = net.forward(&inputs); 
        println!("outputs {:#?}", output.data); 

        let diff = &targets - &output; 
        println!("=============================================");
        println!("diff {:#?}", diff.data); 
        println!("=============================================\n\n");
        let squared_diff = &diff*&diff; 
        squared_diff.backward().unwrap();
        net.update_parameters(learning_rate);
        net.clear_grad();
    }
}
