use autograd_rs::{
    layer::{fully_connected::DenseLayer, Layer},
    ops::activation::ActivationFuncs,
    Tensor,
};

use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{concatenate, Array2, Axis};
use rand::rngs::OsRng;

extern crate blas_src;

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
        self.layers
            .iter()
            .fold(data.clone(), |input, layer| layer.forward(&input))
    }

    fn update_parameters(&mut self, rate: f32) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.update_parameters(rate))
    }

    fn clear_grad(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.clear_grad())
    }
}

fn main() {
    let (trn_size, rows, cols) = (1000, 28, 28);
    let test_size = 200;
    let validation_size = 200;

    let num_classes = 10;
    let layer_1_out = 300;
    let top = [
        LayerTopology {
            input: rows * cols,
            output: layer_1_out,
            activation: Some(ActivationFuncs::TanH),
        },
        LayerTopology {
            input: layer_1_out,
            output: num_classes,
            activation: Some(ActivationFuncs::TanH),
        },
    ];

    let mut model = Model::new(&top);

    let NormalizedMnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(trn_size)
        .validation_set_length(validation_size)
        .test_set_length(test_size)
        .finalize()
        .normalize();

    let input_tensor = mnist_vec_to_tensor(trn_img, rows * cols, trn_size as usize);
    let tartget_tensor = mnist_vec_to_tensor(
        trn_lbl.into_iter().map(|v| v as f32).collect(),
        num_classes,
        trn_size as usize,
    );

    let num_epochs = 300; 
    let learning_rate = - 0.5; 

    for _ in 0..num_epochs { 
        let output = model.forward(&input_tensor); 

        let diff = &tartget_tensor - &output; 
        let squared_diff = &diff * &diff;
        println!("MSE: {:?}", squared_diff.data.mean());
        squared_diff.backward().unwrap();
        model.update_parameters(learning_rate);
        model.clear_grad();

    }
}

/// takes a vector which corresponds into mnist data and converts it into a proper Tensor to use in machine learning models
fn mnist_vec_to_tensor(data: Vec<f32>, image_size: usize, num_images: usize) -> Tensor<f32> {
    let arr = Array2::from_shape_vec((image_size, num_images), data).unwrap();
    Tensor::new_from_arr(arr)
}
