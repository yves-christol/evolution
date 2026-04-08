//! Simple feedforward neural network.
//!
//! Architecture: input → hidden → output, with tanh activation.
//! Weights are stored as a flat Vec<f32> (the genome).

use serde::Serialize;

/// A fixed-architecture feedforward neural network.
#[derive(Debug, Clone, Serialize)]
pub struct Brain {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    /// Flat weight vector: [input→hidden weights, hidden biases, hidden→output weights, output biases]
    pub weights: Vec<f32>,
}

impl Brain {
    /// Total number of weights for the given architecture.
    pub fn weight_count(input_size: usize, hidden_size: usize, output_size: usize) -> usize {
        input_size * hidden_size + hidden_size + hidden_size * output_size + output_size
    }

    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, weights: Vec<f32>) -> Self {
        let expected = Self::weight_count(input_size, hidden_size, output_size);
        assert_eq!(
            weights.len(),
            expected,
            "expected {expected} weights, got {}",
            weights.len()
        );
        Brain {
            input_size,
            hidden_size,
            output_size,
            weights,
        }
    }

    /// Feed input through the network, return output activations.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_size);

        let ih_size = self.input_size * self.hidden_size;
        let ih_weights = &self.weights[..ih_size];
        let h_biases = &self.weights[ih_size..ih_size + self.hidden_size];

        let ho_start = ih_size + self.hidden_size;
        let ho_size = self.hidden_size * self.output_size;
        let ho_weights = &self.weights[ho_start..ho_start + ho_size];
        let o_biases = &self.weights[ho_start + ho_size..];

        // Hidden layer
        let mut hidden = vec![0.0f32; self.hidden_size];
        for h in 0..self.hidden_size {
            let mut sum = h_biases[h];
            for i in 0..self.input_size {
                sum += input[i] * ih_weights[i * self.hidden_size + h];
            }
            hidden[h] = sum.tanh();
        }

        // Output layer
        let mut output = vec![0.0f32; self.output_size];
        for o in 0..self.output_size {
            let mut sum = o_biases[o];
            for h in 0..self.hidden_size {
                sum += hidden[h] * ho_weights[h * self.output_size + o];
            }
            output[o] = sum.tanh();
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_count_correct() {
        assert_eq!(Brain::weight_count(4, 6, 2), 4 * 6 + 6 + 6 * 2 + 2);
    }

    #[test]
    fn forward_produces_output() {
        let n_weights = Brain::weight_count(3, 4, 2);
        let weights = vec![0.1; n_weights];
        let brain = Brain::new(3, 4, 2, weights);
        let out = brain.forward(&[1.0, 0.5, -0.5]);
        assert_eq!(out.len(), 2);
        // With all-positive weights and mixed input, outputs should be in (-1, 1)
        for &v in &out {
            assert!(v > -1.0 && v < 1.0);
        }
    }

    #[test]
    fn zero_weights_give_zero_outputs() {
        let n_weights = Brain::weight_count(3, 4, 2);
        let weights = vec![0.0; n_weights];
        let brain = Brain::new(3, 4, 2, weights);
        let out = brain.forward(&[1.0, 2.0, 3.0]);
        for &v in &out {
            assert!((v).abs() < f32::EPSILON);
        }
    }
}
