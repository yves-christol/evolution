//! Neural creature organism.

use evo_core::Organism;
use serde::Serialize;

/// A creature whose genome is a flat vector of neural network weights.
#[derive(Debug, Clone, Serialize)]
pub struct Creature {
    pub weights: Vec<f32>,
}

impl Organism for Creature {
    type Genome = Vec<f32>;

    fn genome(&self) -> &Vec<f32> {
        &self.weights
    }

    fn with_genome(genome: Vec<f32>) -> Self {
        Creature { weights: genome }
    }
}

impl Creature {
    /// Create a creature with random weights in [-1, 1].
    pub fn random(num_weights: usize, rng: &mut dyn rand::RngCore) -> Self {
        use rand::Rng;
        let weights: Vec<f32> = (0..num_weights)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        Creature { weights }
    }
}
