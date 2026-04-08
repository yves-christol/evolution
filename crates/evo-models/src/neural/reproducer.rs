//! Reproducer for neural creatures: tournament selection, weight crossover, Gaussian mutation.

use super::creature::Creature;
use evo_core::Reproducer;
use rand::Rng;
use rand::RngCore;

#[derive(Debug, Clone)]
pub struct CreatureReproducer {
    pub tournament_size: usize,
    /// Standard deviation of Gaussian noise added to weights during mutation.
    pub mutation_std: f32,
    /// Probability that any given weight is mutated.
    pub mutation_rate: f64,
    /// Fraction of top individuals preserved unchanged.
    pub elitism_ratio: f64,
}

impl Default for CreatureReproducer {
    fn default() -> Self {
        CreatureReproducer {
            tournament_size: 3,
            mutation_std: 0.3,
            mutation_rate: 0.1,
            elitism_ratio: 0.05,
        }
    }
}

impl CreatureReproducer {
    fn tournament_select<'a>(
        &self,
        population: &'a [Creature],
        fitness: &[f64],
        rng: &mut dyn RngCore,
    ) -> &'a Creature {
        let mut best_idx = rng.random_range(0..population.len());
        for _ in 1..self.tournament_size {
            let idx = rng.random_range(0..population.len());
            if fitness[idx] > fitness[best_idx] {
                best_idx = idx;
            }
        }
        &population[best_idx]
    }

    /// Uniform crossover on weight vectors.
    fn crossover(&self, a: &Creature, b: &Creature, rng: &mut dyn RngCore) -> Creature {
        let weights: Vec<f32> = a
            .weights
            .iter()
            .zip(b.weights.iter())
            .map(|(&wa, &wb)| if rng.random_bool(0.5) { wa } else { wb })
            .collect();
        Creature { weights }
    }

    /// Gaussian mutation on weights.
    fn mutate(&self, creature: &mut Creature, rng: &mut dyn RngCore) {
        for w in &mut creature.weights {
            if rng.random_bool(self.mutation_rate) {
                // Box-Muller transform for Gaussian noise
                let u1: f32 = rng.random_range(0.0001f32..1.0f32);
                let u2: f32 = rng.random_range(0.0f32..std::f32::consts::TAU);
                let noise = (-2.0 * u1.ln()).sqrt() * u2.cos() * self.mutation_std;
                *w = (*w + noise).clamp(-5.0, 5.0);
            }
        }
    }
}

impl Reproducer for CreatureReproducer {
    type Org = Creature;

    fn reproduce(
        &self,
        population: &[Creature],
        fitness: &[f64],
        target_size: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<Creature> {
        let elite_count = ((target_size as f64) * self.elitism_ratio).round() as usize;

        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            fitness[b]
                .partial_cmp(&fitness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_gen: Vec<Creature> = indices
            .iter()
            .take(elite_count)
            .map(|&i| population[i].clone())
            .collect();

        while next_gen.len() < target_size {
            let parent_a = self.tournament_select(population, fitness, rng);
            let parent_b = self.tournament_select(population, fitness, rng);
            let mut child = self.crossover(parent_a, parent_b, rng);
            self.mutate(&mut child, rng);
            next_gen.push(child);
        }

        next_gen
    }
}
