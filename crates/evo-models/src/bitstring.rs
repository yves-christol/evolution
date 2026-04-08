//! Bitstring / OneMax model.
//!
//! Classic genetic algorithm benchmark: maximize the number of 1-bits
//! in a fixed-length bitstring.

use evo_core::{Environment, Organism, Reproducer};
use rand::Rng;
use rand::RngCore;
use rayon::prelude::*;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Organism
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct BitstringOrganism {
    pub bits: Vec<bool>,
}

impl Organism for BitstringOrganism {
    type Genome = Vec<bool>;

    fn genome(&self) -> &Vec<bool> {
        &self.bits
    }

    fn with_genome(genome: Vec<bool>) -> Self {
        BitstringOrganism { bits: genome }
    }
}

impl BitstringOrganism {
    /// Create a random organism with `length` bits.
    pub fn random(length: usize, rng: &mut dyn RngCore) -> Self {
        let bits: Vec<bool> = (0..length).map(|_| rng.random_bool(0.5)).collect();
        BitstringOrganism { bits }
    }
}

// ---------------------------------------------------------------------------
// Environment — OneMax
// ---------------------------------------------------------------------------

/// OneMax: fitness = number of 1-bits. Uses rayon for parallel evaluation.
#[derive(Debug, Clone)]
pub struct OneMaxEnvironment;

impl Environment for OneMaxEnvironment {
    type Org = BitstringOrganism;

    fn evaluate(&self, population: &[BitstringOrganism]) -> Vec<f64> {
        population
            .par_iter()
            .map(|org| org.bits.iter().filter(|&&b| b).count() as f64)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Reproducer — tournament selection + single-point crossover + bit-flip mutation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BitstringReproducer {
    /// Number of individuals in each tournament.
    pub tournament_size: usize,
    /// Per-bit probability of flipping during mutation.
    pub mutation_rate: f64,
    /// Fraction of top individuals preserved unchanged.
    pub elitism_ratio: f64,
}

impl Default for BitstringReproducer {
    fn default() -> Self {
        BitstringReproducer {
            tournament_size: 3,
            mutation_rate: 0.01,
            elitism_ratio: 0.05,
        }
    }
}

impl BitstringReproducer {
    /// Tournament selection: pick `tournament_size` random individuals, return the best.
    fn tournament_select<'a>(
        &self,
        population: &'a [BitstringOrganism],
        fitness: &[f64],
        rng: &mut dyn RngCore,
    ) -> &'a BitstringOrganism {
        let mut best_idx = rng.random_range(0..population.len());
        for _ in 1..self.tournament_size {
            let idx = rng.random_range(0..population.len());
            if fitness[idx] > fitness[best_idx] {
                best_idx = idx;
            }
        }
        &population[best_idx]
    }

    /// Single-point crossover.
    fn crossover(
        &self,
        parent_a: &BitstringOrganism,
        parent_b: &BitstringOrganism,
        rng: &mut dyn RngCore,
    ) -> BitstringOrganism {
        let len = parent_a.bits.len();
        let point = rng.random_range(1..len);
        let mut child_bits = Vec::with_capacity(len);
        child_bits.extend_from_slice(&parent_a.bits[..point]);
        child_bits.extend_from_slice(&parent_b.bits[point..]);
        BitstringOrganism { bits: child_bits }
    }

    /// Bit-flip mutation.
    fn mutate(&self, org: &mut BitstringOrganism, rng: &mut dyn RngCore) {
        for bit in &mut org.bits {
            if rng.random_bool(self.mutation_rate) {
                *bit = !*bit;
            }
        }
    }
}

impl Reproducer for BitstringReproducer {
    type Org = BitstringOrganism;

    fn reproduce(
        &self,
        population: &[BitstringOrganism],
        fitness: &[f64],
        target_size: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<BitstringOrganism> {
        let elite_count = ((target_size as f64) * self.elitism_ratio).round() as usize;

        // Sort by fitness descending to pick elites.
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            fitness[b]
                .partial_cmp(&fitness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_gen: Vec<BitstringOrganism> = indices
            .iter()
            .take(elite_count)
            .map(|&i| population[i].clone())
            .collect();

        // Fill remainder via tournament selection + crossover + mutation.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use evo_core::{World, WorldConfig};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn onemax_fitness_counts_ones() {
        let org = BitstringOrganism {
            bits: vec![true, false, true, true, false],
        };
        let env = OneMaxEnvironment;
        let scores = env.evaluate(&[org]);
        assert!((scores[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reproducer_preserves_population_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pop: Vec<BitstringOrganism> = (0..20)
            .map(|_| BitstringOrganism::random(32, &mut rng))
            .collect();
        let env = OneMaxEnvironment;
        let fitness = env.evaluate(&pop);
        let reproducer = BitstringReproducer::default();
        let next = reproducer.reproduce(&pop, &fitness, 20, &mut rng);
        assert_eq!(next.len(), 20);
    }

    #[test]
    fn onemax_converges() {
        let genome_length = 64;
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let initial: Vec<BitstringOrganism> = (0..200)
            .map(|_| BitstringOrganism::random(genome_length, &mut rng))
            .collect();

        let config = WorldConfig {
            population_size: 200,
            max_generations: 200,
            seed: 123,
        };

        let env = OneMaxEnvironment;
        let reproducer = BitstringReproducer {
            tournament_size: 3,
            mutation_rate: 0.01,
            elitism_ratio: 0.05,
        };

        let mut world = World::new(config, env, reproducer, initial);
        let final_snap = world.run(&mut rng, None, None);

        // After 200 generations with pop=200 on a 64-bit string,
        // we should be very close to the optimum (64.0).
        assert!(
            final_snap.best_fitness >= 60.0,
            "expected convergence near optimum, got {}",
            final_snap.best_fitness
        );
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        let run = |seed: u64| -> f64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let initial: Vec<BitstringOrganism> = (0..50)
                .map(|_| BitstringOrganism::random(32, &mut rng))
                .collect();
            let config = WorldConfig {
                population_size: 50,
                max_generations: 20,
                seed,
            };
            let mut world = World::new(config, OneMaxEnvironment, BitstringReproducer::default(), initial);
            let snap = world.run(&mut rng, None, None);
            snap.best_fitness
        };

        let a = run(999);
        let b = run(999);
        assert!(
            (a - b).abs() < f64::EPSILON,
            "same seed should produce same result: {a} vs {b}"
        );
    }
}
