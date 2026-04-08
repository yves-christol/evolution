use rand::RngCore;

/// A single organism in the simulation.
///
/// Generic over its genome representation — could be a bitstring, a neural
/// network weight vector, a tree structure, etc.
pub trait Organism: Clone + Send + Sync {
    type Genome: Clone + Send + Sync;

    fn genome(&self) -> &Self::Genome;
    fn with_genome(genome: Self::Genome) -> Self;
}

/// Evaluates organisms within the context of an environment.
///
/// The environment assigns fitness scores and may itself change over time
/// (e.g., shifting resource distributions, seasonal changes).
pub trait Environment: Send + Sync {
    type Org: Organism;

    /// Evaluate and assign fitness to each organism in the population.
    /// Returns a fitness value for each organism, in the same order.
    fn evaluate(&self, population: &[Self::Org]) -> Vec<f64>;

    /// Advance the environment by one generation. Override for dynamic environments.
    fn step(&mut self, _generation: u64) {}
}

/// Defines selection, crossover, and mutation.
pub trait Reproducer: Send + Sync {
    type Org: Organism;

    /// Given a population with their fitness scores, produce the next generation.
    fn reproduce(
        &self,
        population: &[Self::Org],
        fitness: &[f64],
        target_size: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<Self::Org>;
}
