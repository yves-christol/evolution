use crate::stats::GenerationStats;
use crate::traits::{Environment, Organism, Reproducer};
use crossbeam_channel::Sender;
/// Configuration for a simulation run.
#[derive(Debug, Clone)]
pub struct WorldConfig {
    pub population_size: usize,
    pub max_generations: u64,
    pub seed: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            population_size: 1000,
            max_generations: 1000,
            seed: 42,
        }
    }
}

/// A snapshot emitted after each generation, consumed by observers (GUI, logger, etc.).
#[derive(Debug, Clone)]
pub struct Snapshot<O: Organism> {
    pub generation: u64,
    pub stats: GenerationStats,
    pub best: O,
    pub best_fitness: f64,
    pub population: Vec<O>,
    pub fitness: Vec<f64>,
}

/// Control handle for pausing/stopping the simulation from another thread.
pub struct RunControl {
    stop: std::sync::atomic::AtomicBool,
    pause: std::sync::atomic::AtomicBool,
}

impl RunControl {
    pub fn new() -> Self {
        RunControl {
            stop: std::sync::atomic::AtomicBool::new(false),
            pause: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn request_stop(&self) {
        self.stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn toggle_pause(&self) {
        self.pause
            .fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.pause.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for RunControl {
    fn default() -> Self {
        Self::new()
    }
}

/// The simulation engine.
///
/// Parameterized by organism, environment, and reproducer types for
/// zero-cost monomorphization in the hot loop.
pub struct World<O, E, R>
where
    O: Organism,
    E: Environment<Org = O>,
    R: Reproducer<Org = O>,
{
    pub config: WorldConfig,
    pub environment: E,
    pub reproducer: R,
    pub population: Vec<O>,
    pub fitness: Vec<f64>,
    pub generation: u64,
}

impl<O, E, R> World<O, E, R>
where
    O: Organism + Send + Sync,
    E: Environment<Org = O>,
    R: Reproducer<Org = O>,
{
    pub fn new(config: WorldConfig, environment: E, reproducer: R, initial_population: Vec<O>) -> Self {
        let size = initial_population.len();
        World {
            config,
            environment,
            reproducer,
            population: initial_population,
            fitness: vec![0.0; size],
            generation: 0,
        }
    }

    /// Run a single generation: evaluate, snapshot, select+reproduce.
    /// Returns the snapshot for this generation.
    pub fn step(&mut self, rng: &mut dyn rand::RngCore) -> Snapshot<O> {
        // Evaluate fitness (parallel)
        self.fitness = self.evaluate_parallel();

        // Compute stats
        let stats = GenerationStats::from_fitness(self.generation, &self.fitness);

        // Find best organism
        let best_idx = self
            .fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let snapshot = Snapshot {
            generation: self.generation,
            stats,
            best: self.population[best_idx].clone(),
            best_fitness: self.fitness[best_idx],
            population: self.population.clone(),
            fitness: self.fitness.clone(),
        };

        // Reproduce to create next generation
        let next_gen = self.reproducer.reproduce(
            &self.population,
            &self.fitness,
            self.config.population_size,
            rng,
        );
        self.population = next_gen;

        // Advance environment
        self.environment.step(self.generation);
        self.generation += 1;

        snapshot
    }

    /// Run the full simulation loop, sending snapshots through the channel.
    /// Respects RunControl for pause/stop from another thread.
    pub fn run(
        &mut self,
        rng: &mut dyn rand::RngCore,
        snapshot_tx: Option<&Sender<Snapshot<O>>>,
        control: Option<&RunControl>,
    ) -> Snapshot<O> {
        let mut last_snapshot = None;

        while self.generation < self.config.max_generations {
            // Check stop
            if let Some(ctrl) = control {
                if ctrl.is_stopped() {
                    break;
                }
                // Spin-wait while paused
                while ctrl.is_paused() && !ctrl.is_stopped() {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                if ctrl.is_stopped() {
                    break;
                }
            }

            let snapshot = self.step(rng);

            // Send snapshot to observer (non-blocking — drop if receiver is full)
            if let Some(tx) = snapshot_tx {
                let _ = tx.try_send(snapshot.clone());
            }

            last_snapshot = Some(snapshot);
        }

        last_snapshot.expect("simulation ran zero generations")
    }

    /// Parallel fitness evaluation using rayon.
    fn evaluate_parallel(&self) -> Vec<f64> {
        // For environments that can evaluate individuals independently,
        // we use rayon to parallelize. The Environment trait evaluates
        // the whole batch, so the implementor decides the parallelism
        // strategy internally. We call it on the full population.
        self.environment.evaluate(&self.population)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Environment, Organism, Reproducer};
    use rand::RngCore;

    // Minimal test organism: a single f64 value.
    #[derive(Clone, Debug)]
    struct TestOrg {
        value: f64,
    }

    impl Organism for TestOrg {
        type Genome = f64;

        fn genome(&self) -> &f64 {
            &self.value
        }

        fn with_genome(genome: f64) -> Self {
            TestOrg { value: genome }
        }
    }

    // Environment: fitness = value (maximize value).
    struct MaximizeEnv;

    impl Environment for MaximizeEnv {
        type Org = TestOrg;

        fn evaluate(&self, population: &[TestOrg]) -> Vec<f64> {
            population.iter().map(|o| o.value).collect()
        }
    }

    // Reproducer: keep top half, clone them to fill population, add small perturbation.
    struct SimpleReproducer;

    impl Reproducer for SimpleReproducer {
        type Org = TestOrg;

        fn reproduce(
            &self,
            population: &[TestOrg],
            fitness: &[f64],
            target_size: usize,
            rng: &mut dyn RngCore,
        ) -> Vec<TestOrg> {
            let mut indexed: Vec<(usize, f64)> =
                fitness.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let elite_count = target_size / 2;
            let mut next_gen: Vec<TestOrg> = indexed
                .iter()
                .take(elite_count)
                .map(|(i, _)| population[*i].clone())
                .collect();

            // Fill rest with mutated clones of elites
            let mut buf = [0u8; 8];
            while next_gen.len() < target_size {
                let parent = &next_gen[next_gen.len() % elite_count];
                rng.fill_bytes(&mut buf);
                let noise = f64::from_ne_bytes(buf) % 0.1;
                next_gen.push(TestOrg {
                    value: parent.value + noise.abs(),
                });
            }

            next_gen
        }
    }

    #[test]
    fn world_runs_generations() {
        let config = WorldConfig {
            population_size: 10,
            max_generations: 5,
            seed: 42,
        };

        let initial: Vec<TestOrg> = (0..10)
            .map(|i| TestOrg {
                value: i as f64,
            })
            .collect();

        let mut world = World::new(config, MaximizeEnv, SimpleReproducer, initial);
        let mut rng = rand::rng();

        let final_snapshot = world.run(&mut rng, None, None);

        assert_eq!(final_snapshot.generation, 4); // 0-indexed, ran 5 gens
        assert_eq!(world.generation, 5);
        assert!(final_snapshot.best_fitness >= 5.0); // should improve from initial
    }
}
