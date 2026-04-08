use clap::{Parser, Subcommand};
use evo_core::{GenerationStats, World, WorldConfig};
use evo_models::bitstring::{BitstringOrganism, BitstringReproducer, OneMaxEnvironment};
use evo_models::neural::brain::Brain;
use evo_models::neural::grid::{GridConfig, GridEnvironment, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE};
use evo_models::neural::{Creature, CreatureReproducer};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Parser)]
#[command(name = "evo-cli", about = "Headless evolution simulator")]
struct Cli {
    #[command(subcommand)]
    model: Model,
}

#[derive(Subcommand)]
enum Model {
    /// Run the bitstring / OneMax benchmark
    Bitstring {
        /// Length of the bitstring genome
        #[arg(long, default_value_t = 100)]
        genome_length: usize,
        /// Population size
        #[arg(long, default_value_t = 500)]
        population: usize,
        /// Maximum number of generations
        #[arg(long, default_value_t = 500)]
        generations: u64,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Tournament size
        #[arg(long, default_value_t = 3)]
        tournament_size: usize,
        /// Per-bit mutation rate
        #[arg(long, default_value_t = 0.01)]
        mutation_rate: f64,
        /// Elitism ratio
        #[arg(long, default_value_t = 0.05)]
        elitism: f64,
        /// Print stats every N generations
        #[arg(long, default_value_t = 1)]
        print_every: u64,
    },
    /// Run the neural creature grid simulation
    Neural {
        /// Population size
        #[arg(long, default_value_t = 300)]
        population: usize,
        /// Maximum number of generations
        #[arg(long, default_value_t = 200)]
        generations: u64,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Grid width
        #[arg(long, default_value_t = 30)]
        grid_width: usize,
        /// Grid height
        #[arg(long, default_value_t = 30)]
        grid_height: usize,
        /// Number of food items
        #[arg(long, default_value_t = 40)]
        food: usize,
        /// Number of poison items
        #[arg(long, default_value_t = 15)]
        poison: usize,
        /// Steps per evaluation
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Tournament size
        #[arg(long, default_value_t = 3)]
        tournament_size: usize,
        /// Gaussian mutation std dev
        #[arg(long, default_value_t = 0.3)]
        mutation_std: f32,
        /// Per-weight mutation probability
        #[arg(long, default_value_t = 0.1)]
        mutation_rate: f64,
        /// Elitism ratio
        #[arg(long, default_value_t = 0.05)]
        elitism: f64,
        /// Print stats every N generations
        #[arg(long, default_value_t = 1)]
        print_every: u64,
    },
}

fn print_header() {
    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>5}",
        "gen", "min", "mean", "max", "stddev", "pop"
    );
    println!("{}", "-".repeat(54));
}

fn print_stats(s: &GenerationStats, print_every: u64) {
    if print_every == 0 || s.generation.is_multiple_of(print_every) {
        println!(
            "{:>6}  {:>8.2}  {:>8.2}  {:>8.2}  {:>8.4}  {:>5}",
            s.generation,
            s.min_fitness,
            s.mean_fitness,
            s.max_fitness,
            s.std_dev_fitness,
            s.population_size,
        );
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.model {
        Model::Bitstring {
            genome_length,
            population,
            generations,
            seed,
            tournament_size,
            mutation_rate,
            elitism,
            print_every,
        } => run_bitstring(
            genome_length,
            population,
            generations,
            seed,
            tournament_size,
            mutation_rate,
            elitism,
            print_every,
        ),
        Model::Neural {
            population,
            generations,
            seed,
            grid_width,
            grid_height,
            food,
            poison,
            steps,
            tournament_size,
            mutation_std,
            mutation_rate,
            elitism,
            print_every,
        } => run_neural(
            population,
            generations,
            seed,
            grid_width,
            grid_height,
            food,
            poison,
            steps,
            tournament_size,
            mutation_std,
            mutation_rate,
            elitism,
            print_every,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bitstring(
    genome_length: usize,
    population: usize,
    generations: u64,
    seed: u64,
    tournament_size: usize,
    mutation_rate: f64,
    elitism: f64,
    print_every: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let initial: Vec<BitstringOrganism> = (0..population)
        .map(|_| BitstringOrganism::random(genome_length, &mut rng))
        .collect();

    let config = WorldConfig {
        population_size: population,
        max_generations: generations,
        seed,
    };
    let reproducer = BitstringReproducer {
        tournament_size,
        mutation_rate,
        elitism_ratio: elitism,
    };

    let mut world = World::new(config, OneMaxEnvironment, reproducer, initial);
    let optimum = genome_length as f64;

    print_header();
    loop {
        if world.generation >= generations {
            break;
        }
        let snap = world.step(&mut rng);
        print_stats(&snap.stats, print_every);
        if snap.best_fitness >= optimum {
            println!(
                "\nOptimum reached at generation {}! (fitness = {})",
                snap.generation, snap.best_fitness
            );
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_neural(
    population: usize,
    generations: u64,
    seed: u64,
    grid_width: usize,
    grid_height: usize,
    food: usize,
    poison: usize,
    steps: usize,
    tournament_size: usize,
    mutation_std: f32,
    mutation_rate: f64,
    elitism: f64,
    print_every: u64,
) {
    let num_weights = Brain::weight_count(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let initial: Vec<Creature> = (0..population)
        .map(|_| Creature::random(num_weights, &mut rng))
        .collect();

    let config = WorldConfig {
        population_size: population,
        max_generations: generations,
        seed,
    };
    let grid_config = GridConfig {
        width: grid_width,
        height: grid_height,
        food_count: food,
        poison_count: poison,
        steps_per_eval: steps,
        grid_seed: seed.wrapping_add(1000),
        ..GridConfig::default()
    };
    let env = GridEnvironment::new(grid_config);
    let reproducer = CreatureReproducer {
        tournament_size,
        mutation_std,
        mutation_rate,
        elitism_ratio: elitism,
    };

    let mut world = World::new(config, env, reproducer, initial);

    print_header();
    loop {
        if world.generation >= generations {
            println!("\nSimulation complete after {generations} generations.");
            break;
        }
        let snap = world.step(&mut rng);
        print_stats(&snap.stats, print_every);
    }
}
