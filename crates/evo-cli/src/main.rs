use clap::Parser;
use crossbeam_channel::bounded;
use evo_core::{Snapshot, World, WorldConfig};
use evo_models::bitstring::{BitstringOrganism, BitstringReproducer, OneMaxEnvironment};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Parser)]
#[command(name = "evo-cli", about = "Headless evolution simulator")]
struct Cli {
    /// Length of the bitstring genome
    #[arg(long, default_value_t = 100)]
    genome_length: usize,

    /// Population size
    #[arg(long, default_value_t = 500)]
    population: usize,

    /// Maximum number of generations
    #[arg(long, default_value_t = 500)]
    generations: u64,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Tournament size for selection
    #[arg(long, default_value_t = 3)]
    tournament_size: usize,

    /// Per-bit mutation rate
    #[arg(long, default_value_t = 0.01)]
    mutation_rate: f64,

    /// Fraction of elites preserved each generation
    #[arg(long, default_value_t = 0.05)]
    elitism: f64,

    /// Print stats every N generations (0 = every generation)
    #[arg(long, default_value_t = 1)]
    print_every: u64,
}

fn main() {
    let cli = Cli::parse();

    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);

    let initial: Vec<BitstringOrganism> = (0..cli.population)
        .map(|_| BitstringOrganism::random(cli.genome_length, &mut rng))
        .collect();

    let config = WorldConfig {
        population_size: cli.population,
        max_generations: cli.generations,
        seed: cli.seed,
    };

    let env = OneMaxEnvironment;
    let reproducer = BitstringReproducer {
        tournament_size: cli.tournament_size,
        mutation_rate: cli.mutation_rate,
        elitism_ratio: cli.elitism,
    };

    let mut world = World::new(config, env, reproducer, initial);

    // Channel for snapshots — buffer a few so the engine doesn't block.
    let (tx, rx) = bounded::<Snapshot<BitstringOrganism>>(8);

    // Print header
    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>5}",
        "gen", "min", "mean", "max", "stddev", "pop"
    );
    println!("{}", "-".repeat(54));

    // Run in the main thread, consuming snapshots inline.
    // For M2+ we'll move the engine to a separate thread.
    let optimum = cli.genome_length as f64;

    loop {
        if world.generation >= cli.generations {
            break;
        }

        let snapshot = world.step(&mut rng);
        let _ = tx.try_send(snapshot);

        if let Ok(snap) = rx.try_recv() {
            let s = &snap.stats;
            if cli.print_every == 0 || s.generation % cli.print_every == 0 {
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

            // Early termination if optimum reached
            if snap.best_fitness >= optimum {
                println!(
                    "\nOptimum reached at generation {}! (fitness = {})",
                    snap.generation, snap.best_fitness
                );
                break;
            }
        }
    }
}
