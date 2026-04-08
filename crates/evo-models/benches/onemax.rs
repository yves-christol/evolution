use criterion::{Criterion, criterion_group, criterion_main};
use evo_core::{World, WorldConfig};
use evo_models::bitstring::{BitstringOrganism, BitstringReproducer, OneMaxEnvironment};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn setup(
    pop_size: usize,
    genome_len: usize,
    max_gen: u64,
) -> (
    World<BitstringOrganism, OneMaxEnvironment, BitstringReproducer>,
    ChaCha8Rng,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let initial: Vec<BitstringOrganism> = (0..pop_size)
        .map(|_| BitstringOrganism::random(genome_len, &mut rng))
        .collect();
    let config = WorldConfig {
        population_size: pop_size,
        max_generations: max_gen,
        seed: 42,
    };
    let world = World::new(config, OneMaxEnvironment, BitstringReproducer::default(), initial);
    (world, rng)
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("onemax_step");

    group.bench_function("pop500_genome100", |b| {
        let (mut world, mut rng) = setup(500, 100, 10000);
        b.iter(|| world.step(&mut rng));
    });

    group.bench_function("pop5000_genome256", |b| {
        let (mut world, mut rng) = setup(5000, 256, 10000);
        b.iter(|| world.step(&mut rng));
    });

    group.finish();
}

fn bench_full_run(c: &mut Criterion) {
    c.bench_function("onemax_100gen_pop1000_genome100", |b| {
        b.iter(|| {
            let (mut world, mut rng) = setup(1000, 100, 100);
            world.run(&mut rng, None, None)
        });
    });
}

criterion_group!(benches, bench_step, bench_full_run);
criterion_main!(benches);
