# Evolution Simulator — Project Plan

## Vision

A high-performance Darwinian evolution simulator that models populations of organisms across many generations, with pluggable life forms, environments, and optional real-time visualization. Runs locally on macOS.

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | **Rust** | Fearless concurrency, zero-cost abstractions, no GC pauses, trait system ideal for polymorphic models |
| Parallelism | **rayon** + **crossbeam** | Data-parallel iteration over populations (rayon), lock-free channels for engine↔viz communication (crossbeam) |
| Serialization | **serde** + **bincode**/**json** | Save/load simulation state; JSON for config, bincode for fast snapshots |
| RNG | **rand** + **rand_chacha** | Reproducible, seedable, per-thread RNG for deterministic parallel runs |
| Visualization | **egui** (via **eframe**) | Immediate-mode GUI, native macOS support, easy to iterate on, no heavy game-engine overhead |
| Plotting | **egui_plot** | Inline charts for population stats, fitness curves, gene distributions |
| CLI | **clap** | Headless mode with full config via command-line args |

---

## Crate Architecture

```
evolution/                  (workspace root)
├── Cargo.toml              (workspace definition)
├── crates/
│   ├── evo-core/           (library — the simulation engine)
│   ├── evo-models/         (library — built-in life forms & environments)
│   ├── evo-cli/            (binary — headless runner)
│   └── evo-gui/            (binary — visualization app)
├── configs/                (example TOML/JSON simulation configs)
├── PLAN.md
└── README.md
```

### `evo-core` — The Engine

The heart of the project. Zero dependencies on visualization. Defines the core trait interfaces and runs the simulation loop.

**Key traits:**

```rust
/// A single organism in the simulation.
trait Organism: Clone + Send + Sync {
    type Genome: Clone + Send + Sync;

    fn genome(&self) -> &Self::Genome;
    fn fitness(&self) -> f64;
    fn with_genome(genome: Self::Genome) -> Self;
}

/// Evaluates organisms in the context of an environment.
trait Environment: Send + Sync {
    type Org: Organism;

    /// Assign fitness scores to a population (may mutate organisms).
    fn evaluate(&self, population: &mut [Self::Org]);

    /// Optional: environment can change over time.
    fn step(&mut self, generation: u64) {}
}

/// Defines how reproduction and mutation work.
trait Reproducer: Send + Sync {
    type Org: Organism;

    fn select_and_reproduce(
        &self,
        population: &[Self::Org],
        target_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<Self::Org>;
}
```

**Simulation engine:**

- `World<O, E, R>` struct parameterized by Organism, Environment, Reproducer
- Generational loop: evaluate → select → reproduce → mutate → (optional) environment step
- Parallel fitness evaluation via `rayon::par_iter_mut`
- Per-generation statistics (min/max/mean fitness, diversity metrics)
- Event channel (`crossbeam::channel`) to stream snapshots to an observer (GUI or logger)
- Configurable: population size, mutation rate, elitism ratio, max generations, seed
- Checkpoint/resume via serde serialization

**Threading model:**

- The simulation runs on its own thread (or thread pool via rayon)
- An `mpsc` channel sends `GenerationSnapshot` events to consumers
- Consumers (GUI, CLI logger, file writer) are fully decoupled
- The engine never blocks on the consumer — it can drop frames if the consumer is slow

### `evo-models` — Built-in Models

Ships with a few example models to demonstrate the framework:

1. **Bitstring / OneMax** — classic GA benchmark (maximize 1s in a bitstring)
2. **Neural Creature** — 2D creatures with simple neural-net brains navigating a grid, seeking food, avoiding poison. Genome encodes network weights.
3. **Morphology** — organisms with evolvable body plans (limb count, size, color) competing for resources in a spatial environment.

Each model implements the core traits. Users can add new models by implementing the same traits in `evo-models` or in their own crate.

### `evo-cli` — Headless Runner

- Loads a simulation config (TOML file or CLI args)
- Runs the engine to completion
- Streams stats to stdout (generation, fitness, diversity)
- Optionally writes snapshots to disk
- Exit code reflects convergence status

### `evo-gui` — Visualization App

- Launches the engine in a background thread
- Receives `GenerationSnapshot` via channel
- Renders with egui/eframe:
  - **Control panel**: start/pause/step/reset, parameter sliders
  - **Fitness plot**: real-time line chart of min/avg/max fitness over generations
  - **Population view**: model-specific visualization (grid world, gene heatmap, etc.)
  - **Genome inspector**: click an organism to see its genome details
- Model-specific rendering via a `Visualizable` trait (optional, only needed for GUI)

---

## Milestone Plan

### M0 — Scaffolding
- Workspace setup, crate structure, CI (cargo check/test/clippy)
- Core trait definitions in `evo-core`
- Basic `World` struct with generational loop (single-threaded first)

### M1 — Engine + Bitstring Model
- Implement `Organism`, `Environment`, `Reproducer` for bitstring/OneMax
- Parallel fitness evaluation with rayon
- Generation snapshots via crossbeam channel
- CLI runner that prints stats per generation
- Reproducible runs via seeded RNG
- Unit tests + benchmarks

### M2 — GUI Shell
- eframe app with egui
- Engine runs in background thread, GUI consumes snapshots
- Control panel (start/pause/step/speed slider)
- Fitness-over-time plot with egui_plot
- Gene distribution histogram

### M3 — Neural Creature Model
- 2D grid environment
- Creatures with simple feedforward neural nets
- Genome = flattened weight vector
- Fitness = food collected over N timesteps
- GUI: animated grid view showing creatures moving

### M4 — Polish & Extensibility
- TOML config files for simulation parameters
- Checkpoint save/load (serde + bincode)
- Documentation + examples for adding custom models
- Performance profiling and optimization pass

---

## Key Design Decisions

| Decision | Choice | Alternative considered |
|----------|--------|----------------------|
| Monomorphization via generics | `World<O, E, R>` with trait bounds | `dyn Trait` objects — rejected for performance in hot loops |
| Channel-based decoupling | crossbeam mpsc | Shared `Arc<Mutex<>>` — rejected to avoid contention |
| egui for viz | Immediate-mode, native | Bevy (too heavy), macroquad (less UI), web-based (extra complexity) |
| Workspace with multiple crates | Clean separation of concerns | Single crate with feature flags — harder to enforce decoupling |
| Per-thread RNG | Deterministic parallelism | Global RNG — not thread-safe, not reproducible |
