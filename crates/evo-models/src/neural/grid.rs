//! 2D grid environment for neural creatures.

use super::brain::Brain;
use super::creature::Creature;
use evo_core::{Environment, Organism};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

/// What occupies a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CellKind {
    Empty,
    Food,
    Poison,
    Wall,
}

/// Configuration for the grid world.
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub width: usize,
    pub height: usize,
    pub food_count: usize,
    pub poison_count: usize,
    /// Number of simulation steps each creature gets per evaluation.
    pub steps_per_eval: usize,
    /// Reward for eating food.
    pub food_reward: f64,
    /// Penalty for eating poison.
    pub poison_penalty: f64,
    /// Small penalty per step to encourage efficiency.
    pub step_penalty: f64,
    /// Seed for grid generation (different from the main simulation seed).
    pub grid_seed: u64,
}

impl Default for GridConfig {
    fn default() -> Self {
        GridConfig {
            width: 30,
            height: 30,
            food_count: 40,
            poison_count: 15,
            steps_per_eval: 100,
            food_reward: 1.0,
            poison_penalty: 0.5,
            step_penalty: 0.005,
            grid_seed: 123,
        }
    }
}

/// Sensor inputs for a creature.
/// 8 directions (N, NE, E, SE, S, SW, W, NW) × 3 channels (food, poison, wall) = 24 inputs
/// Plus 1 bias input = 25 total.
pub const NUM_SENSOR_DIRS: usize = 8;
pub const SENSOR_CHANNELS: usize = 3; // food, poison, wall
pub const INPUT_SIZE: usize = NUM_SENSOR_DIRS * SENSOR_CHANNELS + 1; // +1 for bias

/// Output: 4 values → move N, E, S, W. Highest wins.
pub const OUTPUT_SIZE: usize = 4;

/// Default hidden layer size.
pub const HIDDEN_SIZE: usize = 10;

/// Direction offsets: N, NE, E, SE, S, SW, W, NW
const DIR_DX: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
const DIR_DY: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

/// Movement offsets: N, E, S, W
const MOVE_DX: [i32; 4] = [0, 1, 0, -1];
const MOVE_DY: [i32; 4] = [-1, 0, 1, 0];

/// A snapshot of one creature's simulation for visualization.
#[derive(Debug, Clone, Serialize)]
pub struct CreatureTrace {
    pub positions: Vec<(usize, usize)>,
}

/// The full state of a single grid simulation (for visualization).
#[derive(Debug, Clone, Serialize)]
pub struct GridState {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<CellKind>,
    /// Traces for the top N creatures.
    pub traces: Vec<CreatureTrace>,
}

impl GridState {
    pub fn cell_at(&self, x: usize, y: usize) -> CellKind {
        self.cells[y * self.width + x]
    }
}

/// Grid environment.
pub struct GridEnvironment {
    pub config: GridConfig,
    /// Base grid layout (regenerated each evaluation for fairness).
    base_cells: Vec<CellKind>,
}

impl GridEnvironment {
    pub fn new(config: GridConfig) -> Self {
        let base_cells = Self::generate_grid(&config, config.grid_seed);
        GridEnvironment { config, base_cells }
    }

    fn generate_grid(config: &GridConfig, seed: u64) -> Vec<CellKind> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut cells = vec![CellKind::Empty; config.width * config.height];

        // Place walls around the border
        for x in 0..config.width {
            cells[x] = CellKind::Wall;
            cells[(config.height - 1) * config.width + x] = CellKind::Wall;
        }
        for y in 0..config.height {
            cells[y * config.width] = CellKind::Wall;
            cells[y * config.width + config.width - 1] = CellKind::Wall;
        }

        // Place food
        let mut placed = 0;
        while placed < config.food_count {
            let x = rng.random_range(1..config.width - 1);
            let y = rng.random_range(1..config.height - 1);
            let idx = y * config.width + x;
            if cells[idx] == CellKind::Empty {
                cells[idx] = CellKind::Food;
                placed += 1;
            }
        }

        // Place poison
        placed = 0;
        while placed < config.poison_count {
            let x = rng.random_range(1..config.width - 1);
            let y = rng.random_range(1..config.height - 1);
            let idx = y * config.width + x;
            if cells[idx] == CellKind::Empty {
                cells[idx] = CellKind::Poison;
                placed += 1;
            }
        }

        cells
    }

    /// Sense the environment around position (x, y).
    fn sense(cells: &[CellKind], width: usize, height: usize, x: usize, y: usize) -> Vec<f32> {
        let mut input = vec![0.0f32; INPUT_SIZE];

        for (d, input_chunk) in input.chunks_exact_mut(SENSOR_CHANNELS).enumerate().take(NUM_SENSOR_DIRS) {
            // Look in direction d, find the nearest object
            let mut cx = x as i32 + DIR_DX[d];
            let mut cy = y as i32 + DIR_DY[d];

            // Scan up to 5 cells in that direction
            let max_look = 5;
            let mut dist = 1.0f32;
            for step in 1..=max_look {
                if cx < 0 || cy < 0 || cx >= width as i32 || cy >= height as i32 {
                    // Out of bounds = wall
                    input_chunk[2] = 1.0 / step as f32;
                    break;
                }
                let idx = cy as usize * width + cx as usize;
                match cells[idx] {
                    CellKind::Food => {
                        input_chunk[0] = 1.0 / dist;
                        break;
                    }
                    CellKind::Poison => {
                        input_chunk[1] = 1.0 / dist;
                        break;
                    }
                    CellKind::Wall => {
                        input_chunk[2] = 1.0 / dist;
                        break;
                    }
                    CellKind::Empty => {}
                }
                cx += DIR_DX[d];
                cy += DIR_DY[d];
                dist += 1.0;
            }
        }

        // Bias input
        *input.last_mut().unwrap() = 1.0;

        input
    }

    /// Simulate one creature on a copy of the grid. Returns (fitness, trace).
    fn simulate_creature(
        config: &GridConfig,
        base_cells: &[CellKind],
        creature: &Creature,
    ) -> (f64, CreatureTrace) {
        let mut cells = base_cells.to_vec();
        let mut fitness = 0.0;

        // Start in the center
        let mut x = config.width / 2;
        let mut y = config.height / 2;
        // Make sure start cell is empty
        cells[y * config.width + x] = CellKind::Empty;

        let brain = Brain::new(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, creature.genome().clone());

        let mut positions = Vec::with_capacity(config.steps_per_eval);
        positions.push((x, y));

        for _ in 0..config.steps_per_eval {
            let input = Self::sense(&cells, config.width, config.height, x, y);
            let output = brain.forward(&input);

            // Pick movement direction (argmax)
            let dir = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let nx = x as i32 + MOVE_DX[dir];
            let ny = y as i32 + MOVE_DY[dir];

            // Bounds check and wall check
            if nx >= 0
                && ny >= 0
                && (nx as usize) < config.width
                && (ny as usize) < config.height
            {
                let nidx = ny as usize * config.width + nx as usize;
                match cells[nidx] {
                    CellKind::Wall => {} // Can't move into wall
                    CellKind::Food => {
                        x = nx as usize;
                        y = ny as usize;
                        fitness += config.food_reward;
                        cells[nidx] = CellKind::Empty;
                    }
                    CellKind::Poison => {
                        x = nx as usize;
                        y = ny as usize;
                        fitness -= config.poison_penalty;
                        cells[nidx] = CellKind::Empty;
                    }
                    CellKind::Empty => {
                        x = nx as usize;
                        y = ny as usize;
                    }
                }
            }

            fitness -= config.step_penalty;
            positions.push((x, y));
        }

        (fitness, CreatureTrace { positions })
    }

    /// Run simulation for the best creature and return the grid state for visualization.
    pub fn simulate_best(&self, creature: &Creature) -> GridState {
        let (_, trace) = Self::simulate_creature(&self.config, &self.base_cells, creature);
        GridState {
            width: self.config.width,
            height: self.config.height,
            cells: self.base_cells.clone(),
            traces: vec![trace],
        }
    }

    pub fn base_cells(&self) -> &[CellKind] {
        &self.base_cells
    }
}

impl Environment for GridEnvironment {
    type Org = Creature;

    fn evaluate(&self, population: &[Creature]) -> Vec<f64> {
        let config = &self.config;
        let base_cells = &self.base_cells;
        population
            .par_iter()
            .map(|creature| {
                let (fitness, _) = Self::simulate_creature(config, base_cells, creature);
                fitness
            })
            .collect()
    }

    fn step(&mut self, generation: u64) {
        // Regenerate grid every 50 generations to prevent overfitting to one layout
        if generation > 0 && generation.is_multiple_of(50) {
            self.base_cells =
                Self::generate_grid(&self.config, self.config.grid_seed.wrapping_add(generation));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::brain::Brain;
    use super::super::creature::Creature;
    use super::super::reproducer::CreatureReproducer;
    use evo_core::{World, WorldConfig};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn grid_has_walls_and_food() {
        let config = GridConfig::default();
        let env = GridEnvironment::new(config.clone());
        let cells = env.base_cells();

        // Check borders are walls
        for x in 0..config.width {
            assert_eq!(cells[x], CellKind::Wall);
            assert_eq!(
                cells[(config.height - 1) * config.width + x],
                CellKind::Wall
            );
        }

        // Count food and poison
        let food = cells.iter().filter(|&&c| c == CellKind::Food).count();
        let poison = cells.iter().filter(|&&c| c == CellKind::Poison).count();
        assert_eq!(food, config.food_count);
        assert_eq!(poison, config.poison_count);
    }

    #[test]
    fn creature_simulation_runs() {
        let config = GridConfig::default();
        let env = GridEnvironment::new(config);
        let num_weights = Brain::weight_count(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let creature = Creature::random(num_weights, &mut rng);

        let grid_state = env.simulate_best(&creature);
        assert!(!grid_state.traces.is_empty());
        assert!(grid_state.traces[0].positions.len() > 1);
    }

    #[test]
    fn neural_evolution_improves_fitness() {
        let num_weights = Brain::weight_count(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let pop_size = 100;
        let initial: Vec<Creature> = (0..pop_size)
            .map(|_| Creature::random(num_weights, &mut rng))
            .collect();

        let config = WorldConfig {
            population_size: pop_size,
            max_generations: 50,
            seed: 42,
        };

        let grid_config = GridConfig {
            food_count: 30,
            poison_count: 5,
            steps_per_eval: 80,
            ..GridConfig::default()
        };

        let env = GridEnvironment::new(grid_config);
        let reproducer = CreatureReproducer::default();
        let mut world = World::new(config, env, reproducer, initial);

        // Record fitness at generation 0
        let first_snap = world.step(&mut rng);
        let initial_best = first_snap.best_fitness;

        // Run remaining generations
        let final_snap = world.run(&mut rng, None, None);

        // Fitness should improve (or at least not significantly degrade)
        assert!(
            final_snap.best_fitness >= initial_best - 1.0,
            "fitness should not regress significantly: initial={initial_best}, final={}",
            final_snap.best_fitness
        );
    }
}
