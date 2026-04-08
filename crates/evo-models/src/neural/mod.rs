//! Neural creature model.
//!
//! Creatures with simple feedforward neural-net brains navigate a 2D grid,
//! seeking food and avoiding poison. Genome encodes network weights.

pub mod brain;
pub mod creature;
pub mod grid;
pub mod reproducer;

pub use creature::Creature;
pub use grid::{GridEnvironment, GridConfig, GridState, CellKind};
pub use reproducer::CreatureReproducer;
