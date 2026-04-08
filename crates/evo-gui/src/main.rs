use crossbeam_channel::{Receiver, bounded};
use eframe::egui;
use evo_core::{GenerationStats, RunControl, Snapshot, World, WorldConfig};
use evo_models::bitstring::{BitstringOrganism, BitstringReproducer, OneMaxEnvironment};
use evo_models::neural::{
    CellKind, Creature, CreatureReproducer, GridConfig, GridEnvironment, GridState,
};
use evo_models::neural::brain::Brain;
use evo_models::neural::grid::{HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::thread::JoinHandle;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Evolution Simulator",
        options,
        Box::new(|_cc| Ok(Box::new(EvolutionApp::new()))),
    )
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    Bitstring,
    NeuralCreature,
}

impl ModelKind {
    const ALL: [ModelKind; 2] = [ModelKind::Bitstring, ModelKind::NeuralCreature];

    fn label(self) -> &'static str {
        match self {
            ModelKind::Bitstring => "Bitstring / OneMax",
            ModelKind::NeuralCreature => "Neural Creatures",
        }
    }
}

// ---------------------------------------------------------------------------
// Type-erased snapshot data the GUI can consume
// ---------------------------------------------------------------------------

struct SnapshotData {
    generation: u64,
    stats: GenerationStats,
    best_fitness: f64,
    fitness: Vec<f64>,
    /// Model-specific data.
    model_data: ModelSnapshot,
}

enum ModelSnapshot {
    Bitstring { _best_bits: Vec<bool> },
    Neural { best_weights: Vec<f32> },
}

// ---------------------------------------------------------------------------
// Type-erased simulation handle
// ---------------------------------------------------------------------------

struct SimHandle {
    control: Arc<RunControl>,
    /// We receive snapshots as type-erased SnapshotData.
    rx: Receiver<SnapshotData>,
    thread: Option<JoinHandle<()>>,
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------

struct BitstringParams {
    genome_length: usize,
    mutation_rate: f64,
}

impl Default for BitstringParams {
    fn default() -> Self {
        BitstringParams {
            genome_length: 100,
            mutation_rate: 0.01,
        }
    }
}

struct NeuralParams {
    grid_width: usize,
    grid_height: usize,
    food_count: usize,
    poison_count: usize,
    steps_per_eval: usize,
    mutation_std: f32,
    mutation_rate: f64,
}

impl Default for NeuralParams {
    fn default() -> Self {
        NeuralParams {
            grid_width: 30,
            grid_height: 30,
            food_count: 40,
            poison_count: 15,
            steps_per_eval: 100,
            mutation_std: 0.3,
            mutation_rate: 0.1,
        }
    }
}

struct CommonParams {
    population_size: usize,
    max_generations: u64,
    seed: u64,
    tournament_size: usize,
    elitism_ratio: f64,
}

impl Default for CommonParams {
    fn default() -> Self {
        CommonParams {
            population_size: 500,
            max_generations: 1000,
            seed: 42,
            tournament_size: 3,
            elitism_ratio: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct EvolutionApp {
    model: ModelKind,
    common: CommonParams,
    bitstring: BitstringParams,
    neural: NeuralParams,

    handle: Option<SimHandle>,
    paused: bool,
    delay_ms: u64,

    // History
    history: Vec<GenerationStats>,
    latest_fitness: Vec<f64>,
    latest_best_fitness: f64,
    latest_generation: u64,

    // Model-specific latest data
    latest_model: Option<ModelSnapshot>,

    // Neural creature grid visualization
    grid_state: Option<GridState>,
    /// Which step of the trace to show (animated).
    trace_step: usize,
    /// Auto-advance animation.
    animate_trace: bool,

    // Bitstring optimum
    optimum_reached: Option<u64>,
}

impl EvolutionApp {
    fn new() -> Self {
        EvolutionApp {
            model: ModelKind::Bitstring,
            common: CommonParams::default(),
            bitstring: BitstringParams::default(),
            neural: NeuralParams::default(),
            handle: None,
            paused: false,
            delay_ms: 0,
            history: Vec::new(),
            latest_fitness: Vec::new(),
            latest_best_fitness: 0.0,
            latest_generation: 0,
            latest_model: None,
            grid_state: None,
            trace_step: 0,
            animate_trace: true,
            optimum_reached: None,
        }
    }

    fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    fn reset_state(&mut self) {
        self.history.clear();
        self.latest_fitness.clear();
        self.latest_best_fitness = 0.0;
        self.latest_generation = 0;
        self.latest_model = None;
        self.grid_state = None;
        self.trace_step = 0;
        self.optimum_reached = None;
        self.paused = false;
    }

    fn start_bitstring(&mut self) {
        self.reset_state();
        let pop_size = self.common.population_size;
        let max_gen = self.common.max_generations;
        let seed = self.common.seed;
        let genome_length = self.bitstring.genome_length;
        let reproducer = BitstringReproducer {
            tournament_size: self.common.tournament_size,
            mutation_rate: self.bitstring.mutation_rate,
            elitism_ratio: self.common.elitism_ratio,
        };

        let (tx, rx) = bounded::<SnapshotData>(16);
        let control = Arc::new(RunControl::new());
        control.set_delay_ms(self.delay_ms);

        let ctrl = Arc::clone(&control);
        let thread = std::thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let initial: Vec<BitstringOrganism> = (0..pop_size)
                .map(|_| BitstringOrganism::random(genome_length, &mut rng))
                .collect();
            let config = WorldConfig {
                population_size: pop_size,
                max_generations: max_gen,
                seed,
            };
            let mut world = World::new(config, OneMaxEnvironment, reproducer, initial);

            let (snap_tx, snap_rx) = bounded::<Snapshot<BitstringOrganism>>(16);
            // Run engine, forwarding typed snapshots as erased ones
            let ctrl2 = Arc::clone(&ctrl);
            let engine_thread = std::thread::spawn(move || {
                world.run(&mut rng, Some(&snap_tx), Some(&ctrl2));
            });

            while let Ok(snap) = snap_rx.recv() {
                let data = SnapshotData {
                    generation: snap.generation,
                    stats: snap.stats,
                    best_fitness: snap.best_fitness,
                    fitness: snap.fitness,
                    model_data: ModelSnapshot::Bitstring {
                        _best_bits: snap.best.bits.clone(),
                    },
                };
                if tx.try_send(data).is_err() {
                    // GUI not keeping up, drop frame
                }
                if ctrl.is_stopped() {
                    break;
                }
            }
            let _ = engine_thread.join();
        });

        self.handle = Some(SimHandle {
            control,
            rx,
            thread: Some(thread),
        });
    }

    fn start_neural(&mut self) {
        self.reset_state();
        let pop_size = self.common.population_size;
        let max_gen = self.common.max_generations;
        let seed = self.common.seed;
        let grid_config = GridConfig {
            width: self.neural.grid_width,
            height: self.neural.grid_height,
            food_count: self.neural.food_count,
            poison_count: self.neural.poison_count,
            steps_per_eval: self.neural.steps_per_eval,
            grid_seed: seed.wrapping_add(1000),
            ..GridConfig::default()
        };
        let reproducer = CreatureReproducer {
            tournament_size: self.common.tournament_size,
            mutation_std: self.neural.mutation_std,
            mutation_rate: self.neural.mutation_rate,
            elitism_ratio: self.common.elitism_ratio,
        };

        let (tx, rx) = bounded::<SnapshotData>(16);
        let control = Arc::new(RunControl::new());
        control.set_delay_ms(self.delay_ms);

        let ctrl = Arc::clone(&control);
        let thread = std::thread::spawn(move || {
            let num_weights = Brain::weight_count(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let initial: Vec<Creature> = (0..pop_size)
                .map(|_| Creature::random(num_weights, &mut rng))
                .collect();
            let config = WorldConfig {
                population_size: pop_size,
                max_generations: max_gen,
                seed,
            };
            let env = GridEnvironment::new(grid_config);
            let mut world = World::new(config, env, reproducer, initial);

            let (snap_tx, snap_rx) = bounded::<Snapshot<Creature>>(16);
            let ctrl2 = Arc::clone(&ctrl);
            let engine_thread = std::thread::spawn(move || {
                world.run(&mut rng, Some(&snap_tx), Some(&ctrl2));
            });

            while let Ok(snap) = snap_rx.recv() {
                let data = SnapshotData {
                    generation: snap.generation,
                    stats: snap.stats,
                    best_fitness: snap.best_fitness,
                    fitness: snap.fitness,
                    model_data: ModelSnapshot::Neural {
                        best_weights: snap.best.weights.clone(),
                    },
                };
                if tx.try_send(data).is_err() {
                    // drop frame
                }
                if ctrl.is_stopped() {
                    break;
                }
            }
            let _ = engine_thread.join();
        });

        self.handle = Some(SimHandle {
            control,
            rx,
            thread: Some(thread),
        });
    }

    fn start(&mut self) {
        match self.model {
            ModelKind::Bitstring => self.start_bitstring(),
            ModelKind::NeuralCreature => self.start_neural(),
        }
    }

    fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.control.request_stop();
            if let Some(thread) = handle.thread {
                let _ = thread.join();
            }
        }
        self.paused = false;
    }

    fn drain_snapshots(&mut self) {
        if let Some(handle) = &self.handle {
            while let Ok(snap) = handle.rx.try_recv() {
                self.latest_generation = snap.generation;
                self.latest_best_fitness = snap.best_fitness;
                self.latest_fitness = snap.fitness;
                self.history.push(snap.stats);

                // Check bitstring optimum
                if self.model == ModelKind::Bitstring && self.optimum_reached.is_none() {
                    let optimum = self.bitstring.genome_length as f64;
                    if snap.best_fitness >= optimum {
                        self.optimum_reached = Some(snap.generation);
                    }
                }

                // Update grid state for neural model
                if let ModelSnapshot::Neural { ref best_weights } = snap.model_data {
                    let grid_config = GridConfig {
                        width: self.neural.grid_width,
                        height: self.neural.grid_height,
                        food_count: self.neural.food_count,
                        poison_count: self.neural.poison_count,
                        steps_per_eval: self.neural.steps_per_eval,
                        grid_seed: self.common.seed.wrapping_add(1000)
                            .wrapping_add((snap.generation / 50) * 50),
                        ..GridConfig::default()
                    };
                    let env = GridEnvironment::new(grid_config);
                    let creature = Creature {
                        weights: best_weights.clone(),
                    };
                    self.grid_state = Some(env.simulate_best(&creature));
                    self.trace_step = 0;
                }

                self.latest_model = Some(snap.model_data);
            }

            // Check if thread finished naturally
            let finished = self
                .handle
                .as_ref()
                .is_some_and(|h| {
                    !h.control.is_stopped()
                        && h.thread.as_ref().is_some_and(|t| t.is_finished())
                });
            if finished
                && let Some(h) = self.handle.take()
                && let Some(t) = h.thread
            {
                let _ = t.join();
            }
        }
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Evolution Simulator");
        ui.separator();

        let running = self.is_running();

        // Model selector (disabled while running)
        ui.add_enabled_ui(!running, |ui| {
            ui.label("Model");
            for kind in ModelKind::ALL {
                ui.radio_value(&mut self.model, kind, kind.label());
            }
        });

        ui.separator();

        // Common parameters
        ui.add_enabled_ui(!running, |ui| {
            ui.label("Common parameters");
            ui.horizontal(|ui| {
                ui.label("Population:");
                ui.add(
                    egui::DragValue::new(&mut self.common.population_size).range(10..=100000),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Max generations:");
                ui.add(
                    egui::DragValue::new(&mut self.common.max_generations).range(1..=1000000),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Seed:");
                ui.add(egui::DragValue::new(&mut self.common.seed));
            });
            ui.horizontal(|ui| {
                ui.label("Tournament size:");
                ui.add(
                    egui::DragValue::new(&mut self.common.tournament_size).range(2..=20),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Elitism ratio:");
                ui.add(
                    egui::DragValue::new(&mut self.common.elitism_ratio)
                        .speed(0.01)
                        .range(0.0..=1.0),
                );
            });
        });

        ui.separator();

        // Model-specific parameters
        ui.add_enabled_ui(!running, |ui| {
            match self.model {
                ModelKind::Bitstring => {
                    ui.label("Bitstring parameters");
                    ui.horizontal(|ui| {
                        ui.label("Genome length:");
                        ui.add(
                            egui::DragValue::new(&mut self.bitstring.genome_length)
                                .range(8..=10000),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mutation rate:");
                        ui.add(
                            egui::DragValue::new(&mut self.bitstring.mutation_rate)
                                .speed(0.001)
                                .range(0.0..=1.0),
                        );
                    });
                }
                ModelKind::NeuralCreature => {
                    ui.label("Neural creature parameters");
                    ui.horizontal(|ui| {
                        ui.label("Grid size:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.grid_width).range(10..=100),
                        );
                        ui.label("x");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.grid_height).range(10..=100),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Food count:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.food_count).range(1..=500),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Poison count:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.poison_count).range(0..=500),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Steps/eval:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.steps_per_eval)
                                .range(10..=1000),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mutation std:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.mutation_std)
                                .speed(0.01)
                                .range(0.0..=2.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mutation rate:");
                        ui.add(
                            egui::DragValue::new(&mut self.neural.mutation_rate)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                    });
                }
            }
        });

        ui.separator();

        // Speed control
        ui.label("Speed control");
        ui.horizontal(|ui| {
            ui.label("Delay (ms):");
            let resp =
                ui.add(egui::Slider::new(&mut self.delay_ms, 0..=200).logarithmic(true));
            if resp.changed()
                && let Some(ref handle) = self.handle
            {
                handle.control.set_delay_ms(self.delay_ms);
            }
        });

        ui.separator();

        // Control buttons
        ui.horizontal(|ui| {
            if !running {
                if ui.button("Start").clicked() {
                    self.start();
                }
            } else {
                let pause_label = if self.paused { "Resume" } else { "Pause" };
                if ui.button(pause_label).clicked() {
                    self.paused = !self.paused;
                    if let Some(ref handle) = self.handle {
                        handle.control.set_paused(self.paused);
                    }
                    if !self.paused {
                        ctx.request_repaint();
                    }
                }
                if ui.button("Stop").clicked() {
                    self.stop();
                }
            }
        });

        ui.separator();

        // Status
        ui.label(format!("Generation: {}", self.latest_generation));
        ui.label(format!("Best fitness: {:.2}", self.latest_best_fitness));

        if self.model == ModelKind::Bitstring {
            ui.label(format!(
                "Optimum: {:.0}",
                self.bitstring.genome_length as f64
            ));
            if let Some(at_gen) = self.optimum_reached {
                ui.colored_label(
                    egui::Color32::GREEN,
                    format!("Optimum reached at gen {at_gen}!"),
                );
            }
        }

        if !running && !self.history.is_empty() {
            ui.colored_label(egui::Color32::YELLOW, "Simulation finished");
        }
    }

    fn draw_fitness_plot(&self, ui: &mut egui::Ui, height: f32) {
        ui.label("Fitness over generations");
        let fitness_plot = egui_plot::Plot::new("fitness_plot")
            .height(height)
            .x_axis_label("Generation")
            .y_axis_label("Fitness")
            .legend(egui_plot::Legend::default());

        fitness_plot.show(ui, |plot_ui| {
            if !self.history.is_empty() {
                let min_pts: Vec<[f64; 2]> = self
                    .history
                    .iter()
                    .map(|s| [s.generation as f64, s.min_fitness])
                    .collect();
                let mean_pts: Vec<[f64; 2]> = self
                    .history
                    .iter()
                    .map(|s| [s.generation as f64, s.mean_fitness])
                    .collect();
                let max_pts: Vec<[f64; 2]> = self
                    .history
                    .iter()
                    .map(|s| [s.generation as f64, s.max_fitness])
                    .collect();

                plot_ui.line(
                    egui_plot::Line::new(egui_plot::PlotPoints::new(min_pts))
                        .name("Min")
                        .color(egui::Color32::from_rgb(230, 100, 100)),
                );
                plot_ui.line(
                    egui_plot::Line::new(egui_plot::PlotPoints::new(mean_pts))
                        .name("Mean")
                        .color(egui::Color32::from_rgb(100, 180, 230)),
                );
                plot_ui.line(
                    egui_plot::Line::new(egui_plot::PlotPoints::new(max_pts))
                        .name("Max")
                        .color(egui::Color32::from_rgb(100, 230, 100)),
                );
            }
        });
    }

    fn draw_histogram(&self, ui: &mut egui::Ui, height: f32) {
        ui.label("Population fitness distribution");
        let hist_plot = egui_plot::Plot::new("hist_plot")
            .height(height)
            .x_axis_label("Fitness")
            .y_axis_label("Count")
            .legend(egui_plot::Legend::default());

        hist_plot.show(ui, |plot_ui| {
            if !self.latest_fitness.is_empty() {
                let min_f = self
                    .latest_fitness
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let max_f = self
                    .latest_fitness
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = (max_f - min_f).max(1.0);
                let num_bins: usize = 40;
                let bin_width = range / num_bins as f64;

                let mut bins = vec![0u32; num_bins + 1];
                for &f in &self.latest_fitness {
                    let idx = ((f - min_f) / range * num_bins as f64) as usize;
                    let idx = idx.min(num_bins);
                    bins[idx] += 1;
                }

                let bars: Vec<egui_plot::Bar> = bins
                    .iter()
                    .enumerate()
                    .filter(|(_, count)| **count > 0)
                    .map(|(i, count)| {
                        let count = *count;
                        let x = min_f + i as f64 * bin_width + bin_width / 2.0;
                        egui_plot::Bar::new(x, count as f64).width(bin_width * 0.9)
                    })
                    .collect();

                plot_ui.bar_chart(
                    egui_plot::BarChart::new(bars)
                        .name("Fitness distribution")
                        .color(egui::Color32::from_rgb(140, 180, 255)),
                );
            }
        });
    }

    fn draw_grid(&mut self, ui: &mut egui::Ui, size: f32) {
        let Some(ref grid) = self.grid_state else {
            ui.label("No grid data yet. Start a simulation.");
            return;
        };

        let cell_size = (size / grid.width.max(grid.height) as f32).max(2.0);
        let desired = egui::vec2(
            grid.width as f32 * cell_size,
            grid.height as f32 * cell_size,
        );
        let (response, painter) =
            ui.allocate_painter(desired, egui::Sense::hover());
        let origin = response.rect.left_top();

        // Draw cells
        for y in 0..grid.height {
            for x in 0..grid.width {
                let color = match grid.cell_at(x, y) {
                    CellKind::Empty => continue,
                    CellKind::Food => egui::Color32::from_rgb(80, 200, 80),
                    CellKind::Poison => egui::Color32::from_rgb(200, 60, 200),
                    CellKind::Wall => egui::Color32::from_rgb(100, 100, 100),
                };
                let rect = egui::Rect::from_min_size(
                    origin + egui::vec2(x as f32 * cell_size, y as f32 * cell_size),
                    egui::vec2(cell_size, cell_size),
                );
                painter.rect_filled(rect, 0.0, color);
            }
        }

        // Draw creature trace
        if let Some(trace) = grid.traces.first() {
            let step = self.trace_step.min(trace.positions.len().saturating_sub(1));

            // Draw trail (fading)
            let trail_len = 20;
            let start = step.saturating_sub(trail_len);
            for i in start..step {
                let (tx, ty) = trace.positions[i];
                let alpha = ((i - start) as f32 / trail_len as f32 * 150.0) as u8;
                let rect = egui::Rect::from_min_size(
                    origin + egui::vec2(tx as f32 * cell_size, ty as f32 * cell_size),
                    egui::vec2(cell_size, cell_size),
                );
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgba_unmultiplied(255, 200, 50, alpha));
            }

            // Draw creature current position
            if let Some(&(cx, cy)) = trace.positions.get(step) {
                let rect = egui::Rect::from_min_size(
                    origin + egui::vec2(cx as f32 * cell_size, cy as f32 * cell_size),
                    egui::vec2(cell_size, cell_size),
                );
                painter.rect_filled(rect, cell_size / 4.0, egui::Color32::from_rgb(255, 220, 50));
            }

            // Animate
            if self.animate_trace && step < trace.positions.len().saturating_sub(1) {
                self.trace_step += 1;
            }
        }

        // Controls
        ui.horizontal(|ui| {
            if ui.button("Replay").clicked() {
                self.trace_step = 0;
            }
            ui.checkbox(&mut self.animate_trace, "Animate");
            if let Some(trace) = grid.traces.first() {
                ui.label(format!(
                    "Step: {} / {}",
                    self.trace_step.min(trace.positions.len().saturating_sub(1)),
                    trace.positions.len().saturating_sub(1)
                ));
            }
        });
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_snapshots();

        // Request repaint while running or animating
        let needs_repaint = (self.is_running() && !self.paused)
            || (self.animate_trace && self.grid_state.is_some());
        if needs_repaint {
            ctx.request_repaint();
        }

        // --- Left side panel: controls ---
        egui::SidePanel::left("controls")
            .min_width(250.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_controls(ui, ctx);
                });
            });

        // --- Central panel ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();

            match self.model {
                ModelKind::Bitstring => {
                    let plot_height = (available.y - 30.0) / 2.0;
                    self.draw_fitness_plot(ui, plot_height);
                    ui.separator();
                    self.draw_histogram(ui, plot_height);
                }
                ModelKind::NeuralCreature => {
                    // Top: fitness plot (1/3 height)
                    let plot_height = (available.y - 40.0) / 3.0;
                    self.draw_fitness_plot(ui, plot_height);
                    ui.separator();

                    // Bottom: grid view (2/3 height)
                    ui.label("Best creature behavior");
                    let grid_size = available.y - plot_height - 80.0;
                    self.draw_grid(ui, grid_size);
                }
            }
        });
    }
}
