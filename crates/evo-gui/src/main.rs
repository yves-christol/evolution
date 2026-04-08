use crossbeam_channel::{Receiver, bounded};
use eframe::egui;
use evo_core::{GenerationStats, RunControl, Snapshot, World, WorldConfig};
use evo_models::bitstring::{BitstringOrganism, BitstringReproducer, OneMaxEnvironment};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::thread::JoinHandle;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Evolution Simulator",
        options,
        Box::new(|_cc| Ok(Box::new(EvolutionApp::new()))),
    )
}

// ---------------------------------------------------------------------------
// Simulation parameters (editable in the GUI before starting)
// ---------------------------------------------------------------------------

struct SimParams {
    genome_length: usize,
    population_size: usize,
    max_generations: u64,
    seed: u64,
    tournament_size: usize,
    mutation_rate: f64,
    elitism_ratio: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            genome_length: 100,
            population_size: 500,
            max_generations: 1000,
            seed: 42,
            tournament_size: 3,
            mutation_rate: 0.01,
            elitism_ratio: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Running simulation handle
// ---------------------------------------------------------------------------

struct SimHandle {
    control: Arc<RunControl>,
    rx: Receiver<Snapshot<BitstringOrganism>>,
    thread: Option<JoinHandle<()>>,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct EvolutionApp {
    params: SimParams,
    handle: Option<SimHandle>,
    paused: bool,
    delay_ms: u64,

    // History for plotting
    history: Vec<GenerationStats>,

    // Latest snapshot for gene histogram
    latest_fitness: Vec<f64>,
    latest_best_fitness: f64,
    latest_generation: u64,
    latest_best_genome: Vec<bool>,

    // Reached optimum?
    optimum_reached: Option<u64>,
}

impl EvolutionApp {
    fn new() -> Self {
        EvolutionApp {
            params: SimParams::default(),
            handle: None,
            paused: false,
            delay_ms: 0,
            history: Vec::new(),
            latest_fitness: Vec::new(),
            latest_best_fitness: 0.0,
            latest_generation: 0,
            latest_best_genome: Vec::new(),
            optimum_reached: None,
        }
    }

    fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    fn start(&mut self) {
        // Reset state
        self.history.clear();
        self.latest_fitness.clear();
        self.latest_best_fitness = 0.0;
        self.latest_generation = 0;
        self.latest_best_genome.clear();
        self.optimum_reached = None;
        self.paused = false;

        let params = &self.params;
        let config = WorldConfig {
            population_size: params.population_size,
            max_generations: params.max_generations,
            seed: params.seed,
        };
        let reproducer = BitstringReproducer {
            tournament_size: params.tournament_size,
            mutation_rate: params.mutation_rate,
            elitism_ratio: params.elitism_ratio,
        };

        let genome_length = params.genome_length;
        let seed = params.seed;
        let (tx, rx) = bounded::<Snapshot<BitstringOrganism>>(16);
        let control = Arc::new(RunControl::new());
        control.set_delay_ms(self.delay_ms);

        let ctrl = Arc::clone(&control);
        let thread = std::thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let initial: Vec<BitstringOrganism> = (0..config.population_size)
                .map(|_| BitstringOrganism::random(genome_length, &mut rng))
                .collect();
            let mut world = World::new(config, OneMaxEnvironment, reproducer, initial);
            world.run(&mut rng, Some(&tx), Some(&ctrl));
        });

        self.handle = Some(SimHandle {
            control,
            rx,
            thread: Some(thread),
        });
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
            let optimum = self.params.genome_length as f64;
            while let Ok(snap) = handle.rx.try_recv() {
                self.latest_generation = snap.generation;
                self.latest_best_fitness = snap.best_fitness;
                self.latest_fitness = snap.fitness;
                self.latest_best_genome = snap.best.bits.clone();
                self.history.push(snap.stats);

                if self.optimum_reached.is_none() && snap.best_fitness >= optimum {
                    self.optimum_reached = Some(snap.generation);
                }
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
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_snapshots();

        // Request repaint while running so we keep draining snapshots
        if self.is_running() && !self.paused {
            ctx.request_repaint();
        }

        // --- Left side panel: controls ---
        egui::SidePanel::left("controls")
            .min_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Evolution Simulator");
                ui.separator();

                let running = self.is_running();

                // Parameters (disabled while running)
                ui.add_enabled_ui(!running, |ui| {
                    ui.label("Parameters");
                    ui.horizontal(|ui| {
                        ui.label("Genome length:");
                        ui.add(egui::DragValue::new(&mut self.params.genome_length).range(8..=10000));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Population:");
                        ui.add(egui::DragValue::new(&mut self.params.population_size).range(10..=100000));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max generations:");
                        ui.add(egui::DragValue::new(&mut self.params.max_generations).range(1..=1000000));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Seed:");
                        ui.add(egui::DragValue::new(&mut self.params.seed));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Tournament size:");
                        ui.add(egui::DragValue::new(&mut self.params.tournament_size).range(2..=20));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mutation rate:");
                        ui.add(egui::DragValue::new(&mut self.params.mutation_rate).speed(0.001).range(0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Elitism ratio:");
                        ui.add(egui::DragValue::new(&mut self.params.elitism_ratio).speed(0.01).range(0.0..=1.0));
                    });
                });

                ui.separator();

                // Speed control (always available)
                ui.label("Speed control");
                ui.horizontal(|ui| {
                    ui.label("Delay (ms):");
                    let resp = ui.add(egui::Slider::new(&mut self.delay_ms, 0..=200).logarithmic(true));
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
                ui.label(format!("Best fitness: {:.1}", self.latest_best_fitness));
                ui.label(format!(
                    "Optimum: {:.0}",
                    self.params.genome_length as f64
                ));

                if let Some(at_gen) = self.optimum_reached {
                    ui.colored_label(egui::Color32::GREEN, format!("Optimum reached at gen {at_gen}!"));
                }

                if !running && !self.history.is_empty() {
                    ui.colored_label(egui::Color32::YELLOW, "Simulation finished");
                }
            });

        // --- Central panel: plots ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // Top half: fitness over time
            let available = ui.available_size();
            let plot_height = (available.y - 20.0) / 2.0;

            ui.label("Fitness over generations");
            let fitness_plot = egui_plot::Plot::new("fitness_plot")
                .height(plot_height)
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

            ui.separator();

            // Bottom half: gene distribution histogram (count of 1-bits per organism)
            ui.label("Population fitness distribution");
            let hist_plot = egui_plot::Plot::new("hist_plot")
                .height(plot_height)
                .x_axis_label("Fitness (number of 1-bits)")
                .y_axis_label("Count")
                .legend(egui_plot::Legend::default());

            hist_plot.show(ui, |plot_ui| {
                if !self.latest_fitness.is_empty() {
                    let max_val = self.params.genome_length as f64;
                    let num_bins = 40.min(self.params.genome_length);
                    let bin_width = max_val / num_bins as f64;

                    let mut bins = vec![0u32; num_bins + 1];
                    for &f in &self.latest_fitness {
                        let idx = ((f / max_val) * num_bins as f64) as usize;
                        let idx = idx.min(num_bins);
                        bins[idx] += 1;
                    }

                    let bars: Vec<egui_plot::Bar> = bins
                        .iter()
                        .enumerate()
                        .filter(|(_, count)| **count > 0)
                        .map(|(i, count)| {
                            let count = *count;
                            let x = i as f64 * bin_width + bin_width / 2.0;
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
        });
    }
}
