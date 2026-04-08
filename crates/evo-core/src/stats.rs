use serde::Serialize;

/// Per-generation statistics.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    pub generation: u64,
    pub population_size: usize,
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub mean_fitness: f64,
    pub std_dev_fitness: f64,
}

impl GenerationStats {
    pub fn from_fitness(generation: u64, fitness: &[f64]) -> Self {
        let n = fitness.len();
        assert!(n > 0, "cannot compute stats for empty population");

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;

        for &f in fitness {
            if f < min {
                min = f;
            }
            if f > max {
                max = f;
            }
            sum += f;
        }

        let mean = sum / n as f64;

        let variance = fitness.iter().map(|&f| (f - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        GenerationStats {
            generation,
            population_size: n,
            min_fitness: min,
            max_fitness: max,
            mean_fitness: mean,
            std_dev_fitness: std_dev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let fitness = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = GenerationStats::from_fitness(0, &fitness);
        assert_eq!(stats.population_size, 5);
        assert!((stats.min_fitness - 1.0).abs() < f64::EPSILON);
        assert!((stats.max_fitness - 5.0).abs() < f64::EPSILON);
        assert!((stats.mean_fitness - 3.0).abs() < f64::EPSILON);
        // std dev of [1,2,3,4,5] = sqrt(2)
        assert!((stats.std_dev_fitness - std::f64::consts::SQRT_2).abs() < 1e-10);
    }
}
