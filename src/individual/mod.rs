use std::ops::{Deref, DerefMut};

use rand::Rng;
use serde::{Deserialize, Serialize};
use set_genome::Genome;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Individual {
    pub genome: Genome,
    pub behavior: Vec<f64>,
    pub fitness: f64,
}

impl Deref for Individual {
    type Target = Genome;

    fn deref(&self) -> &Self::Target {
        &self.genome
    }
}

impl DerefMut for Individual {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.genome
    }
}

impl Individual {
    pub fn from_genome(genome: Genome) -> Self {
        Self {
            genome,
            behavior: Vec::new(),
            fitness: 0.0,
        }
    }

    // self is fitter if it has higher score or in case of equal score has fewer genes, i.e. less complexity
    pub fn is_fitter_than(&self, other: &Self) -> bool {
        self.fitness > other.fitness
            || ((self.fitness - other.fitness).abs() < f64::EPSILON
                && self.genome.len() < other.genome.len())
    }

    pub fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let (fitter, weaker) = if self.is_fitter_than(other) {
            (&self.genome, &other.genome)
        } else {
            (&other.genome, &self.genome)
        };

        Individual {
            genome: fitter.cross_in(weaker, rng),
            behavior: Vec::new(),
            fitness: 0.0,
        }
    }
}
