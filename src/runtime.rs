use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use set_genome::GenomeContext;
use tracing::{debug, info};

use crate::{elites_map::ElitesMap, parameters::Parameters, Individual};

pub struct Runtime {
    fitness_function: Box<dyn Fn(&Individual) -> (f64, Vec<f64>) + Send + Sync>,
    pub parameters: Parameters,
}

pub struct RuntimeIterator<'a> {
    runtime: &'a Runtime,
    elites_map: ElitesMap,
    genome_context: GenomeContext,
}

impl Runtime {
    pub fn new(
        path: &str,
        fitness_function: Box<dyn Fn(&Individual) -> (f64, Vec<f64>) + Send + Sync>,
    ) -> Self {
        let parameters = Parameters::new(path).unwrap();
        Self {
            parameters,
            fitness_function,
        }
    }

    /* fn evaluate(&self, individual: &mut Individual) {
        let (fitness, behavior) = (self.fitness_function)(individual);
        individual.fitness = fitness;
        individual.behavior = behavior;
    } */

    fn evaluate_parallel(&self, individuals: &mut Vec<Individual>) {
        info!("evaluating {} individuals in parallel", individuals.len());

        individuals
            .par_iter_mut()
            .enumerate()
            .map(|(index, individual)| {
                let (fitness, behavior) = (self.fitness_function)(individual);
                individual.fitness = fitness;
                individual.behavior = behavior;
                debug!("evaluated {}th individual", index);
            })
            .collect::<()>()
    }

    pub fn initilize(&self) -> RuntimeIterator {
        info!("starting runtime initialization");

        let mut genome_context = GenomeContext::new(self.parameters.genome.clone());

        // generate individual with initial ids for genome
        let initial_individual = Individual::from_genome(genome_context.uninitialized_genome());

        let mut elites_map = ElitesMap::new(
            self.parameters.map_elites.map_resolution,
            self.parameters.map_elites.feature_ranges.clone(),
        );

        let mut initial_individuals: Vec<Individual> = (0..self.parameters.map_elites.initial_runs)
            .map(|_| {
                let mut other_individual = initial_individual.clone();
                other_individual.init_with_context(&mut genome_context);
                other_individual.mutate_with_context(&mut genome_context);
                other_individual
            })
            .collect();

        self.evaluate_parallel(&mut initial_individuals);

        for individual in initial_individuals {
            elites_map.place_individual(individual);
        }

        RuntimeIterator {
            genome_context,
            elites_map,
            runtime: self,
        }
    }
}

impl<'a> Iterator for RuntimeIterator<'a> {
    type Item = ElitesMap;

    fn next(&mut self) -> Option<Self::Item> {
        info!("selecting next individual batch");

        let mut random_individuals: Vec<Individual> =
            (0..self.runtime.parameters.map_elites.batch_size)
                .map(|_| {
                    let mut random_individual = self
                        .elites_map
                        .get_random_individual(&mut self.genome_context.rng);
                    random_individual.mutate_with_context(&mut self.genome_context);
                    random_individual
                })
                .collect();

        info!("evaluating selected individual batch");

        self.runtime.evaluate_parallel(&mut random_individuals);

        info!("placing evaluated individual batch");

        for individual in random_individuals {
            self.elites_map.place_individual(individual);
        }

        info!("finished batch");

        Some(self.elites_map.clone())
    }
}
