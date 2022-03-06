use std::collections::HashMap;

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::individual::Individual;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ElitesMap {
    map: HashMap<Vec<usize>, Individual>,
    resolution: usize,
    feature_ranges: Vec<(f64, f64)>,
}

impl ElitesMap {
    pub fn new(resolution: usize, feature_ranges: Vec<(f64, f64)>) -> Self {
        Self {
            map: HashMap::new(),
            resolution,
            feature_ranges,
        }
    }

    // #[tracing::instrument]
    pub fn place_individual(&mut self, individual: Individual) {
        assert!(
            individual.behavior.len() == self.feature_ranges.len(),
            "behavior descriptor did not match features ranges"
        );

        let cell_index: Vec<usize> = individual
            .behavior
            .iter()
            .enumerate()
            .zip(self.feature_ranges.iter())
            .map(|((index, mut feature_value), (feature_min, feature_max))| {
                // cap feature value to configured interval
                if feature_value >= feature_max {
                    tracing::warn!(
                        "capping feature {} with value {} to feature_max {}",
                        index,
                        feature_value,
                        feature_max,
                    );
                    feature_value = feature_max;
                }
                if feature_value < feature_min {
                    tracing::warn!(
                        "capping feature {} with value {} to feature_min {}",
                        index,
                        feature_value,
                        feature_min,
                    );
                    feature_value = feature_min;
                }

                // add some epsilon to spanned range to have [min, max[ resolution
                ((feature_value - feature_min) / (feature_max - feature_min + 1E-15)
                    * self.resolution as f64)
                    .floor() as usize
            })
            .collect();

        self.map
            .entry(cell_index)
            .and_modify(|previous_individual| {
                if previous_individual.fitness > individual.fitness {
                    // fitness did not improve, do nothing
                } else {
                    *previous_individual = individual.clone();
                }
            })
            .or_insert(individual);
    }

    // ACTUALLY RANDOM
    /* pub fn get_random_individual(&self, rng: &mut impl Rng) -> Individual {
        self.map
            .values()
            .choose(rng)
            .cloned()
            .expect("map did not held any individual")
    } */

    // RANDOM BUT WEIGHTED BY GLOBAL FITNESS
    /* pub fn get_random_individual(&self, rng: &mut impl Rng) -> Individual {
        let individuals: Vec<&Individual> = self.map.values().collect();
        let fitnesses: Vec<f64> = individuals
            .iter()
            .map(|individual| individual.fitness)
            .collect();

        let mut min_fitness = f64::MAX;

        for &fitness in &fitnesses {
            if fitness < min_fitness {
                min_fitness = fitness;
            }
        }

        let weights: Vec<f64> = fitnesses
            .iter()
            // .map(|&fitness| (fitness - min_fitness + 1.0).log2())
            .map(|&fitness| (fitness - min_fitness))
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        individuals[dist.sample(rng)].clone()
    } */

    // RANDOM BUT WEIGHTED BY DOMINATED NEIGHBORS
    pub fn get_random_individual(&self, rng: &mut impl Rng) -> Individual {
        let individuals: Vec<(&Vec<usize>, &Individual)> = self.map.iter().collect();

        let weights: Vec<f64> = individuals
            .iter()
            .map(|(position, individual)| {
                let neighbors_count = self.neighbors(position).count() as f64;
                let dominated_neighbors_count = self
                    .neighbors(position)
                    .filter(|neighbor| individual.fitness > neighbor.fitness)
                    .count() as f64;
                // always count oneself in as dominated to produce non-zero weight
                (dominated_neighbors_count + 1.0) / (neighbors_count + 1.0)
            })
            .collect();

        /* if weights.iter().any(|&w| w < 1.0) {
            dbg!(&weights);
        } */

        let individuals: Vec<&Individual> = individuals
            .iter()
            .map(|&(_, individual)| individual)
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        individuals[dist.sample(rng)].clone()
    }

    pub fn update_resolution(&mut self, resolution: usize) {
        let stored_individuals = std::mem::replace(&mut self.map, HashMap::new());

        self.resolution = resolution;

        for (_, individual) in stored_individuals {
            self.place_individual(individual);
        }
    }

    pub fn top_individual(&self) -> Individual {
        self.map
            .values()
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .expect("could not compare floats")
            })
            .cloned()
            .expect("map did not held any individual")
    }

    pub fn sorted_individuals(&self) -> Vec<&Individual> {
        let mut individuals: Vec<&Individual> = self.map.values().collect();

        individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .expect("could not compare floats")
        });

        individuals
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn capacity(&self) -> usize {
        self.resolution.pow(self.feature_ranges.len() as u32)
    }
}

impl ElitesMap {
    fn neighbors<'a>(&'a self, position: &'a Vec<usize>) -> impl Iterator<Item = &'a Individual> {
        assert!(
            position.len() == self.feature_ranges.len(),
            "requested neighbors for invalid position {:?}",
            position
        );

        assert!(
            position.iter().all(|&chunk| chunk < self.resolution),
            "position contains invalid chunk index"
        );

        (0..position.len())
            .flat_map(move |feature| {
                let maximum = self.resolution - 1;
                let mut up = position.clone();
                let mut down = position.clone();

                match position[feature] {
                    0 => {
                        up[feature] += 1;
                        vec![up]
                    }
                    max_chunk if max_chunk == maximum => {
                        down[feature] -= 1;
                        vec![down]
                    }
                    _ => {
                        down[feature] -= 1;
                        up[feature] += 1;
                        vec![up, down]
                    }
                }
            })
            .flat_map(move |neighbor_position| self.map.get(&neighbor_position))
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::ThreadRng;

    use super::ElitesMap;
    use crate::individual::Individual;
    #[test]
    fn place_and_retrieve() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0)]);

        let individual = Individual {
            behavior: vec![3.0],
            fitness: 4.2,
            ..Default::default()
        };

        elites_map.place_individual(individual);

        assert_eq!(
            elites_map
                .map
                .get(&vec![3])
                .map(|individual| individual.fitness),
            Some(4.2)
        );
    }

    #[test]
    fn get_random() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0)]);

        let individual = Individual {
            behavior: vec![3.0],
            fitness: 4.2,
            ..Default::default()
        };

        elites_map.place_individual(individual);

        let mut rng = ThreadRng::default();

        assert!((elites_map.get_random_individual(&mut rng).fitness - 4.2).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_features() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0), (-5.0, 5.0)]);

        let individual = Individual {
            behavior: vec![3.0, -3.0],
            fitness: 4.2,
            ..Default::default()
        };

        elites_map.place_individual(individual);

        assert_eq!(
            elites_map
                .map
                .get(&vec![3, 0])
                .map(|individual| individual.fitness),
            Some(4.2)
        );
    }

    #[test]
    fn handle_out_of_feature_range() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, -1.0), (1.0, 5.0)]);

        let individual = Individual {
            behavior: vec![-9.0, 9.0],
            fitness: 3.9,
            ..Default::default()
        };

        elites_map.place_individual(individual);

        assert_eq!(
            elites_map
                .map
                .get(&vec![0, 3])
                .map(|individual| individual.fitness),
            Some(3.9)
        );
    }

    #[test]
    fn update_when_fitter() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0)]);

        let individual_base = Individual {
            behavior: vec![3.0],
            fitness: 1.0,
            ..Default::default()
        };

        let individual_less_fit = Individual {
            behavior: vec![3.0],
            fitness: 0.0,
            ..Default::default()
        };

        let individual_more_fit = Individual {
            behavior: vec![3.0],
            fitness: 2.0,
            ..Default::default()
        };

        elites_map.place_individual(individual_base);

        assert_eq!(
            elites_map
                .map
                .get(&vec![3])
                .map(|individual| individual.fitness),
            Some(1.0)
        );

        elites_map.place_individual(individual_less_fit);

        assert_eq!(
            elites_map
                .map
                .get(&vec![3])
                .map(|individual| individual.fitness),
            Some(1.0)
        );

        elites_map.place_individual(individual_more_fit);

        assert_eq!(
            elites_map
                .map
                .get(&vec![3])
                .map(|individual| individual.fitness),
            Some(2.0)
        );
    }

    #[test]
    fn update_resolution() {
        let mut elites_map = ElitesMap::new(2, vec![(1.0, 2.0)]);

        let individual = Individual {
            behavior: vec![1.5],
            fitness: 3.9,
            ..Default::default()
        };

        elites_map.place_individual(individual);

        assert_eq!(
            elites_map
                .map
                .get(&vec![0])
                .map(|individual| individual.fitness),
            Some(3.9)
        );

        elites_map.update_resolution(3);

        assert_eq!(
            elites_map
                .map
                .get(&vec![1])
                .map(|individual| individual.fitness),
            Some(3.9)
        );
    }

    #[test]
    fn get_top_individual() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0)]);

        let individual_less_fit = Individual {
            behavior: vec![3.0],
            fitness: 0.0,
            ..Default::default()
        };

        let individual_more_fit = Individual {
            behavior: vec![-3.0],
            fitness: 2.0,
            ..Default::default()
        };

        elites_map.place_individual(individual_less_fit);
        elites_map.place_individual(individual_more_fit);

        assert!((elites_map.top_individual().fitness - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_capacity() {
        let elites_map = ElitesMap::new(4, vec![(-5.0, 5.0), (-5.0, 5.0)]);

        assert_eq!(elites_map.capacity(), 16);
    }

    #[test]
    fn get_sorted_individuals() {
        let mut elites_map = ElitesMap::new(4, vec![(-5.0, 5.0)]);

        let individual_less_fit = Individual {
            behavior: vec![3.0],
            fitness: 1.0,
            ..Default::default()
        };

        let individual_more_fit = Individual {
            behavior: vec![-3.0],
            fitness: 2.0,
            ..Default::default()
        };

        elites_map.place_individual(individual_less_fit);
        elites_map.place_individual(individual_more_fit);

        let sorted_individuals = elites_map.sorted_individuals();

        assert!((sorted_individuals[0].fitness - 2.0).abs() < f64::EPSILON);
        assert!((sorted_individuals[1].fitness - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_neighbors() {
        let mut elites_map = ElitesMap::new(3, vec![(0.0, 3.0)]);

        let individual_up = Individual {
            behavior: vec![3.0],
            ..Default::default()
        };

        let individual_center = Individual {
            behavior: vec![2.0],
            ..Default::default()
        };

        let individual_down = Individual {
            behavior: vec![1.0],
            ..Default::default()
        };

        elites_map.place_individual(individual_up);
        elites_map.place_individual(individual_center);
        elites_map.place_individual(individual_down);

        let position = vec![1usize];

        let neighbors: Vec<&Individual> = elites_map.neighbors(&position).collect();

        dbg!(neighbors.len());

        assert_eq!(neighbors[0].behavior, vec![3.0]);
        assert_eq!(neighbors[1].behavior, vec![1.0]);
    }
}
