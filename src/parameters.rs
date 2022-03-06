use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};
use set_genome::Parameters as GenomeParameters;

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Parameters {
    pub map_elites: MapElitesParameters,
    pub genome: GenomeParameters,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct MapElitesParameters {
    pub map_resolution: usize,
    pub feature_ranges: Vec<(f64, f64)>,
    pub initial_runs: usize,
    pub batch_size: usize,
}

impl Parameters {
    pub fn new(path: &str) -> Result<Self, ConfigError> {
        let mut s = Config::new();

        // Start off by merging in the "default" configuration file
        s.merge(File::with_name(path))?;

        // You can deserialize (and thus freeze) the entire configuration as
        s.try_into()
    }
}
