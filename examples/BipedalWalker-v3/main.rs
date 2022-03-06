use favannat::matrix::fabricator::RecurrentMatrixFabricator;
use favannat::network::{StatefulEvaluator, StatefulFabricator};
use gym::{utility::StandardScaler, SpaceData, SpaceTemplate, State};
use map_elites::{Individual, Runtime};
use ndarray::{stack, Array2, Axis};

use std::time::SystemTime;
use std::{env, fs};
use std::{ops::Deref, time::Instant};
use tracing::{error, info};

pub const TRAINING_RUNS: usize = 1;
pub const VALIDATION_RUNS: usize = 100;
pub const SIMULATION_STEPS: usize = 1600;
pub const ENV: &str = "BipedalWalker-v3";
pub const REQUIRED_FITNESS: f64 = 300.0;

static mut TIMESTAMP: u64 = 0;

fn timestamp() -> u64 {
    unsafe { TIMESTAMP }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    unsafe {
        TIMESTAMP = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    let file_appender = tracing_appender::rolling::never(
        format!("./examples/{}/logs", ENV),
        format!("{}_stats.log", timestamp()),
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .json()
        .init();

    if args.get(1).is_some() {
        let winner_json = fs::read_to_string(format!("examples/{}/winner_1601592694.json", ENV))
            .expect("cant read file");
        let winner: Individual = serde_json::from_str(&winner_json).unwrap();
        let standard_scaler: StandardScaler = serde_json::from_str(&winner_json).unwrap();
        run(&standard_scaler, &winner, 1, SIMULATION_STEPS, true, false);
    } else {
        train(StandardScaler::for_environment(ENV));
    }
}

fn train(standard_scaler: StandardScaler) {
    let other_standard_scaler = standard_scaler.clone();

    let fitness_function = move |individual: &Individual| -> (f64, Vec<f64>) {
        let (training_fitness, training_observations) = run(
            &standard_scaler,
            individual,
            TRAINING_RUNS,
            SIMULATION_STEPS,
            false,
            false,
        );

        let mut observations = training_observations;
        let mut fitness = training_fitness;
        let behavior_characterization;

        if fitness > 0.0 {
            dbg!(fitness);
        }

        if fitness >= REQUIRED_FITNESS {
            info!("hit task theshold, starting validation runs...");
            let (validation_fitness, validation_observations) = run(
                &standard_scaler,
                individual,
                VALIDATION_RUNS,
                SIMULATION_STEPS,
                false,
                false,
            );

            // log possible solutions to file
            info!(
                "finished validation runs with {} average fitness",
                validation_fitness
            );
            if validation_fitness > REQUIRED_FITNESS {
                fitness = validation_fitness;
                observations = validation_observations;
            } else {
                fitness = REQUIRED_FITNESS - 1.0;
            }
        }

        let observation_means = observations.mean_axis(Axis(0)).unwrap();
        // let observation_std_dev = observations.std_axis(Axis(0), 0.0);

        behavior_characterization = vec![
            observation_means[[4]],
            observation_means[[6]],
            observation_means[[8]],
            observation_means[[9]],
            observation_means[[11]],
            observation_means[[13]],
        ];

        // behavior_characterization = observation_means.iter().take(14).cloned().collect();
        /* observation_means
        .iter()
        .take(14)
        .cloned()
        .chain(observation_std_dev.iter().take(14).cloned())
        .collect(), */

        (fitness, behavior_characterization)
    };

    let neat = Runtime::new(
        &format!("examples/{}/config.toml", ENV),
        Box::new(fitness_function),
    );

    let now = Instant::now();

    info!(target: "app::parameters", "starting training: {:#?}", neat.parameters);

    if let Some(winner_map) = neat.initilize().find(|elites_map| {
        println!(
            "map cells: {}, map capacity: {}",
            elites_map.len(),
            elites_map.capacity()
        );
        info!(target: "training::top-fitness", fitness = ?elites_map.top_individual().fitness);
        for top in &elites_map.sorted_individuals()[0..5] {
            run(
                &other_standard_scaler,
                top,
                1,
                SIMULATION_STEPS,
                true,
                false,
            );
        }
        elites_map.top_individual().fitness > REQUIRED_FITNESS
    }) {
        fs::write(
            format!("examples/{}/{}_winner.json", ENV, timestamp()),
            serde_json::to_string(&winner_map.top_individual()).unwrap(),
        )
        .expect("Unable to write file");
        fs::write(
            format!("examples/{}/{}_winner_map.json", ENV, timestamp()),
            serde_json::to_string(&winner_map).unwrap(),
        )
        .expect("Unable to write file");
        fs::write(
            format!("examples/{}/{}_winner_parameters.json", ENV, timestamp()),
            serde_json::to_string(&neat.parameters).unwrap(),
        )
        .expect("Unable to write file");
        fs::write(
            format!(
                "examples/{}/{}_winner_standard_scaler.json",
                ENV,
                timestamp()
            ),
            serde_json::to_string(&other_standard_scaler).unwrap(),
        )
        .expect("Unable to write file");

        info!(
            "winning individual ({},{}) after {} seconds: {:?}",
            winner_map.top_individual().nodes().count(),
            winner_map.top_individual().feed_forward.len(),
            now.elapsed().as_secs(),
            winner_map.top_individual()
        );
    }
}

fn run(
    standard_scaler: &StandardScaler,
    net: &Individual,
    runs: usize,
    steps: usize,
    render: bool,
    debug: bool,
) -> (f64, Array2<f64>) {
    let gym = gym::GymClient::default();
    let env = gym.make(ENV);

    let mut evaluator = RecurrentMatrixFabricator::fabricate(net.deref()).unwrap();
    // let mut evaluator = LoopingFabricator::fabricate(net).unwrap();
    let mut fitness = 0.0;
    let mut all_observations;

    if let SpaceTemplate::BOX { shape, .. } = env.observation_space() {
        all_observations = Array2::zeros((1, shape[0]));
    } else {
        panic!("is no box observation space")
    }

    if debug {
        dbg!(net);
        dbg!(&evaluator);
    }

    for run in 0..runs {
        evaluator.reset_internal_state();
        let mut recent_observation = env.reset().expect("Unable to reset");
        let mut total_reward = 0.0;

        if debug {
            dbg!(run);
        }

        for step in 0..steps {
            if render {
                env.render();
            }

            let mut observations = recent_observation.get_box().unwrap();

            all_observations = stack![
                Axis(0),
                all_observations,
                observations.clone().insert_axis(Axis(0))
            ];

            standard_scaler.scale_inplace(observations.view_mut());

            // add bias input
            let input = stack![Axis(0), observations, [1.0]];
            let output = evaluator.evaluate(input.clone());

            if debug {
                dbg!(&input);
                dbg!(&output);
            }

            let (observation, reward, is_done) = match env.step(&SpaceData::BOX(output.clone())) {
                Ok(State {
                    observation,
                    reward,
                    is_done,
                }) => (observation, reward, is_done),
                Err(err) => {
                    error!("evaluation error: {}", err);
                    dbg!(run);
                    dbg!(input);
                    dbg!(output);
                    dbg!(evaluator);
                    dbg!(net);
                    dbg!(all_observations);
                    panic!("evaluation error");
                }
            };

            recent_observation = observation;
            total_reward += reward;

            if is_done {
                if render {
                    println!("finished with reward {} after {} steps", total_reward, step);
                }
                break;
            }
        }
        fitness += total_reward;
    }

    if debug {
        dbg!(&all_observations);
    }

    (fitness / runs as f64, all_observations)
}
