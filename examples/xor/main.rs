use favannat::matrix::fabricator::FeedForwardMatrixFabricator;
use favannat::network::{Evaluator, Fabricator};
use ndarray::array;
use std::{ops::Deref, time::Instant};

use map_elites::{Individual, Runtime};

fn main() {
    let fitness_function = |individual: &Individual| -> (f64, Vec<f64>) {
        let result_0;
        let result_1;
        let result_2;
        let result_3;

        /* match LoopingFabricator::fabricate(individual) {
        Ok(mut evaluator) => { */
        match FeedForwardMatrixFabricator::fabricate(individual.deref()) {
            Ok(evaluator) => {
                result_0 = evaluator.evaluate(array![1.0, 1.0, 0.0]);
                result_1 = evaluator.evaluate(array![1.0, 1.0, 1.0]);
                result_2 = evaluator.evaluate(array![1.0, 0.0, 1.0]);
                result_3 = evaluator.evaluate(array![1.0, 0.0, 0.0]);
            }
            Err(e) => {
                println!("error fabricating individual: {:?} {:?}", individual, e);
                panic!("")
            }
        }

        // calculate fitness

        let fitness = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        (
            fitness,
            vec![
                individual.feed_forward.len() as f64,
                individual.hidden.len() as f64,
            ],
        )
    };

    let runtime = Runtime::new("examples/xor/config.toml", Box::new(fitness_function));

    let mut millis_elapsed_in_run = Vec::new();
    let mut connections_in_winner_in_run = Vec::new();
    let mut nodes_in_winner_in_run = Vec::new();
    let mut generations_till_winner_in_run = Vec::new();

    for i in 0..100 {
        let now = Instant::now();

        if let Some((generations, winner_map)) =
            runtime
                .initilize()
                .enumerate()
                .find(|(iteration, elites_map)| {
                    dbg!(iteration);
                    dbg!(elites_map.top_individual().fitness);

                    elites_map.top_individual().fitness > 15.9
                })
        {
            millis_elapsed_in_run.push(now.elapsed().as_millis() as f64);
            connections_in_winner_in_run.push(winner_map.top_individual().feed_forward.len());
            nodes_in_winner_in_run.push(winner_map.top_individual().nodes().count());
            generations_till_winner_in_run.push(generations);
            println!(
                "finished run {} in {} seconds ({}, {}) {}",
                i,
                millis_elapsed_in_run.last().unwrap() / 1000.0,
                winner_map.top_individual().nodes().count(),
                winner_map.top_individual().feed_forward.len(),
                generations
            );
        }
    }

    let num_runs = millis_elapsed_in_run.len() as f64;

    let total_millis: f64 = millis_elapsed_in_run.iter().sum();
    let total_connections: usize = connections_in_winner_in_run.iter().sum();
    let total_nodes: usize = nodes_in_winner_in_run.iter().sum();
    let total_generations: usize = generations_till_winner_in_run.iter().sum();

    println!(
        "did {} runs in {} seconds / {} nodes average / {} connections / {} batches per run",
        num_runs,
        total_millis / num_runs / 1000.0,
        total_nodes as f64 / num_runs,
        total_connections as f64 / num_runs,
        total_generations as f64 / num_runs
    );

    /* let now = Instant::now();

    if let Some(winner) = neat
        .run()
        .filter_map(|evaluation| match evaluation {
            Evaluation::Progress(report) => {
                println!("{:#?}", report);
                None
            }
            Evaluation::Solution(genome) => Some(genome),
        })
        .next()
    {
        let secs = now.elapsed().as_millis();
        println!(
            "winning genome ({},{}) after {} seconds: {:?}",
            winner.nodes().count(),
            winner.feed_forward.len(),
            secs as f64 / 1000.0,
            winner
        );
        let evaluator = LoopingFabricator::fabricate(&winner).unwrap();
        println!("as evaluator {:#?}", evaluator);
    } */
}

#[cfg(test)]
mod tests {
    #[test]
    fn fitness_function_good_result() {
        let result_0: Vec<f64> = vec![1.0];
        let result_1: Vec<f64> = vec![0.0];
        let result_2: Vec<f64> = vec![1.0];
        let result_3: Vec<f64> = vec![0.0];

        let result = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        println!("result {:?}", res/*  */ult);

        assert_eq!(result, 16.0);
    }

    #[test]
    fn fitness_function_bad_result() {
        let result_0: Vec<f64> = vec![0.0];
        let result_1: Vec<f64> = vec![1.0];
        let result_2: Vec<f64> = vec![0.0];
        let result_3: Vec<f64> = vec![1.0];

        let result = (4.0
            - ((1.0 - result_0[0])
                + (0.0 - result_1[0]).abs()
                + (1.0 - result_2[0])
                + (0.0 - result_3[0]).abs()))
        .powi(2);

        println!("result {:?}", result);

        assert_eq!(result, 0.0);
    }
}
