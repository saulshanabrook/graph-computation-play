use rand::Rng;
use rand::distributions::{IndependentSample, Range};

extern crate rand;
const L: usize = 100;

type Gene = u8;
type Genome = [u8; L];
type Error = u8;

fn new_gene<R: Rng>(random_gene: &Range<Gene>, rng: &mut R) -> Gene {
    return random_gene.ind_sample(rng);
}

fn new_genome<R: Rng>(random_gene: &Range<Gene>, rng: &mut R) -> Genome {
    let mut genome = [0; L];
    for i in 0..L {
        genome[i] = new_gene(random_gene, rng);
    }
    return genome;
}

fn compute_error(genome: &Genome) -> Error {
    return genome.into_iter().sum();
}

fn mutate<R: Rng>(
    random_index: &Range<usize>,
    random_gene: &Range<Gene>,
    rng: &mut R,
    mut genome: Genome,
) -> Genome {
    let index_to_change = random_index.ind_sample(rng);
    genome[index_to_change] = new_gene(random_gene, rng);
    return genome;
}

fn crossover<R: Rng>(
    random_crossover_point: &Range<usize>,
    rng: &mut R,
    mut genome_a: Genome,
    genome_b: Genome,
) -> [u8; L] {
    let crossover_point = random_crossover_point.ind_sample(rng);
    for i in 0..crossover_point {
        genome_a[i] = genome_b[i];
    }
    return genome_a;
}

const POP_SIZE: usize = 100;
type Genomes = [Genome; POP_SIZE];
type Errors = [Error; POP_SIZE];

fn errors_from_genomes(genomes: Genomes) -> Errors {
    let mut es: Errors = [0; POP_SIZE];
    for i in 0..L {
        es[i] = compute_error(&genomes[i]);
    }
    return es;
}
struct Population {
    genomes: Genomes,
    errors: Errors,
}

fn population_from_genomes(genomes: Genomes) -> Population {
    return Population {
        genomes: genomes,
        errors: errors_from_genomes(genomes),
    };
}

fn initial_population<R: Rng>(random_gene: &Range<Gene>, rng: &mut R) -> Population {
    let mut gs: Genomes = [[0; L]; POP_SIZE];
    for i in 0..L {
        gs[i] = new_genome(random_gene, rng);
    }
    return population_from_genomes(gs);
}



// use https://docs.rs/rand/0.3.17/rand/distributions/struct.WeightedChoice.html for sampling




fn main() {
    let mut rng = rand::thread_rng();
    let random_gene = Range::new(0, 2);
    let random_index = Range::new(0, L);
    let random_crossover_point = Range::new(0, L + 1);

    let genome = new_genome(&random_gene, &mut rng);
    println!("{:?}", compute_error(&genome));
    let genome_ = mutate(&random_index, &random_gene, &mut rng, genome);
    println!("{:?}", compute_error(&genome));
    println!("{:?}", compute_error(&genome_));
    let genome__ = new_genome(&random_gene, &mut rng);
    println!("{:?}", compute_error(&genome__));
    let another_gene = crossover(&random_crossover_point, &mut rng, genome, genome__);
    println!("{:?}", compute_error(&another_gene));
}
