
# coding: utf-8

# # Genetic Algorithm in Tensorflow (part 2)
# 
# OK! So we have a working GA in pure Python. Now let's implement in tensorflow and see what the graph looks like...
# 

# In[1]:


import tensorflow as tf
import numpy as np
import collections


# In[2]:


def _r(*args, **kwargs):
    """
    Runs in tensorflow, for testing.
    """
    return tf.Session().run(*args, **kwargs)


# In[3]:

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

def new_genome():
    """
    Creates a random genome.
    
    A genome is a length 100 array of 1s and 0s
    """
    return tf.random_uniform(
        [100], minval=0, maxval=2, dtype=tf.int32
    )

# assert(len(_r(new_genome())) == 100)
# assert(1 in _r(new_genome()))
# assert(0 in _r(new_genome()))
# assert(2 not in _r(new_genome()))


# In[4]:


# the best possible genome, our solution we are looking for
genome_1s = tf.ones(100, dtype=tf.int32)
# # the worst genome
genome_0s = tf.zeros(100, dtype=tf.int32)


# In[5]:


def compute_error(genome):
    """
    Returns the error for a genome. This is the number 0s in it.
    """
    return 100 - tf.reduce_sum(genome)
# 
# assert _r(compute_error(genome_1s)) == 0
# assert _r(compute_error(genome_0s)) == 100


# In[6]:


def _mutate(genome, index_to_change, value):
    res = tf.concat(
        [
            genome[:index_to_change],
            value,
            genome[index_to_change + 1:]
        ],
        0
    )
    res.set_shape([100])
    return res

# assert _r(compute_error(_mutate(genome_1s, 0, [0]))) == 1


def mutate(genome):
    """
    Returns a new genome with one item in it randomly changed to a 1 or 0
    """
    index_to_change = tf.random_uniform([1], minval=0, maxval=100, dtype=tf.int32)[0]
    
    new_val = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)
    return _mutate(genome, index_to_change, new_val)

# assert _r(compute_error(mutate(genome_1s))) in [0, 1]
# assert _r(compute_error(mutate(genome_0s))) in [100, 99]


# In[7]:


def _crossover(genome_a, genome_b, point):
    res =  tf.concat(
        [
            genome_a[:point],
            genome_b[point:]
        ],
        0
    )
    res.set_shape([100])
    return res

def tf_equals(a, b):
    return tf.reduce_all(tf.equal(a, b))
# 
# assert _r(
#     tf_equals(
#         _crossover(genome_1s, genome_0s, 0), 
#         genome_0s
#     )
# )
# 
# assert _r(
#     tf_equals(
#         _crossover(genome_1s, genome_0s, 100), 
#         genome_1s
#     )
# )
# 
# assert _r(compute_error(_crossover(genome_1s, genome_0s, 50))) == 50

def crossover(genome_a, genome_b):
    """
    Returns a new genome from two others, choosing a random pivot point
    and the part of genome_a up to that, with the part of genome_b from
    that point on.
    """
    crossover_point = tf.random_uniform([1], minval=0, maxval=101, dtype=tf.int32)[0]
    return _crossover(genome_a, genome_b, crossover_point)
    


# In[8]:


Population = collections.namedtuple('Population', ['genomes', 'errors'])


# In[9]:


def errors_from_genomes(genomes):
    return tf.map_fn(compute_error, genomes)

def population_from_genomes(genomes):
    return Population(genomes=genomes, errors=errors_from_genomes(genomes))

def initial_population():
    """
    Returns an initial population of individuals. A population has
    two arrays, genomes and error. Each of length 100 so that that
    errors[i] is the error for genome[i].
    """
    genomes = tf.map_fn(
        lambda _: new_genome(),
        tf.zeros((100, 100), dtype=tf.int32)
    )
    return population_from_genomes(genomes)

# assert len(_r(initial_population().genomes)) == 100


# In[11]:


def select_parent(population):
    """
    Selects a parent from the population, with probability inversly
    proportional to it's error
    """
    unnormalized_ps = tf.reciprocal(tf.to_float(population.errors))
    # hack for random choice till it's implemented
    # https://github.com/tensorflow/tensorflow/issues/8496
    index = tf.cast(
        tf.multinomial(tf.log([unnormalized_ps]), 1)[0][0],
        tf.int32
    )
    return population.genomes[index]

# very_bad = np.zeros(100)
# very_good = np.ones(100)
# very_good[0] = 0
# genomes = tf.constant(
#     np.concatenate((np.tile(very_bad, [50, 1]), np.tile(very_good, [50, 1])))
# )
# population = population_from_genomes(genomes)
# assert _r(compute_error(select_parent(population))) == _r(compute_error(very_good))


# In[12]:


def create_child(population):
    should_mutate = tf.random_uniform([1], minval=0, maxval=1)[0] < 0.1
    
    def mutate_child(): return mutate(select_parent(population))
    def crossover_child(): return crossover(select_parent(population), select_parent(population))
    return tf.cond(
        should_mutate,
        mutate_child,
        crossover_child 
    )

# assert len(_r(create_child(initial_population()))) == 100


# In[ ]:


def create_children(population):
    return tf.map_fn(
        lambda _: create_child(population),
        tf.zeros((100, 100), dtype=tf.int32)
    )

# assert len(_r(create_children(initial_population()))) == 100


# In[13]:


def best_error(population):
    return tf.reduce_min(population.errors)


def should_continue(population):
    return tf.Print(
        best_error(population) > 0,
        [best_error(population)]
    )

# _r(should_continue(initial_population()))


# In[14]:



# In[ ]:


# _r(best_error(initial_population()))
# _r(tf.Print(1, [1]))

# In[ ]:


def next_generation(population):
    new_genomes = create_children(population)
    new_population = population_from_genomes(new_genomes)
    return [new_population]
def run_ga():
    return tf.while_loop(
        should_continue,
        next_generation,
        [initial_population()]
    )[0]


# In[ ]:

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
with jit_scope():
    sess.run(run_ga())