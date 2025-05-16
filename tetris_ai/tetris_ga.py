# Required imports for the genetic algorithm implementation
import pygame         # For game visualization
import random        # For random number generation
import numpy as np   # For numerical operations
import copy          # For deep copying objects
import tetris_ai.tetris_base as game        # Base Tetris game mechanics
import tetris_ai.tetris_ga_evaluator as ai  # AI evaluation module


class Chromosome():
  """
  Represents a single solution (individual) in the genetic algorithm population.
  Each chromosome encodes a potential AI strategy through a set of weights.
  These weights determine how the AI evaluates different aspects of the game state
  to make decisions about piece placement.

  The chromosome's fitness is determined by how well it performs in actual gameplay,
  measured by the score achieved in a Tetris game.
  """

  def __init__(self, weights):
    """
    Initialize a chromosome with given weights.

    Args:
        weights (list): List of floating-point weights that determine how different
                      game features are valued during decision making. Each weight
                      corresponds to a specific game feature (e.g., holes, lines cleared).
    """
    self.weights = weights  # Weights for evaluating different game features
    self.score = 0         # Game score achieved by this chromosome
    # Fitness score (identical to game score in this implementation)
    self.fitness = 0

  def calc_fitness(self, game_state):
    """
    Calculate and store the fitness score for this chromosome based on its game performance.
    In this implementation, fitness is directly mapped to the game score, meaning
    chromosomes that achieve higher scores in the game are considered more fit.

    Args:
        game_state (tuple): Contains various game statistics including the final score
                          at index 2
    """
    self.score = game_state[2]    # Extract and store the game score
    self.fitness = self.score     # Use game score directly as fitness measure

  def calc_best_move(self, board, piece, show_game=False):
    """
    Calculate the optimal move for the current piece using this chromosome's weights.
    This method evaluates all possible positions and rotations for the current piece
    and selects the one that maximizes the weighted sum of game features.

    Args:
        board (list): 2D list representing current game board state
        piece (dict): Dictionary containing current piece information including shape and position
        show_game (bool): Flag to enable/disable game visualization

    Returns:
        tuple: (best_X, best_R) where:
              best_X: Optimal horizontal position for the piece
              best_R: Optimal rotation state for the piece
    """
    best_X = 0          # Best horizontal position found
    best_R = 0          # Best rotation state found
    best_Y = 0          # Best vertical position found
    best_score = -100000  # Initialize with worst possible score

    # Calculate initial board statistics for comparison
    num_holes_bef, num_blocking_blocks_bef = game.calc_initial_move_info(board)

    # Evaluate all possible piece rotations
    for r in range(len(game.PIECES[piece['shape']])):
      # For each rotation, try all possible horizontal positions
      for x in range(-2, game.BOARDWIDTH-2):
        # Calculate metrics for this specific move combination
        movement_info = game.calc_move_info(board, piece, x, r,
                                            num_holes_bef,
                                            num_blocking_blocks_bef)

        if (movement_info[0]):  # If the move is valid
          # Calculate weighted score using this chromosome's weights
          movement_score = 0
          for i in range(1, len(movement_info)):
            movement_score += self.weights[i-1]*movement_info[i]

          # Update best move if current score is better
          if (movement_score > best_score):
            best_score = movement_score
            best_X = x
            best_R = r
            best_Y = piece['y']

    # Set piece position based on visualization mode
    if (show_game):
      piece['y'] = best_Y
    else:
      piece['y'] = -2

    piece['x'] = best_X
    piece['rotation'] = best_R

    return best_X, best_R


class GA:
  """
  Genetic Algorithm implementation for optimizing Tetris AI weights.
  This class manages a population of chromosomes and implements the genetic algorithm
  operations (selection, crossover, mutation) to evolve better solutions over time.

  The algorithm works by:
  1. Maintaining a population of potential solutions (chromosomes)
  2. Evaluating each solution's fitness through gameplay
  3. Selecting better solutions for reproduction
  4. Creating new solutions through crossover and mutation
  5. Replacing worse solutions with better ones
  """

  def __init__(self, num_pop, num_weights=7, lb=-1, ub=1):
    """
    Initialize the genetic algorithm with a random population.

    Args:
        num_pop (int): Size of the population to maintain
        num_weights (int): Number of weights per chromosome (default: 7)
        lb (float): Lower bound for random weight initialization (default: -1)
        ub (float): Upper bound for random weight initialization (default: 1)
    """
    self.chromosomes = []

    # Create initial population with random weights
    for i in range(num_pop):
      # Generate random weights within specified bounds
      weights = np.random.uniform(lb, ub, size=(num_weights))
      chrom = Chromosome(weights)
      self.chromosomes.append(chrom)

      # Evaluate initial fitness through gameplay
      game_state = ai.run_game(self.chromosomes[i], 1000, 200000, True)
      self.chromosomes[i].calc_fitness(game_state)

  def __str__(self):
    """
    Create a string representation of the population showing each chromosome's
    weights and fitness score.

    Returns:
        str: Formatted string showing population details
    """
    for i, chromo in enumerate(self.chromosomes):
      print(f"Individual {i+1}")
      print(f"   Weights: {chromo.weights}")
      print(f"   Score: {chromo.score}")
    return ''

  def selection(self, chromosomes, num_selection, type="roulette"):
    """
    Select chromosomes for reproduction using specified selection method.
    Selection is biased towards chromosomes with higher fitness scores.

    Args:
        chromosomes (list): Pool of chromosomes to select from
        num_selection (int): Number of chromosomes to select
        type (str): Selection method (currently only "roulette" is supported)

    Returns:
        list: Selected chromosomes for reproduction
    """
    if (type == "roulette"):
      selected_chromos = self._roulette(chromosomes, num_selection)
    else:
      raise ValueError(f"Selection type {type} not defined")

    return selected_chromos

  def _roulette(self, chromosomes, num_selection):
    """
    Implement roulette wheel selection.
    This method selects chromosomes with probability proportional to their fitness.
    Higher fitness individuals have a better chance of being selected.

    Args:
        chromosomes (list): Pool of chromosomes to select from
        num_selection (int): Number of chromosomes to select

    Returns:
        list: Selected chromosomes based on roulette wheel selection
    """
    # Extract fitness scores into numpy array
    fitness = np.array([chrom.score for chrom in chromosomes])

    # Normalize fitness values to create probability distribution
    norm_fitness = fitness/fitness.sum()

    # Calculate cumulative probabilities for selection
    roulette_prob = np.cumsum(norm_fitness)

    # Select chromosomes using roulette wheel method
    pop_selected = []
    while len(pop_selected) < num_selection:
      pick = random.random()  # Random number between 0 and 1
      for index, individual in enumerate(self.chromosomes):
        if pick < roulette_prob[index]:
          pop_selected.append(individual)
          break

    return pop_selected

  def operator(self, chromosomes, crossover="arithmetic", mutation="uniform",
               crossover_rate=0.5, mutation_rate=0.1):
    """
    Apply genetic operators (crossover and mutation) to create new chromosomes.
    This process creates genetic diversity and explores new potential solutions.

    Args:
        chromosomes (list): Parent chromosomes for breeding
        crossover (str): Crossover method (currently only "arithmetic" supported)
        mutation (str): Mutation method (currently only "uniform" supported)
        crossover_rate (float): Probability of crossover occurring (0-1)
        mutation_rate (float): Probability of mutation occurring (0-1)

    Returns:
        list: New chromosomes after applying genetic operators
    """
    new_chromo = self._arithmetic_crossover(chromosomes, mutation,
                                            crossover_rate, mutation_rate)

    self.mutation(new_chromo, mutation, mutation_rate)

    return new_chromo

  def _arithmetic_crossover(self, selected_pop, mutation, cross_rate=0.4,
                            mutation_rate=0.1):
    """
    Perform arithmetic crossover between pairs of chromosomes.
    Creates new chromosomes by taking weighted averages of parent chromosomes' weights.

    Args:
        selected_pop (list): Selected parent chromosomes
        mutation (str): Mutation method to apply after crossover
        cross_rate (float): Probability of crossover occurring (0-1)
        mutation_rate (float): Probability of mutation occurring (0-1)

    Returns:
        list: New chromosomes created through arithmetic crossover
    """
    N_genes = len(selected_pop[0].weights)
    new_chromo = [copy.deepcopy(c) for c in selected_pop]

    # Perform crossover on pairs of chromosomes
    for i in range(0, len(selected_pop), 2):
      a = random.random()  # Random weight for arithmetic averaging

      # Apply crossover with given probability
      tc_parent_1 = random.randint(0, 100)
      tc_parent_2 = random.randint(0, 100)
      if (tc_parent_1 < cross_rate*100 and tc_parent_2 < cross_rate*100):
        try:
          for j in range(0, N_genes):
            # Create new weights through weighted averaging
            new_chromo[i].weights[j] = a*new_chromo[i].weights[j] \
                + (1 - a)*new_chromo[i+1].weights[j]

            new_chromo[i+1].weights[j] = a*new_chromo[i+1].weights[j] \
                + (1 - a)*new_chromo[i].weights[j]

        except IndexError:
          pass

    return new_chromo

  def mutation(self, chromosome, type, mutation_rate):
    """
    Apply mutation to chromosomes to maintain genetic diversity.
    Mutation helps prevent the population from converging to local optima.

    Args:
        chromosome (list): Chromosomes to potentially mutate
        type (str): Mutation type (currently only "random" supported)
        mutation_rate (float): Probability of mutation occurring (0-1)
    """
    if (type == "random"):
      self._rand_mutation(chromosome, mutation_rate)
    else:
      raise ValueError(f"Type {type} not defined")

  def _rand_mutation(self, chromosome, mutation_rate):
    """
    Implement random mutation by randomly changing weights.
    Each weight has a chance to be replaced with a new random value.

    Args:
        chromosome (list): Chromosomes to potentially mutate
        mutation_rate (float): Probability of each weight being mutated (0-1)
    """
    for chromo in chromosome:
      for i, point in enumerate(chromo.weights):
        if random.random() < mutation_rate:
          chromo.weights[i] = random.uniform(-1.0, 1.0)

  def replace(self, new_chromo):
    """
    Replace worst-performing chromosomes in population with new ones.
    This maintains population size while improving overall fitness.

    Args:
        new_chromo (list): New chromosomes to add to population
    """
    # Sort population by fitness (descending order)
    new_pop = sorted(self.chromosomes, key=lambda x: x.score, reverse=True)
    # Replace worst chromosomes with new ones
    new_pop[-(len(new_chromo)):] = new_chromo
    # Shuffle population to maintain diversity
    random.shuffle(new_pop)

    self.chromosomes = new_pop
