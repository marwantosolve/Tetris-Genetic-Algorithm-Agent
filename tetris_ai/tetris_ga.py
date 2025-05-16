import pygame
import random
import numpy as np
import copy
import tetris_ai.tetris_base as game
import tetris_ai.tetris_ga_evaluator as ai


class Chromosome():
  def __init__(self, weights):
    self.weights = weights
    self.score = 0

  def calc_fitness(self, game_state):
    """Calculate fitness"""
    self.score = game_state[2]

  def calc_best_move(self, board, piece, show_game=False):
    """Calculate best movement
    Select the best move based on the chromosome weights.

    """
    best_X = 0
    best_R = 0
    best_Y = 0
    best_score = -100000

    num_holes_bef, num_blocking_blocks_bef = game.calc_initial_move_info(board)
    for r in range(len(game.PIECES[piece['shape']])):

      for x in range(-2, game.BOARDWIDTH-2):

        movement_info = game.calc_move_info(board, piece, x, r,
                                            num_holes_bef,
                                            num_blocking_blocks_bef)

        if (movement_info[0]):

          movement_score = 0
          for i in range(1, len(movement_info)):
            movement_score += self.weights[i-1]*movement_info[i]

          if (movement_score > best_score):
            best_score = movement_score
            best_X = x
            best_R = r
            best_Y = piece['y']

    if (show_game):
      piece['y'] = best_Y
    else:
      piece['y'] = -2

    piece['x'] = best_X
    piece['rotation'] = best_R

    return best_X, best_R

# here we implement the genetic algorithm class


class GA:
  # Create a population of chromosomes
  def __init__(self, num_pop, num_weights=7, lb=-1, ub=1):
    self.chromosomes = []

    for i in range(num_pop):
      weights = np.random.uniform(lb, ub, size=(num_weights))
      chrom = Chromosome(weights)
      self.chromosomes.append(chrom)

      game_state = ai.run_game(self.chromosomes[i], 1000, 200000, True)
      self.chromosomes[i].calc_fitness(game_state)

  # Create a new population of chromosomes
  def __str__(self):
    for i, chromo in enumerate(self.chromosomes):
      print(f"Inidividuo {i+1}")
      print(f"   Weights: {chromo.weights}")
      print(f"   Score: {chromo.score}")

    return ''

  # Select the best chromosomes from the population (ABDO)

  # Selection method using roulette wheel
  def _roulette(self, chromosomes, num_selection):
    """Selection method using roulette wheel"""

    fitness = np.array([chrom.score for chrom in chromosomes])

    norm_fitness = fitness/fitness.sum()

    roulette_prob = np.cumsum(norm_fitness)

    pop_selected = []
    while len(pop_selected) < num_selection:
      pick = random.random()
      for index, individual in enumerate(self.chromosomes):
        if pick < roulette_prob[index]:
          pop_selected.append(individual)
          break

    return pop_selected

  # Create a new population of chromosomes using crossover and mutation (ABDO)

  # Crossover method using arithmetic crossover (ABDO)

  # Mutation method using uniform mutation (REDA)

  # Mutation method using random mutation (REDA)

  # Replace chromosomes from population with the new ones (REDA)
