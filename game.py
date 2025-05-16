# Import required modules
import tetris_ai.tetris_ga as ga          # Genetic Algorithm implementation
import tetris_ai.tetris_base as game      # Base Tetris game implementation
import tetris_ai.tetris_ga_evaluator as ai  # AI evaluation module
import argparse                           # Command line argument parser
import copy                              # For deep copying objects


def main(no_show_game):
  """
  Main training function that implements the genetic algorithm to find optimal weights
  for playing Tetris.

  Args:
      no_show_game (bool): Whether to display the game visualization during training

  Returns:
      list: The best weights found during training
  """
  # Configuration parameters for the genetic algorithm
  GAME_SPEED = 600      # Speed of game execution (milliseconds)
  NUM_GEN = 100         # Number of generations to evolve
  NUM_POP = 15          # Size of population in each generation
  NUM_EXP = 10          # Number of experiments to run
  # Generation gap (fraction of population to be replaced)
  GAP = 0.3
  # Number of children to produce in each generation
  NUM_CHILD = round(NUM_POP*GAP)
  MUTATION_RATE = 0.2   # Probability of mutation
  CROSSOVER_RATE = 0.75  # Probability of crossover
  MAX_SCORE = 200000    # Maximum score to achieve before stopping

  # Initialize genetic algorithm with population
  genetic_alg = ga.GA(NUM_POP)
  best_weights = None   # Store the best weights found
  best_fitness = 0      # Store the best fitness achieved

  # Create initial population
  init_pop = ga.GA(NUM_POP)

  # Run multiple experiments
  for e in range(NUM_EXP):
    print(f'\nStarting experiment {e + 1}/{NUM_EXP}')
    # Create a deep copy of initial population for each experiment
    pop = copy.deepcopy(init_pop)

    # Evolution process for each generation
    for g in range(NUM_GEN):
      print(f'\n - - - - Exp: {e}\t Generation: {g} - - - - \n')

      # Select parents using roulette wheel selection
      selected_pop = pop.selection(pop.chromosomes, NUM_CHILD,
                                   type="roulette")

      # Create new chromosomes through crossover and mutation
      new_chromo = pop.operator(selected_pop,
                                crossover="uniform",
                                mutation="random",
                                crossover_rate=CROSSOVER_RATE,
                                mutation_rate=MUTATION_RATE)

      # Evaluate fitness of new chromosomes
      for i in range(NUM_CHILD):
        # Run game simulation with current weights
        game_state = ai.run_game(pop.chromosomes[i], GAME_SPEED,
                                 MAX_SCORE, no_show_game)
        new_chromo[i].calc_fitness(game_state)

        # Update best weights if better fitness is found
        if new_chromo[i].fitness > best_fitness:
          best_fitness = new_chromo[i].fitness
          best_weights = new_chromo[i].weights

      # Replace old chromosomes with new ones
      pop.replace(new_chromo)

      # Display current generation's fitness information
      fitness = [chrom.score for chrom in pop.chromosomes]
      print("Current population fitness:", fitness)
      print(f"Best fitness so far: {best_fitness}")
      print(f"Best weights: {best_weights}")

  return best_weights


if __name__ == "__main__":
  # Set up command line argument parser
  parser = argparse.ArgumentParser(description="Tetris AI")
  parser.add_argument('--train',
                      action='store_true',
                      help='Whether or not to train the AI')
  parser.add_argument('--game',
                      action='store_true',
                      help='Run the base game without AI')
  parser.add_argument('--no-show',
                      action='store_true',
                      help='Whether to show the game')

  args = parser.parse_args()

  # Execute based on command line arguments
  if (args.train):
    # Train the AI to find optimal weights
    best_weights = main(args.no_show)
    print("\nTraining completed!")
    print(f"Final best weights: {best_weights}")

  elif (args.game):
    # Run manual Tetris game without AI
    game.MANUAL_GAME = True
    game.main()

  else:
    # Run game with pre-trained optimal weights
    optimal_weights = [-0.97, 5.47, -13.74, -0.73,
                       7.99, -0.86, -0.72]  # after 100 generations
    chromo = ga.Chromosome(optimal_weights)
    ai.run_game(chromo, speed=100, max_score=200000, no_show=False)
