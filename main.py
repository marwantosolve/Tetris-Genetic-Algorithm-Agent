import tetris_ai.tetris_genetic_algorithm as ga
import tetris_ai.tetris_base as game
import tetris_ai.optimized_ai_player as ai
import argparse
import copy


def main(no_show_game):
  GAME_SPEED = 600
  NUM_GEN = 100
  NUM_POP = 15
  NUM_EXP = 10
  GAP = 0.3
  NUM_CHILD = round(NUM_POP*GAP)
  MUTATION_RATE = 0.2
  CROSSOVER_RATE = 0.75
  MAX_SCORE = 200000

  genetic_alg = ga.GA(NUM_POP)
  best_weights = None
  best_fitness = 0

  init_pop = ga.GA(NUM_POP)

  for e in range(NUM_EXP):
    print(f'\nStarting experiment {e + 1}/{NUM_EXP}')
    pop = copy.deepcopy(init_pop)

    for g in range(NUM_GEN):
      print(f'\n - - - - Exp: {e}\t Generation: {g} - - - - \n')

      selected_pop = pop.selection(pop.chromosomes, NUM_CHILD,
                                   type="roulette")

      new_chromo = pop.operator(selected_pop,
                                crossover="uniform",
                                mutation="random",
                                crossover_rate=CROSSOVER_RATE,
                                mutation_rate=MUTATION_RATE)

      for i in range(NUM_CHILD):
        game_state = ai.run_game(pop.chromosomes[i], GAME_SPEED,
                                 MAX_SCORE, no_show_game)
        new_chromo[i].calc_fitness(game_state)

        # Track best weights
        if new_chromo[i].fitness > best_fitness:
          best_fitness = new_chromo[i].fitness
          best_weights = new_chromo[i].weights

      pop.replace(new_chromo)
      fitness = [chrom.score for chrom in pop.chromosomes]
      print("Current population fitness:", fitness)
      print(f"Best fitness so far: {best_fitness}")
      print(f"Best weights: {best_weights}")

  return best_weights


if __name__ == "__main__":
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

  if (args.train):
    best_weights = main(args.no_show)
    print("\nTraining completed!")
    print(f"Final best weights: {best_weights}")

  elif (args.game):
    game.MANUAL_GAME = True
    game.main()

  else:
    optimal_weights = [-0.97, 5.47, -13.74, -0.73, 7.99, -0.86, -0.72]
    chromo = ga.Chromosome(optimal_weights)
    ai.run_game(chromo, speed=50, max_score=200000, no_show=False)
