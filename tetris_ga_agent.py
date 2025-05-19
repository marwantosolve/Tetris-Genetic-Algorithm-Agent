"""
A Genetic Algorithm implementation for training an AI agent to play Tetris.

This module implements a genetic algorithm to evolve weights for a Tetris-playing AI.
The AI evaluates possible moves based on seven features:
1. Total column height (negative weight)
2. Lines cleared (positive weight)
3. Number of holes (negative weight)
4. Blocks above holes (negative weight)
5. Piece contact with other blocks (positive weight)
6. Contact with floor (positive weight)
7. Contact with wall (mixed weight)

Key Components:
- Genetic Algorithm parameters (population size, generations, mutation rates, etc.)
- Game mechanics integration with the AI decision-making process
- Tournament selection for parent chromosomes
- Uniform crossover for breeding
- Mutation operator for genetic diversity
- Visualization of gameplay and training progress
- Comprehensive logging of training results

Classes:
    None (functional programming approach)

Functions:
    play_and_score(weights): Evaluates a chromosome by playing a game
    pick_best_move(board, piece, holes, blockers, weights): Determines optimal move
    run_genetic_algorithm(): Main GA loop
    select_by_tournament(population, fitness): Tournament selection
    do_crossover(parent1, parent2): Uniform crossover
    apply_mutation(candidate): Random weight adjustment
    play_with_weights(weights, max_pieces): Visual gameplay
    main_menu(): User interface
    show_training_screen(): Training progress display
    main(): Program entry point

Dependencies:
    - pygame: For visualization
    - numpy: For numerical operations
    - matplotlib: For plotting training progress
    - tetris_base: Base Tetris game mechanics

Usage:
    Run main() to start the program and choose between:
    1. Playing with trained AI
    2. Training new AI weights
    3. Playing manual Tetris

Author: [marwantosolve]
Date: [Last Edit: 20.05.2025 - 1:20:12 AM] 
"""

# --- Import necessary libraries ---
import tetris_base as tb
from tetris_base import *
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import pygame
import sys
from pygame.locals import *

# --- Genetic Algorithm Parameters ---
POP_SIZE = 15        # Size of each generation's population
GENS = 10             # Number of generations to evolve
TOURNEY_SIZE = 2     # How many chromosomes compete in tournament selection
MUTATE_CHANCE = 0.1  # Probability of mutation for each gene (0-1)
CROSSOVER_CHANCE = 0.7  # Probability of crossover between parents (0-1) 
ELITE_COUNT = 2      # Number of best chromosomes kept for next generation
PIECE_LIMIT = 300    # Maximum Tetris pieces per game during training
AI_FALL_SPEED = 0.0000001  # Speed of piece falling when AI plays (smaller = faster)


def play_and_score(weights):
  """
  Play a game of Tetris using the provided weights and return the score and number of pieces used.
  """
  board = get_blank_board()
  total_score = 0
  current_level = 1
  falling_piece = get_new_piece()
  next_piece = get_new_piece()
  is_game_over = False
  pieces_used = 0

  while not is_game_over and pieces_used < PIECE_LIMIT:
    holes, blockers = calc_initial_move_info(board)
    move = pick_best_move(board, falling_piece, holes, blockers, weights)
    if move is None:
      is_game_over = True
      continue
    x_pos, rotation = move
    falling_piece['rotation'] = rotation
    falling_piece['x'] = x_pos
    while is_valid_position(board, falling_piece, adj_Y=1):
      falling_piece['y'] += 1
    add_to_board(board, falling_piece)
    pieces_used += 1
    lines_cleared = remove_complete_lines(board)
    if lines_cleared == 1:
      total_score += 40 * current_level
    elif lines_cleared == 2:
      total_score += 100 * current_level
    elif lines_cleared == 3:
      total_score += 300 * current_level
    elif lines_cleared == 4:
      total_score += 1200 * current_level
    current_level, _ = calc_level_and_fall_freq(total_score)
    falling_piece = next_piece
    next_piece = get_new_piece()
    if not is_valid_position(board, falling_piece):
      is_game_over = True
  return total_score, pieces_used


def pick_best_move(board, piece, holes, blockers, weights):
  """
  Try all possible moves for the current piece and pick the one with the best weighted score.
  Returns (x, rotation) or None if no valid move exists.
  """
  top_score = float('-inf')
  chosen_move = None
  for rotation in range(len(PIECES[piece['shape']])):
    for x in range(-2, BOARDWIDTH - 2):
      test_piece = copy.deepcopy(piece)
      move_stats = calc_move_info(
          board, test_piece, x, rotation, holes, blockers)
      if move_stats[0]:
        # move_stats: [valid, max_height, lines_cleared, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
        score = (
            weights[0] * move_stats[1] +
            weights[1] * move_stats[2] +
            weights[2] * move_stats[3] +
            weights[3] * move_stats[4] +
            weights[4] * move_stats[5] +
            weights[5] * move_stats[6] +
            weights[6] * move_stats[7]
        )
        if score > top_score:
          top_score = score
          chosen_move = (x, rotation)
  return chosen_move


def run_genetic_algorithm():
  """
  Main loop for the genetic algorithm. Evolves weights for the Tetris evaluation function.
  """
  # --- Initial Population ---
  population = []
  best_scores = []
  avg_scores = []
  best_per_gen = []
  all_generations_log = []  # To store all chromosomes' info per generation
  print("Creating initial population...")
  for _ in range(POP_SIZE):
    # Randomize weights for each feature
    weights = [
        -random.uniform(0.1, 1),     # Height (negative)
        random.uniform(0.1, 1),      # Lines cleared (positive)
        -random.uniform(0.1, 1),     # Holes (negative)
        -random.uniform(0.1, 1),     # Blocking blocks (negative)
        random.uniform(0.1, 1),      # Piece sides (positive)
        random.uniform(0.1, 1),      # Floor sides (positive)
        random.uniform(-0.5, 0.5),   # Wall sides (mixed)
    ]
    population.append(weights)
  # --- Evolution Loop ---
  for gen in range(GENS):
    print(f"Generation {gen+1}/{GENS}")
    fitness = []
    pieces_used = []
    gen_chromosomes_log = []  # Log all chromosomes for this generation
    for idx, chromosome in enumerate(population):
      score, pieces = play_and_score(chromosome)
      fitness.append(score)
      pieces_used.append(pieces)
      print(
          f"Chromosome score: {score}, Pieces: {pieces}, Weights: {[round(w, 2) for w in chromosome]}")
      gen_chromosomes_log.append({
          'index': idx,
          'score': score,
          'pieces': pieces,
          'weights': [round(w, 4) for w in chromosome]
      })
    all_generations_log.append({
        'generation': gen + 1,
        'chromosomes': gen_chromosomes_log
    })
    best_score = max(fitness)
    avg_score = sum(fitness) / len(fitness)
    best_scores.append(best_score)
    avg_scores.append(avg_score)
    # Find best and second best for logging
    sorted_idx = sorted(range(len(fitness)),
                        key=lambda i: fitness[i], reverse=True)
    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    gen_log = {
        "generation": gen + 1,
        "best": {
            "weights": [round(w, 4) for w in population[best_idx]],
            "score": fitness[best_idx],
            "pieces_played": pieces_used[best_idx]
        },
        "second_best": {
            "weights": [round(w, 4) for w in population[second_idx]],
            "score": fitness[second_idx],
            "pieces_played": pieces_used[second_idx]
        }
    }
    best_per_gen.append(gen_log)
    print(f"Best score this gen: {best_score}")
    print(f"Best weights: {[round(w, 4) for w in population[best_idx]]}")
    # --- New Generation ---
    next_gen = []
    # Elitism: keep top performers
    for i in range(ELITE_COUNT):
      next_gen.append(population[sorted_idx[i]])
    # Fill up the rest of the population
    while len(next_gen) < POP_SIZE:
      parent_a = select_by_tournament(population, fitness)
      parent_b = select_by_tournament(population, fitness)
      if random.random() < CROSSOVER_CHANCE:
        child1, child2 = do_crossover(parent_a, parent_b)
      else:
        child1, child2 = parent_a.copy(), parent_b.copy()
      apply_mutation(child1)
      apply_mutation(child2)
      next_gen.append(child1)
      if len(next_gen) < POP_SIZE:
        next_gen.append(child2)
    population = next_gen
  # --- Final Evaluation ---
  final_scores = []
  final_pieces = []
  for chromosome in population:
    score, pieces = play_and_score(chromosome)
    final_scores.append(score)
    final_pieces.append(pieces)
  best_idx = final_scores.index(max(final_scores))
  best_weights = population[best_idx]
  best_score = final_scores[best_idx]
  best_pieces = final_pieces[best_idx]
  print("\nGenetic Algorithm finished!")
  print(f"Best weights: {[round(w, 4) for w in best_weights]}")
  print(f"Best score: {best_score} with {best_pieces} pieces used")
  # --- Find best and second-best overall across all generations ---
  all_chromosomes = []
  for gen_log in all_generations_log:
    for chromo in gen_log['chromosomes']:
      all_chromosomes.append({
          'generation': gen_log['generation'],
          **chromo
      })
  all_chromosomes_sorted = sorted(
      all_chromosomes, key=lambda c: c['score'], reverse=True)
  best_overall = all_chromosomes_sorted[0]
  second_overall = all_chromosomes_sorted[1]
  # --- Plotting ---
  plt.figure(figsize=(10, 6))
  plt.plot(range(1, GENS + 1), best_scores, label='Best Score')
  plt.plot(range(1, GENS + 1), avg_scores, label='Average Score')
  plt.xlabel('Generation')
  plt.ylabel('Score')
  plt.title('Genetic Algorithm Progress')
  plt.legend()
  plt.grid(True)
  plt.savefig('fitness_evolution.png')
  plt.show()
  # --- Save Results ---
  with open('chromosomes_log.txt', 'w') as f:
    f.write("# Tetris Genetic Algorithm Log\n\n")
    f.write("## All Chromosomes Per Generation\n\n")
    for gen_log in all_generations_log:
      f.write(f"Generation {gen_log['generation']}\n")
      for chromo in gen_log['chromosomes']:
        f.write(
            f"  Chromosome {chromo['index']}: Score={chromo['score']}, Pieces={chromo['pieces']}, Weights={chromo['weights']}\n")
      f.write("\n")
    f.write("## Best and Second-Best Chromosome Per Generation\n\n")
    for gen_data in best_per_gen:
      f.write(f"Generation {gen_data['generation']}:\n")
      f.write(
          f"  Best: Score={gen_data['best']['score']}, Pieces={gen_data['best']['pieces_played']}, Weights={gen_data['best']['weights']}\n")
      f.write(
          f"  Second: Score={gen_data['second_best']['score']}, Pieces={gen_data['second_best']['pieces_played']}, Weights={gen_data['second_best']['weights']}\n\n")
    f.write("## Best Chromosome Overall (All Generations)\n\n")
    f.write(
        f"Best Overall: Generation={best_overall['generation']}, Index={best_overall['index']}, Score={best_overall['score']}, Pieces={best_overall['pieces']}, Weights={best_overall['weights']}\n")
    f.write(
        f"Second Best Overall: Generation={second_overall['generation']}, Index={second_overall['index']}, Score={second_overall['score']}, Pieces={second_overall['pieces']}, Weights={second_overall['weights']}\n\n")
  return best_weights


def select_by_tournament(population, fitness):
  """
  Tournament selection: randomly pick a few chromosomes and return the best among them.
  """
  chosen = random.sample(range(len(population)), TOURNEY_SIZE)
  best = chosen[0]
  for idx in chosen:
    if fitness[idx] > fitness[best]:
      best = idx
  return population[best].copy()


def do_crossover(parent1, parent2):
  """
  Uniform crossover: for each gene, randomly pick from one of the parents.
  """
  child1 = []
  child2 = []
  for i in range(len(parent1)):
    if random.random() < 0.5:
      child1.append(parent1[i])
      child2.append(parent2[i])
    else:
      child1.append(parent2[i])
      child2.append(parent1[i])
  return child1, child2


def apply_mutation(candidate):
  """
  Randomly tweak some weights in the candidate solution.
  """
  for i in range(len(candidate)):
    if random.random() < MUTATE_CHANCE:
      if i in [0, 2, 3]:  # Negative weights
        candidate[i] = -random.uniform(0.1, 1)
      elif i in [1, 4, 5]:  # Positive weights
        candidate[i] = random.uniform(0.1, 1)
      else:  # Wall contact (mixed)
        candidate[i] = random.uniform(-0.5, 0.5)


def play_with_weights(weights, max_pieces=PIECE_LIMIT):
  """
  Play a game visually using the provided weights. This is for manual/visual testing.
  """
  global MANUAL_GAME
  old_manual = MANUAL_GAME
  MANUAL_GAME = True
  board = get_blank_board()
  score = 0
  level = 1
  falling_piece = get_new_piece()
  next_piece = get_new_piece()
  pieces_used = 0
  # --- Setup Pygame for visualization ---
  pygame.init()
  FPSCLOCK = pygame.time.Clock()
  DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
  BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
  BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
  pygame.display.set_caption('Tetris AI')
  # Set globals in tetris_base for drawing functions
  tb.DISPLAYSURF = DISPLAYSURF
  tb.BASICFONT = BASICFONT
  tb.BIGFONT = BIGFONT
  tb.FPSCLOCK = FPSCLOCK
  last_fall = time.time()
  fall_speed = AI_FALL_SPEED  # Controlled by constant
  is_game_over = False
  while not is_game_over and pieces_used < max_pieces:
    check_quit()
    for event in pygame.event.get():
      if event.type == KEYUP and event.key == K_ESCAPE:
        pygame.quit()
        return 0, 0  # Always return a tuple to avoid unpacking None
    if falling_piece is None:
      falling_piece = next_piece
      next_piece = get_new_piece()
      last_fall = time.time()
      if not is_valid_position(board, falling_piece):
        is_game_over = True
        continue
    holes, blockers = calc_initial_move_info(board)
    move = pick_best_move(board, falling_piece, holes, blockers, weights)
    if move is None:
      is_game_over = True
      continue
    x_pos, rotation = move
    falling_piece['rotation'] = rotation
    falling_piece['x'] = x_pos
    if time.time() - last_fall > fall_speed:
      if is_valid_position(board, falling_piece, adj_Y=1):
        falling_piece['y'] += 1
        last_fall = time.time()
      else:
        add_to_board(board, falling_piece)
        pieces_used += 1
        lines_cleared = remove_complete_lines(board)
        if lines_cleared == 1:
          score += 40 * level
        elif lines_cleared == 2:
          score += 100 * level
        elif lines_cleared == 3:
          score += 300 * level
        elif lines_cleared == 4:
          score += 1200 * level
        falling_piece = None
    DISPLAYSURF.fill(BGCOLOR)
    draw_board(board)
    draw_status(score, level)
    pieces_surf = BASICFONT.render(
        f'Pieces: {pieces_used}/{max_pieces}', True, TEXTCOLOR)
    pieces_rect = pieces_surf.get_rect()
    pieces_rect.topleft = (WINDOWWIDTH - 150, 180)
    DISPLAYSURF.blit(pieces_surf, pieces_rect)
    if next_piece:
      draw_next_piece(next_piece)
    if falling_piece:
      draw_piece(falling_piece)
    pygame.display.update()
    FPSCLOCK.tick(FPS)
  show_text_screen('Game Over')
  MANUAL_GAME = old_manual
  return score, pieces_used


def main_menu():
  """
  Display a simple Pygame menu for mode selection.
  Returns the selected mode as a string.
  """
  pygame.init()
  screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
  pygame.display.set_caption('Tetris Genetic Algorithm - Main Menu')
  font = pygame.font.Font('freesansbold.ttf', 32)
  small_font = pygame.font.Font('freesansbold.ttf', 20)
  clock = pygame.time.Clock()
  menu_items = [
      '1. Play with AI (Best Weights)',
      '2. Train the AI (Genetic Algorithm)',
      '3. Play Manually (Classic Tetris)',
      'ESC. Quit'
  ]
  selected = 0
  while True:
    screen.fill(BGCOLOR)
    title_surf = font.render('Tetris Genetic Algorithm', True, TEXTCOLOR)
    title_rect = title_surf.get_rect(center=(WINDOWWIDTH//2, 100))
    screen.blit(title_surf, title_rect)
    for i, item in enumerate(menu_items):
      color = TEXTCOLOR if i != selected else (255, 255, 0)
      surf = small_font.render(item, True, color)
      rect = surf.get_rect(center=(WINDOWWIDTH//2, 220 + i*50))
      screen.blit(surf, rect)
    pygame.display.update()
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit()
      elif event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          pygame.quit()
          sys.exit()
        elif event.key in [K_DOWN, K_s]:
          selected = (selected + 1) % len(menu_items)
        elif event.key in [K_UP, K_w]:
          selected = (selected - 1) % len(menu_items)
        elif event.key in [K_RETURN, K_KP_ENTER, K_SPACE]:
          if selected == 0:
            return 'ai_play'
          elif selected == 1:
            return 'train'
          elif selected == 2:
            return 'manual'
          elif selected == 3:
            pygame.quit()
            sys.exit()
        elif event.key == K_1:
          return 'ai_play'
        elif event.key == K_2:
          return 'train'
        elif event.key == K_3:
          return 'manual'
    clock.tick(30)


def show_training_screen():
  """
  Show a simple Pygame screen indicating that training is in progress.
  """
  pygame.init()
  screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
  pygame.display.set_caption('Training Genetic Algorithm...')
  font = pygame.font.Font('freesansbold.ttf', 32)
  small_font = pygame.font.Font('freesansbold.ttf', 20)
  screen.fill(BGCOLOR)
  surf = font.render('Training Genetic Algorithm...', True, TEXTCOLOR)
  rect = surf.get_rect(center=(WINDOWWIDTH//2, WINDOWHEIGHT//2-40))
  screen.blit(surf, rect)
  surf2 = small_font.render(
      'This may take a while. Please wait...', True, TEXTCOLOR)
  rect2 = surf2.get_rect(center=(WINDOWWIDTH//2, WINDOWHEIGHT//2+20))
  screen.blit(surf2, rect2)
  pygame.display.update()


def main():
  """
  Main entry point for the Tetris Genetic Algorithm project.
  Shows a menu and runs the selected mode.
  """
  while True:
    mode = main_menu()
    if mode == 'ai_play':
      # Play with best weights (AI mode)
      # Try to load best weights from file, else use default
      try:
        with open('best_weights.txt', 'r') as f:
          lines = [line.strip()
                   for line in f if not line.startswith('#') and line.strip()]
          best_weights = [float(w) for w in lines]
      except Exception:
        best_weights = [-0.1, 0.9, -0.99, -0.22, 0.61, 0.8, 0.36]
      print("\nPlaying a test game with the best weights (600 pieces limit)...")
      play_with_weights(best_weights, max_pieces=600)
    elif mode == 'train':
      # Show a training screen, then run the genetic algorithm
      show_training_screen()
      best_weights = run_genetic_algorithm()
      # After training, let user play with the new best weights
      play_with_weights(best_weights, max_pieces=600)
    elif mode == 'manual':
      # Play the normal game (manual, no AI)
      # Set MANUAL_GAME True and call tetris_base.main()
      global MANUAL_GAME
      MANUAL_GAME = True
      tb.MANUAL_GAME = True
      tb.main()
    else:
      break


main()