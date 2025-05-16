import random
import time
import pygame
import sys
from pygame.locals import *
import tetris_ai.tetris_base as game

# Game window dimensions
size = [640, 480]  # Width: 640px, Height: 480px
# Initialize the Pygame display window
screen = pygame.display.set_mode((size[0], size[1]))


def run_game(chromosome, speed, max_score=20000, no_show=False):
  """
  Run a single game of Tetris using the provided chromosome's decision-making strategy.
  This function serves as the fitness evaluation function for the genetic algorithm.

  Args:
      chromosome: The chromosome (individual) whose performance we're evaluating
      speed (int): Game speed (frames per second)
      max_score (int): Score at which to terminate the game (default: 20000)
      no_show (bool): If True, run without visual display for faster evaluation

  Returns:
      tuple: Game statistics containing:
          - Number of pieces used
          - Count of lines cleared (1-line, 2-line, 3-line, 4-line)
          - Final score
          - Whether max_score was reached (win condition)
  """
  # Set game speed based on parameter
  game.FPS = int(speed)
  game.main()

  # Initialize game state
  board = game.get_blank_board()          # Create empty game board
  last_fall_time = time.time()            # Track last piece fall time
  score = 0                               # Initialize score
  level, fall_freq = game.calc_level_and_fall_freq(
      score)  # Calculate initial level and piece fall speed
  falling_piece = game.get_new_piece()    # Get first falling piece
  next_piece = game.get_new_piece()       # Get next piece in queue

  # Calculate initial best move for first piece
  chromosome.calc_best_move(board, falling_piece)

  # Game statistics tracking
  num_used_pieces = 0                     # Count total pieces used
  # Track lines cleared (1-line, 2-line, 3-line, 4-line)
  removed_lines = [0, 0, 0, 0]

  # Game state flags
  alive = True                            # Whether game is still running
  win = False                             # Whether max_score was reached

  while alive:
    # Handle Pygame events (e.g., window closing)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        print("Game exited by user")
        exit()

    # Handle piece transition (current piece landed)
    if falling_piece == None:
      # Move next piece to current and generate new next piece
      falling_piece = next_piece
      next_piece = game.get_new_piece()

      # Calculate best move for new piece using chromosome's strategy
      chromosome.calc_best_move(board, falling_piece, no_show)

      # Update statistics
      num_used_pieces += 1
      score += 1  # Base score for each piece

      # Reset fall timer
      last_fall_time = time.time()

      # Check if new piece can be placed (game over condition)
      if (not game.is_valid_position(board, falling_piece)):
        alive = False  # Game over if piece can't be placed

    # Handle piece falling
    if no_show or time.time() - last_fall_time > fall_freq:
      # Check if piece can fall further
      if (not game.is_valid_position(board, falling_piece, adj_Y=1)):
        # Land the piece if it can't fall further
        game.add_to_board(board, falling_piece)

        # Check for completed lines and update score
        num_removed_lines = game.remove_complete_lines(board)
        # Score calculation based on number of lines cleared simultaneously
        if (num_removed_lines == 1):
          score += 40              # Single line clear
          removed_lines[0] += 1
        elif (num_removed_lines == 2):
          score += 120             # Double line clear
          removed_lines[1] += 1
        elif (num_removed_lines == 3):
          score += 300             # Triple line clear
          removed_lines[2] += 1
        elif (num_removed_lines == 4):
          score += 1200            # Tetris (4-line clear)
          removed_lines[3] += 1

        falling_piece = None         # Trigger new piece generation
      else:
        # Move piece down if possible
        falling_piece['y'] += 1
        last_fall_time = time.time()

    # Update game display if visualization is enabled
    if (not no_show):
      draw_game_on_screen(board, score, level, next_piece, falling_piece,
                          chromosome)

    # Check win condition (max score reached)
    if (score > max_score):
      alive = False
      win = True

  # Prepare final game statistics
  game_state = [num_used_pieces, removed_lines, score, win]

  return game_state


def draw_game_on_screen(board, score, level, next_piece, falling_piece, chromosome):
  """
  Draw the current game state on the screen using Pygame.

  Args:
      board (list): Current game board state
      score (int): Current game score
      level (int): Current game level
      next_piece (dict): Next piece to fall
      falling_piece (dict): Currently falling piece
      chromosome: Current chromosome (for potential AI visualization)
  """
  # Clear screen with background color
  game.DISPLAYSURF.fill(game.BGCOLOR)

  # Draw game elements
  game.draw_board(board)                # Draw game board with placed pieces
  game.draw_status(score, level)        # Draw score and level information
  game.draw_next_piece(next_piece)      # Draw preview of next piece

  # Draw current falling piece if one exists
  if falling_piece != None:
    game.draw_piece(falling_piece)

  # Update display and maintain frame rate
  pygame.display.update()               # Refresh screen
  game.FPSCLOCK.tick(game.FPS)         # Control game speed
