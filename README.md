# CS181-Project: Halma AI

### Description
This project implements the game of Halma with a graphical user interface (GUI) and various AI agents. Players can compete against each other or against different AI opponents. The project also includes scripts for automated game testing and parameter analysis. Halma is a strategy board game where players move their pieces to the opposite corner of the board.

### Features
-   Playable Halma game with a GUI.
-   Supports 2-player and 4-player games.
-   Two game modes: 'classic' (reach the target corner) and 'score' (based on piece advancement).
-   Configurable board sizes (4x4, 8x8, 10x10, 12x12).
-   A variety of AI agents with different strategies.
-   Scripts for running multiple games headlessly for AI evaluation and testing.
-   Tools for analyzing game parameters like `jump_scalar`.

### Player Types
The following player type codes are used in the command-line arguments:
-   `H`: Human Player (GUI interaction)
-   `M`: Minimax AI
-   `MLS`: Minimax AI with Local Search optimization
-   `G`: Greedy AI
-   `R`: Random AI
-   `AQL`: Approximate Q-Learning AI
-   `MCTS`: Monte Carlo Tree Search AI
-   `NAQL`: Neural Approximate Q-Learning AI (requires model files)

### Usage

#### 1. Running a Game with GUI (`game.py`)
This script launches the Halma game with a graphical interface.

**Basic command:**
```shell
python src/game.py
```

**Key command-line options for `game.py`:**
-   `-s, --boardsize {4,8,10,12}`: Board size (default: 8).
-   `-m, --mode {classic,score}`: Game mode (default: classic).
-   `-n, --numplayers {2,4}`: Number of players (default: 2).
-   `--max_turns <int>`: Maximum turns per game before draw/score evaluation (default: 500).
-   `-p1, --player1 {H,M,MLS,G,R,AQL,MCTS,NAQL}`: Player 1 (RED) type (default: H).
-   `--p1_model <path>`: Path to Player 1's NAQL model file.
-   `-p2, --player2 {H,M,MLS,G,R,AQL,MCTS,NAQL}`: Player 2 (GREEN) type (default: H).
-   `--p2_model <path>`: Path to Player 2's NAQL model file.
-   `-p3, --player3 {H,M,MLS,G,R,AQL,MCTS,NAQL}`: Player 3 (BLUE) type for 4-player games (default: H).
-   `--p3_model <path>`: Path to Player 3's NAQL model file.
-   `-p4, --player4 {H,M,MLS,G,R,AQL,MCTS,NAQL}`: Player 4 (YELLOW) type for 4-player games (default: H).
-   `--p4_model <path>`: Path to Player 4's NAQL model file.

**Example:**
```shell
python src/game.py -s 8 -n 2 -p1 M -p2 NAQL --p2_model "path/to/green_model.pth"
```

#### 2. Automated Headless Game Evaluation (`autograder.py`)
This script runs multiple games headlessly between specified AI agents to evaluate their performance. Note: The current command-line interface for `autograder.py` only supports configuring two players (P1 RED, P2 GREEN).

**Basic command:**
```shell
python src/autograder.py -p1 M -p2 G -r 20
```

**Key command-line options for `autograder.py`:**
-   `-r, --repetitions <int>`: Number of games to run (default: 10).
-   `-s, --boardsize {4,8,10,12}`: Board size (default: 8).
-   `-m, --mode {classic,score}`: Game mode (default: classic).
-   `--max_turns <int>`: Maximum turns per game before draw (default: 500).
-   `-p1, --player1_type {M,MLS,G,R,MCTS,AQL,NAQL}`: Player 1 (RED) type (default: G).
-   `--p1_model <path>`: Path to Player 1's NAQL model file.
-   `-p2, --player2_type {M,MLS,G,R,MCTS,AQL,NAQL}`: Player 2 (GREEN) type (default: G).
-   `--p2_model <path>`: Path to Player 2's NAQL model file.

#### 3. Miscellaneous Tests (`misc.py`)
This script is used for specific tests, such as analyzing the impact of the `jump_scalar` value. It runs 2-player games headlessly.

**Basic command:**
```shell
python src/misc.py -p1 G -p2 M --jump_score_mode search
```

**Key command-line options for `misc.py`:**
-   `-p1, --player1_type {M,MLS,G}`: Player 1 (RED) type (default: G).
-   `-p2, --player2_type {M,MLS,G}`: Player 2 (GREEN) type (default: G).
-   `-r, --repetitions <int>`: Number of games per test phase (default: from `utils.DEFAULT_ROUND`).
-   `-s, --boardsize {4,8,10,12}`: Board size (default: 8).
-   `-m, --mode {classic,score}`: Game mode (default: score).
-   `--max_turns <int>`: Maximum turns per game (default: from `utils.MAX_TURNS`).
-   `--jump_score_mode {list,search}`: Mode for testing `jump_scalar` (default: search).
    -   `list`: Iterates through predefined `jump_scalars` in `utils.py`.
    -   `search`: Performs a binary search to find an optimal `jump_scalar`.

#### Getting Full Help
For a complete list of command-line options for any script, run:
```shell
python src/game.py --help
python src/autograder.py --help
python src/misc.py --help
```

### Credits
-   The GUI and framework are based on [indrafnugroho/halma](https://github.com/indrafnugroho/halma).
