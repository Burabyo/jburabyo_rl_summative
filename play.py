"""Run the best-performing trained agent in the 3D simulation.

    uv run play.py                 # overall best agent
    uv run play.py --algo dqn      # best DQN agent
"""
import sys

from main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "play", *sys.argv[1:]]
    main()
