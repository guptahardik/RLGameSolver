# Reinforcement Learning: Solving Two Player Games

## Overview

This project explores the application of reinforcement learning (RL) techniques to solve two-person zero-sum games. We investigate the challenges posed by the dynamic and strategic nature of such games, where one player's gain is directly tied to the other's loss.

## Table of Contents

1. [Background](#background)
2. [Implementation](#implementation)
3. [Results](#results)
4. [Challenges and Limitations](#challenges-and-limitations)
5. [Getting Started](#getting-started)
6. [Authors](#authors)

## Background

Reinforcement learning provides a flexible framework for modeling and solving non-zero-sum games by allowing agents to learn and adapt their strategies over time based on feedback from the environment. This project focuses on:

- Long Short-Term Memory (LSTM) networks
- Non-Linear Programming (NLP)
- Exploration-exploitation tradeoffs

## Implementation

We implemented two main scenarios:

1. **Agent vs. Linear Solver**: One machine learning agent plays against a computer using the Simplex Algorithm in an OpenAI Gym environment.
2. **Agent vs. Agent**: Two machine learning agents compete, each using an LSTM brain model to learn and predict optimal mixed strategies.

## Results

Our project demonstrated the effectiveness of RL in solving various two-player zero-sum games:

1. **Rock, Paper, Scissors**: Agents converged to a stable Nash equilibrium.
2. **Colonel Blotto**: Agents learned optimal resource allocation strategies across multiple battlefields.
3. **Random Payoff Matrix**: The model showed convergence for various payoff matrix configurations.

## Challenges and Limitations

1. Non-Convexity
2. Dynamic Environments
3. Computational Complexity
4. Exploration in Stateless Systems

## Getting Started

To run this project:

1. Clone the repository:
2. Install the required dependencies (list them here or refer to a requirements.txt file)
3. Run the main script (provide the command to run the project)

## Authors

- Pratim Chowdhary - Dartmouth College
- Hardik Gupta - Thayer School of Engineering at Dartmouth

For more detailed information, please refer to the [presentation slides](https://github.com/guptahardik/RLGameSolver/blob/main/Final%20Presentation.pptx) and the [research paper](https://github.com/guptahardik/RLGameSolver/blob/main/paper.pdf).
