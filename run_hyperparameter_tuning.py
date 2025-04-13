#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run hyperparameter tuning experiments.
You can specify the type of experiment to run via command line arguments.

Usage:
    python run_hyperparameter_tuning.py --experiment=learning_rate
    python run_hyperparameter_tuning.py --experiment=all
"""

import sys
import argparse
from experiments.hyperparameter_tuning import (
    run_learning_rate_experiment,
    run_weight_decay_experiment,
    run_neurons_experiment,
    run_layers_experiment,
    run_batch_size_experiment,
    run_activation_experiment,
    run_momentum_experiment,
    run_dropout_experiment,
    run_all_hyperparameter_experiments,
    parse_arguments
)


def main():
    """Run hyperparameter tuning experiments"""
    # If no arguments are provided, print help information
    if len(sys.argv) == 1:
        print("Please provide an experiment type parameter. Example: --experiment=learning_rate")
        print("Available experiment types:")
        print("  - learning_rate: Learning rate experiment")
        print("  - weight_decay: Weight decay experiment")
        print("  - neurons: Number of neurons experiment")
        print("  - layers: Number of layers experiment")
        print("  - batch_size: Batch size experiment")
        print("  - activation: Activation function experiment")
        print("  - momentum: Momentum experiment")
        print("  - dropout_rate: Dropout rate experiment")
        print("  - all: Run all experiments")
        print("\nExamples:")
        print("  python run_hyperparameter_tuning.py --experiment=learning_rate")
        print("  python run_hyperparameter_tuning.py --experiment=all --epochs=30")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the specified experiment
    if args.experiment == 'all':
        run_all_hyperparameter_experiments(args)
    elif args.experiment == 'learning_rate':
        run_learning_rate_experiment(args)
    elif args.experiment == 'weight_decay':
        run_weight_decay_experiment(args)
    elif args.experiment == 'neurons':
        run_neurons_experiment(args)
    elif args.experiment == 'layers':
        run_layers_experiment(args)
    elif args.experiment == 'batch_size':
        run_batch_size_experiment(args)
    elif args.experiment == 'activation':
        run_activation_experiment(args)
    elif args.experiment == 'momentum':
        run_momentum_experiment(args)
    elif args.experiment == 'dropout_rate':
        run_dropout_experiment(args)
    else:
        print(f"Error: Unknown experiment type '{args.experiment}'")
        sys.exit(1)


if __name__ == '__main__':
    main() 