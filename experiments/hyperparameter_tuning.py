import os
import numpy as np
import time
import argparse
import json
import matplotlib.pyplot as plt

from modules.model import create_mlp
from modules.loss import get_loss
from modules.optimizers import get_optimizer
from utils.data_loader import load_data, split_train_val
from utils.preprocessing import standardize_features, one_hot_encode
from utils.visualization import (
    plot_hyperparameter_comparison,
    plot_training_history,
    plot_confusion_matrix,
    plot_network_structure_heatmap
)


def run_hyperparameter_experiment(
    args, param_name, param_values, results_subdir,
    create_model_fn=None, compile_model_fn=None, model_args_fn=None, optimizer_args_fn=None
):
    """
    Run a hyperparameter experiment with varying parameter values.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    param_name : str
        Name of the parameter being tuned
    param_values : list
        List of parameter values to try
    results_subdir : str
        Subdirectory for results
    create_model_fn : function or None
        Function to create model with current hyperparameter value
    compile_model_fn : function or None
        Function to compile model with current hyperparameter value
    model_args_fn : function or None
        Function to get model arguments for current hyperparameter value
    optimizer_args_fn : function or None
        Function to get optimizer arguments for current hyperparameter value
    
    Returns:
    --------
    results : dict
        Dictionary with results for each parameter value
    """
    print("\n" + "="*50)
    print(f"Running {param_name} hyperparameter experiment")
    print("="*50)
    
    # Create results directory
    results_dir = os.path.join('results', 'hyperparameter_tuning', results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load data
    print("\nLoading data...")
    train_data, train_labels, test_data, test_labels = load_data(args.data_dir)
    
    # Preprocess data
    print("Preprocessing data...")
    # Standardize features
    train_data_std, test_data_std, _, _ = standardize_features(train_data, test_data)
    
    # Convert labels to one-hot encoding
    train_labels_one_hot = one_hot_encode(train_labels, args.num_classes)
    test_labels_one_hot = one_hot_encode(test_labels, args.num_classes)
    
    # Split training data into training and validation sets
    train_subset, train_subset_labels, val_subset, val_subset_labels = split_train_val(
        train_data_std, train_labels_one_hot, args.val_ratio, args.seed)
    
    # Run experiments for each parameter value
    results = {}
    
    for param_value in param_values:
        print(f"\n{'-'*50}")
        print(f"Running with {param_name} = {param_value}")
        print(f"{'-'*50}")
        
        # Create model
        print("Creating model...")
        
        if create_model_fn is not None:
            model = create_model_fn(args, param_value, train_data.shape[1])
        else:
            model_args = {}
            if model_args_fn is not None:
                model_args = model_args_fn(args, param_value)
            
            model = create_mlp(
                input_dim=train_data.shape[1],
                hidden_dims=[args.n_neurons] * args.n_layers,
                output_dim=args.num_classes,
                activation_name=args.activation,
                use_batch_norm=args.use_batch_norm,
                use_dropout=args.use_dropout,
                dropout_rate=args.dropout_rate if 'dropout_rate' not in model_args else model_args['dropout_rate'],
                weight_scale=args.weight_scale
            )
        
        # Compile model
        print("Compiling model...")
        
        if compile_model_fn is not None:
            loss, optimizer = compile_model_fn(args, param_value)
        else:
            loss = get_loss(args.loss)
            
            optimizer_args = {
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay
            }
            
            # Add momentum parameter only for SGD with momentum
            if args.optimizer.lower() == 'sgd_momentum':
                optimizer_args['momentum'] = args.momentum
            
            if optimizer_args_fn is not None:
                optimizer_args.update(optimizer_args_fn(args, param_value))
            
            optimizer = get_optimizer(args.optimizer, **optimizer_args)
        
        model.compile(loss=loss, optimizer=optimizer)
        
        # Print model details
        print(f"Model parameter {param_name}: {param_value}")
        print(f"Number of parameters: {model.count_parameters()}")
        
        # Train model
        print("\nTraining model...")
        start_time = time.time()
        
        # Use the param_value for batch_size if that's the parameter we're tuning
        current_batch_size = param_value if param_name == 'batch_size' else args.batch_size
        
        history = model.train(
            x_train=train_subset,
            y_train=train_subset_labels,
            x_val=val_subset,
            y_val=val_subset_labels,
            epochs=args.epochs,
            batch_size=current_batch_size,
            shuffle=True,
            use_early_stopping=args.use_early_stopping,
            patience=args.patience,
            use_gradient_clipping=args.use_gradient_clipping,
            clip_value=args.clip_value,
            save_best=True
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model on test set
        print("\nEvaluating model on test set...")
        metrics = model.evaluate(test_data_std, test_labels_one_hot)
        
        # Calculate convergence speed (epochs to reach 90% of final performance)
        final_val_acc = history['val_acc'][-1]
        threshold = 0.9 * final_val_acc
        convergence_epoch = next((i for i, acc in enumerate(history['val_acc']) if acc >= threshold), len(history['val_acc']))
        
        # Calculate overfitting (difference between training and validation accuracy)
        overfitting = abs(history['train_acc'][-1] - history['val_acc'][-1])
        
        # Print results
        print("\nResults:")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Convergence speed: {convergence_epoch} epochs")
        print(f"Overfitting: {overfitting:.4f}")
        
        # Save model and results
        param_str = str(param_value).replace('.', '_')
        param_dir = os.path.join(results_dir, f"{param_name}_{param_str}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Save model
        model.save_weights(os.path.join(param_dir, 'model_weights.npy'))
        
        # Save results
        param_results = {
            'param_value': float(param_value) if isinstance(param_value, (int, float)) else param_value,
            'test_loss': float(metrics['loss']),
            'test_accuracy': float(metrics['accuracy']),
            'test_precision': float(metrics['avg_precision']),
            'test_recall': float(metrics['avg_recall']),
            'test_f1_score': float(metrics['avg_f1_score']),
            'training_time': float(training_time),
            'num_parameters': int(model.count_parameters()),
            'convergence_speed': int(convergence_epoch),
            'overfitting': float(overfitting),
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']]
            },
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        
        with open(os.path.join(param_dir, 'results.json'), 'w') as f:
            json.dump(param_results, f, indent=2)
        
        # Plot training history
        plot_training_history(history, save_path=os.path.join(param_dir, 'training_history.png'))
        
        # Plot confusion matrix
        y_true = np.argmax(test_labels_one_hot, axis=1)
        y_pred = model.predict(test_data_std)
        plot_confusion_matrix(y_true, y_pred, args.num_classes, save_path=os.path.join(param_dir, 'confusion_matrix.png'))
        
        # Store results for comparison
        results[param_value] = {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'precision': metrics['avg_precision'],
            'recall': metrics['avg_recall'],
            'f1_score': metrics['avg_f1_score'],
            'training_time': training_time,
            'convergence_speed': convergence_epoch,
            'overfitting': overfitting,
            'num_parameters': model.count_parameters(),
            'history': history
        }
    
    # Generate comparative visualizations
    print("\nGenerating comparative visualizations...")
    
    # Extract parameter values and corresponding metrics
    param_values_list = list(results.keys())
    accuracies = [results[val]['accuracy'] * 100 for val in param_values_list]
    losses = [results[val]['loss'] for val in param_values_list]
    convergence_speeds = [results[val]['convergence_speed'] for val in param_values_list]
    overfitting_values = [results[val]['overfitting'] * 100 for val in param_values_list]
    
    # Plot parameter vs. accuracy
    plot_hyperparameter_comparison(param_values_list, accuracies, param_name, 
                                 save_path=os.path.join(results_dir, f'{param_name}_vs_accuracy.png'))
    
    # Plot parameter vs. loss
    plt.figure(figsize=(10, 6))
    plt.plot(param_values_list, losses, 'o-')
    best_idx = np.argmin(losses)
    best_value = param_values_list[best_idx]
    best_loss = losses[best_idx]
    plt.plot(best_value, best_loss, 'ro', markersize=10, label=f'Best: {best_value}')
    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.title(f'Effect of {param_name} on Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{param_name}_vs_loss.png'))
    plt.close()
    
    # Plot parameter vs. convergence speed
    plt.figure(figsize=(10, 6))
    plt.plot(param_values_list, convergence_speeds, 'o-')
    best_idx = np.argmin(convergence_speeds)
    best_value = param_values_list[best_idx]
    best_speed = convergence_speeds[best_idx]
    plt.plot(best_value, best_speed, 'ro', markersize=10, label=f'Best: {best_value}')
    plt.xlabel(param_name)
    plt.ylabel('Convergence Speed (epochs)')
    plt.title(f'Effect of {param_name} on Convergence Speed')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{param_name}_vs_convergence.png'))
    plt.close()
    
    # Plot parameter vs. overfitting
    plt.figure(figsize=(10, 6))
    plt.plot(param_values_list, overfitting_values, 'o-')
    best_idx = np.argmin(overfitting_values)
    best_value = param_values_list[best_idx]
    best_overfitting = overfitting_values[best_idx]
    plt.plot(best_value, best_overfitting, 'ro', markersize=10, label=f'Best: {best_value}')
    plt.xlabel(param_name)
    plt.ylabel('Overfitting (%)')
    plt.title(f'Effect of {param_name} on Overfitting')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{param_name}_vs_overfitting.png'))
    plt.close()
    
    # Plot validation loss curves for all parameter values
    plt.figure(figsize=(12, 8))
    for param_value in param_values_list:
        plt.plot(results[param_value]['history']['val_loss'], label=f'{param_name}={param_value}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss for Different {param_name} Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{param_name}_validation_loss.png'))
    plt.close()
    
    # Plot validation accuracy curves for all parameter values
    plt.figure(figsize=(12, 8))
    for param_value in param_values_list:
        plt.plot(results[param_value]['history']['val_acc'], label=f'{param_name}={param_value}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy for Different {param_name} Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{param_name}_validation_accuracy.png'))
    plt.close()
    
    # Save overall results
    with open(os.path.join(results_dir, 'overall_results.json'), 'w') as f:
        # Create a copy of results without history to avoid large file
        overall_results = {}
        for param_value in param_values_list:
            overall_results[str(param_value)] = {
                'param_value': float(param_value) if isinstance(param_value, (int, float)) else param_value,
                'accuracy': float(results[param_value]['accuracy']),
                'loss': float(results[param_value]['loss']),
                'precision': float(results[param_value]['precision']),
                'recall': float(results[param_value]['recall']),
                'f1_score': float(results[param_value]['f1_score']),
                'training_time': float(results[param_value]['training_time']),
                'convergence_speed': int(results[param_value]['convergence_speed']),
                'overfitting': float(results[param_value]['overfitting']),
                'num_parameters': int(results[param_value]['num_parameters'])
            }
        
        json.dump(overall_results, f, indent=2)
    
    print(f"\n{param_name} hyperparameter experiment completed.")
    
    # Find best parameter value
    best_param_value = param_values_list[np.argmax(accuracies)]
    print(f"Best {param_name} value: {best_param_value}")
    
    return results


def run_learning_rate_experiment(args):
    """
    Run learning rate experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'learning_rate'
    param_values = [0.0001, 0.001, 0.01, 0.05, 0.1]
    results_subdir = 'learning_rate'
    
    def optimizer_args_fn(args, param_value):
        return {'learning_rate': param_value}
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        optimizer_args_fn=optimizer_args_fn
    )


def run_weight_decay_experiment(args):
    """
    Run weight decay experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'weight_decay'
    param_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    results_subdir = 'weight_decay'
    
    def optimizer_args_fn(args, param_value):
        return {'weight_decay': param_value}
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        optimizer_args_fn=optimizer_args_fn
    )


def run_neurons_experiment(args):
    """
    Run experiment varying the number of neurons per layer.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'n_neurons'
    param_values = [32, 64, 128, 256, 512]
    results_subdir = 'neurons'
    
    def create_model_fn(args, param_value, input_dim):
        model = create_mlp(
            input_dim=input_dim,
            hidden_dims=[param_value] * args.n_layers,
            output_dim=args.num_classes,
            activation_name=args.activation,
            use_batch_norm=args.use_batch_norm,
            use_dropout=args.use_dropout,
            dropout_rate=args.dropout_rate,
            weight_scale=args.weight_scale
        )
        return model
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        create_model_fn=create_model_fn
    )


def run_layers_experiment(args):
    """
    Run experiment varying the number of hidden layers.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'n_layers'
    param_values = [1, 2, 3, 4, 5]
    results_subdir = 'layers'
    
    def create_model_fn(args, param_value, input_dim):
        model = create_mlp(
            input_dim=input_dim,
            hidden_dims=[args.n_neurons] * param_value,
            output_dim=args.num_classes,
            activation_name=args.activation,
            use_batch_norm=args.use_batch_norm,
            use_dropout=args.use_dropout,
            dropout_rate=args.dropout_rate,
            weight_scale=args.weight_scale
        )
        return model
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        create_model_fn=create_model_fn
    )


def run_batch_size_experiment(args):
    """
    Run experiment varying the batch size.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'batch_size'
    param_values = [16, 32, 64, 128, 256]
    results_subdir = 'batch_size'
    
    # We will use the default model creation, but modify the batch size during training
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir
    )


def run_activation_experiment(args):
    """
    Run experiment varying the activation function.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'activation'
    param_values = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu']
    results_subdir = 'activation'
    
    def create_model_fn(args, param_value, input_dim):
        model = create_mlp(
            input_dim=input_dim,
            hidden_dims=[args.n_neurons] * args.n_layers,
            output_dim=args.num_classes,
            activation_name=param_value,  # Use the activation function being tested
            use_batch_norm=args.use_batch_norm,
            use_dropout=args.use_dropout,
            dropout_rate=args.dropout_rate,
            weight_scale=args.weight_scale
        )
        return model
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        create_model_fn=create_model_fn
    )


def run_momentum_experiment(args):
    """
    Run momentum experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'momentum'
    param_values = [0.0, 0.5, 0.9, 0.95, 0.99]
    results_subdir = 'momentum'
    
    def optimizer_args_fn(args, param_value):
        return {'momentum': param_value}
    
    def compile_model_fn(args, param_value):
        loss = get_loss(args.loss)
        
        # Use SGD momentum optimizer
        optimizer_name = 'sgd' if param_value == 0.0 else 'sgd_momentum'
        
        if param_value == 0.0:
            # 普通SGD不需要momentum参数
            optimizer = get_optimizer(
                optimizer_name,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            # SGDMomentum需要momentum参数
            optimizer = get_optimizer(
                optimizer_name,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=param_value
            )
        
        return loss, optimizer
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        compile_model_fn=compile_model_fn
    )


def run_dropout_experiment(args):
    """
    Run dropout rate experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    param_name = 'dropout_rate'
    param_values = [0.0, 0.1, 0.3, 0.5, 0.7]
    results_subdir = 'dropout_rate'
    
    def create_model_fn(args, param_value, input_dim):
        # When dropout_rate is 0, don't use dropout
        use_dropout = param_value > 0.0
        
        model = create_mlp(
            input_dim=input_dim,
            hidden_dims=[args.n_neurons] * args.n_layers,
            output_dim=args.num_classes,
            activation_name=args.activation,
            use_batch_norm=args.use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=param_value,
            weight_scale=args.weight_scale
        )
        return model
    
    return run_hyperparameter_experiment(
        args=args,
        param_name=param_name,
        param_values=param_values,
        results_subdir=results_subdir,
        create_model_fn=create_model_fn
    )


def parse_arguments():
    """
    Parse command-line arguments for hyperparameter tuning experiments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning experiments')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of training data to use for validation')
    
    # Model arguments
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--weight_scale', type=float, default=0.01, help='Standard deviation for weight initialization')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--use_dropout', action='store_true', help='Use dropout regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type (sgd, adam, rmsprop)')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--use_gradient_clipping', action='store_true', help='Use gradient clipping')
    parser.add_argument('--clip_value', type=float, default=5.0, help='Gradient clipping value')
    
    # Experiment arguments
    parser.add_argument('--experiment', type=str, required=True, 
                        choices=['learning_rate', 'weight_decay', 'neurons', 'layers', 'batch_size', 'activation', 'momentum', 'dropout_rate', 'all'], 
                        help='Experiment to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def run_all_hyperparameter_experiments(args):
    """
    Run all hyperparameter tuning experiments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "="*50)
    print("Running all hyperparameter tuning experiments")
    print("="*50)
    
    # Create results directory
    os.makedirs(os.path.join('results', 'hyperparameter_tuning'), exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join('results', 'hyperparameter_tuning', 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run learning rate experiment
    print("\nRunning learning rate experiment...")
    lr_results = run_learning_rate_experiment(args)
    
    # Run weight decay experiment
    print("\nRunning weight decay experiment...")
    wd_results = run_weight_decay_experiment(args)
    
    # Run neurons experiment
    print("\nRunning neurons experiment...")
    neurons_results = run_neurons_experiment(args)
    
    # Run layers experiment
    print("\nRunning layers experiment...")
    layers_results = run_layers_experiment(args)
    
    # Run batch size experiment
    print("\nRunning batch size experiment...")
    batch_size_results = run_batch_size_experiment(args)
    
    # Run activation function experiment
    print("\nRunning activation function experiment...")
    activation_results = run_activation_experiment(args)
    
    # Run momentum experiment
    print("\nRunning momentum experiment...")
    momentum_results = run_momentum_experiment(args)
    
    # Run dropout rate experiment
    print("\nRunning dropout rate experiment...")
    dropout_results = run_dropout_experiment(args)
    
    # Aggregate results
    all_results = {
        'learning_rate': {
            'param_name': 'learning_rate',
            'best_value': get_best_param_value(lr_results),
            'results': lr_results
        },
        'weight_decay': {
            'param_name': 'weight_decay',
            'best_value': get_best_param_value(wd_results),
            'results': wd_results
        },
        'neurons': {
            'param_name': 'n_neurons',
            'best_value': get_best_param_value(neurons_results),
            'results': neurons_results
        },
        'layers': {
            'param_name': 'n_layers',
            'best_value': get_best_param_value(layers_results),
            'results': layers_results
        },
        'batch_size': {
            'param_name': 'batch_size',
            'best_value': get_best_param_value(batch_size_results),
            'results': batch_size_results
        },
        'activation': {
            'param_name': 'activation',
            'best_value': get_best_param_value(activation_results),
            'results': activation_results
        },
        'momentum': {
            'param_name': 'momentum',
            'best_value': get_best_param_value(momentum_results),
            'results': momentum_results
        },
        'dropout_rate': {
            'param_name': 'dropout_rate',
            'best_value': get_best_param_value(dropout_results),
            'results': dropout_results
        }
    }
    
    # Create comparative plots
    print("\nGenerating comparative plots...")
    create_comparative_plots(all_results, args)
    
    # Print best parameters
    print("\nHyperparameter Tuning Summary:")
    print("-" * 30)
    print(f"Best learning rate: {all_results['learning_rate']['best_value']}")
    print(f"Best weight decay: {all_results['weight_decay']['best_value']}")
    print(f"Best number of neurons: {all_results['neurons']['best_value']}")
    print(f"Best number of layers: {all_results['layers']['best_value']}")
    print(f"Best batch size: {all_results['batch_size']['best_value']}")
    print(f"Best activation function: {all_results['activation']['best_value']}")
    print(f"Best momentum: {all_results['momentum']['best_value']}")
    print(f"Best dropout rate: {all_results['dropout_rate']['best_value']}")
    
    # Save best parameter configuration
    best_config = {
        'learning_rate': float(all_results['learning_rate']['best_value']),
        'weight_decay': float(all_results['weight_decay']['best_value']),
        'n_neurons': int(all_results['neurons']['best_value']),
        'n_layers': int(all_results['layers']['best_value']),
        'batch_size': int(all_results['batch_size']['best_value']),
        'activation': all_results['activation']['best_value'],
        'momentum': float(all_results['momentum']['best_value']),
        'dropout_rate': float(all_results['dropout_rate']['best_value'])
    }
    
    with open(os.path.join('results', 'hyperparameter_tuning', 'best_parameters.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print("\nAll hyperparameter tuning experiments completed!")
    return all_results


def get_best_param_value(results):
    """
    Get the parameter value with the best performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each parameter value
        
    Returns:
    --------
    best_value : float or str
        Parameter value with best performance
    """
    best_accuracy = -1
    best_value = None
    
    for param_value, metrics in results.items():
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_value = param_value
    
    return best_value


def create_comparative_plots(all_results, args):
    """
    Create comparative plots of different hyperparameter experiment results.
    
    Parameters:
    -----------
    all_results : dict
        Results of all hyperparameter experiments
    args : argparse.Namespace
        Command-line arguments
    """
    results_dir = os.path.join('results', 'hyperparameter_tuning')
    
    # Compare best accuracies across all parameters
    param_names = []
    best_accuracies = []
    
    for param_name, param_data in all_results.items():
        param_names.append(param_name)
        best_param_value = param_data['best_value']
        best_accuracies.append(param_data['results'][best_param_value]['accuracy'] * 100)
    
    plt.figure(figsize=(12, 6))
    plt.bar(param_names, best_accuracies)
    for i, v in enumerate(best_accuracies):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    plt.xlabel('Parameter')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Best Accuracy Comparison Across Hyperparameters')
    plt.ylim(0, max(best_accuracies) + 5)
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, 'best_accuracies_comparison.png'))
    plt.close()
    
    # Compare training times
    param_names = []
    train_times = []
    
    for param_name, param_data in all_results.items():
        param_names.append(param_name)
        best_param_value = param_data['best_value']
        train_times.append(param_data['results'][best_param_value]['training_time'])
    
    plt.figure(figsize=(12, 6))
    plt.bar(param_names, train_times)
    for i, v in enumerate(train_times):
        plt.text(i, v + 0.5, f"{v:.1f}s", ha='center')
    plt.xlabel('Parameter')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison Across Hyperparameters')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, 'training_time_comparison.png'))
    plt.close()


if __name__ == '__main__':
    # Import matplotlib
    import matplotlib.pyplot as plt
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print experiment configuration
    print("\n" + "="*50)
    print("Hyperparameter Tuning Experiment Configuration")
    print("="*50)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.n_layers} layers, {args.n_neurons} neurons, {args.activation} activation")
    print(f"Optimizer: {args.optimizer}, Learning rate: {args.learning_rate}, Weight decay: {args.weight_decay}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("="*50 + "\n")
    
    # Create results directory
    os.makedirs(os.path.join('results', 'hyperparameter_tuning'), exist_ok=True)
    
    # Run experiment
    if args.experiment == 'all':
        run_all_hyperparameter_experiments(args)
    else:
        if args.experiment == 'learning_rate':
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
            
        print("\nExperiment completed!")
