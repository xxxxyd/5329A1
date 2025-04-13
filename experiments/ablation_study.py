import os
import numpy as np
import time
import argparse
import json
from copy import deepcopy

from modules.model import create_mlp
from modules.loss import get_loss
from modules.optimizers import get_optimizer
from utils.data_loader import load_data, split_train_val
from utils.preprocessing import standardize_features, one_hot_encode
from utils.visualization import (
    plot_ablation_study,
    plot_training_history,
    plot_confusion_matrix,
    plot_comparison_bar_chart,
    plot_convergence_speed
)


def run_ablation_study_experiment(args):
    """
    Run ablation study experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "="*50)
    print("Running ablation study experiment")
    print("="*50)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results', 'ablation_study')
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
    
    # Define configurations for ablation study
    configurations = {
        'base': {
            'use_dropout': False,
            'use_batch_norm': False,
            'weight_decay': 0.0001,
            'momentum': 0.0
        },
        'base+dropout': {
            'use_dropout': True,
            'use_batch_norm': False,
            'weight_decay': 0.0001,
            'momentum': 0.0
        },
        'base+high_weight_decay': {
            'use_dropout': False,
            'use_batch_norm': False,
            'weight_decay': 0.001,
            'momentum': 0.0
        },
        'base+momentum': {
            'use_dropout': False,
            'use_batch_norm': False,
            'weight_decay': 0.0001,
            'momentum': 0.9
        },
        'base+batch_norm': {
            'use_dropout': False,
            'use_batch_norm': True,
            'weight_decay': 0.0001,
            'momentum': 0.0
        },
        'full': {
            'use_dropout': True,
            'use_batch_norm': True,
            'weight_decay': 0.001,
            'momentum': 0.9
        },
        'full-dropout': {
            'use_dropout': False,
            'use_batch_norm': True,
            'weight_decay': 0.001,
            'momentum': 0.9
        },
        'full-high_weight_decay': {
            'use_dropout': True,
            'use_batch_norm': True,
            'weight_decay': 0.0001,
            'momentum': 0.9
        },
        'full-momentum': {
            'use_dropout': True,
            'use_batch_norm': True,
            'weight_decay': 0.001,
            'momentum': 0.0
        },
        'full-batch_norm': {
            'use_dropout': True,
            'use_batch_norm': False,
            'weight_decay': 0.001,
            'momentum': 0.9
        }
    }
    
    # Run experiments for each configuration
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n{'-'*50}")
        print(f"Running configuration: {config_name}")
        print(f"{'-'*50}")
        
        # Create model
        print("Creating model...")
        model = create_mlp(
            input_dim=train_data.shape[1],
            hidden_dims=[args.n_neurons] * args.n_layers,
            output_dim=args.num_classes,
            activation_name=args.activation,
            use_batch_norm=config['use_batch_norm'],
            use_dropout=config['use_dropout'],
            dropout_rate=args.dropout_rate,
            weight_scale=args.weight_scale
        )
        
        # Decide on optimizer type based on momentum
        optimizer_name = 'sgd_momentum' if config['momentum'] > 0 else 'sgd'
        
        # Compile model
        loss = get_loss(args.loss)
        
        # 构建优化器参数
        optimizer_args = {
            'learning_rate': args.learning_rate,
            'weight_decay': config['weight_decay']
        }
        
        # 只有在使用sgd_momentum优化器时才添加momentum参数
        if optimizer_name == 'sgd_momentum':
            optimizer_args['momentum'] = config['momentum']
        
        optimizer = get_optimizer(optimizer_name, **optimizer_args)
        model.compile(loss=loss, optimizer=optimizer)
        
        # Print model details
        print(f"Configuration details:")
        print(f"  - Use dropout: {config['use_dropout']}")
        print(f"  - Use batch norm: {config['use_batch_norm']}")
        print(f"  - Weight decay: {config['weight_decay']}")
        print(f"  - Momentum: {config['momentum']}")
        print(f"Number of parameters: {model.count_parameters()}")
        
        # Train model
        print("\nTraining model...")
        start_time = time.time()
        history = model.train(
            x_train=train_subset,
            y_train=train_subset_labels,
            x_val=val_subset,
            y_val=val_subset_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
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
        
        # Calculate convergence speed (epochs to reach 95% of final performance)
        final_val_acc = history['val_acc'][-1]
        threshold = 0.95 * final_val_acc
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
        config_dir = os.path.join(results_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save model
        model.save_weights(os.path.join(config_dir, 'model_weights.npy'))
        
        # Save results
        config_results = {
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
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'config': config
        }
        
        with open(os.path.join(config_dir, 'results.json'), 'w') as f:
            json.dump(config_results, f, indent=2)
        
        # Plot training history
        plot_training_history(history, save_path=os.path.join(config_dir, 'training_history.png'))
        
        # Plot confusion matrix
        y_true = np.argmax(test_labels_one_hot, axis=1)
        y_pred = model.predict(test_data_std)
        plot_confusion_matrix(y_true, y_pred, args.num_classes, save_path=os.path.join(config_dir, 'confusion_matrix.png'))
        
        # Store results for comparison
        results[config_name] = {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'precision': metrics['avg_precision'],
            'recall': metrics['avg_recall'],
            'f1_score': metrics['avg_f1_score'],
            'training_time': training_time,
            'convergence_speed': convergence_epoch,
            'overfitting': overfitting,
            'history': deepcopy(history)
        }
    
    # Generate comparative visualizations
    print("\nGenerating comparative visualizations...")
    
    # Plot ablation study results (test accuracy)
    plot_ablation_study(results, save_path=os.path.join(results_dir, 'ablation_study_accuracy.png'))
    
    # Plot convergence speed comparison
    plot_convergence_speed(results, save_path=os.path.join(results_dir, 'convergence_speed.png'))
    
    # Plot training time comparison
    plot_comparison_bar_chart(results, metric='training_time', title='Training Time Comparison',
                            save_path=os.path.join(results_dir, 'training_time.png'))
    
    # Plot overfitting comparison
    plot_comparison_bar_chart(results, metric='overfitting', title='Overfitting Comparison',
                            save_path=os.path.join(results_dir, 'overfitting.png'))
    
    # Plot validation loss curves for all configurations
    plt_configs = []
    for config_name, result in results.items():
        plt_configs.append((config_name, result['history']['val_loss']))
    
    plt.figure(figsize=(12, 8))
    for config_name, val_loss in plt_configs:
        plt.plot(val_loss, label=config_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss for Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'validation_loss_comparison.png'))
    plt.close()
    
    # Plot validation accuracy curves for all configurations
    plt_configs = []
    for config_name, result in results.items():
        plt_configs.append((config_name, result['history']['val_acc']))
    
    plt.figure(figsize=(12, 8))
    for config_name, val_acc in plt_configs:
        plt.plot(val_acc, label=config_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy for Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'validation_accuracy_comparison.png'))
    plt.close()
    
    # Save overall results
    with open(os.path.join(results_dir, 'overall_results.json'), 'w') as f:
        # Create a copy of results without history to avoid large file
        overall_results = {}
        for config_name, result in results.items():
            overall_results[config_name] = {
                'accuracy': float(result['accuracy']),
                'loss': float(result['loss']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'training_time': float(result['training_time']),
                'convergence_speed': int(result['convergence_speed']),
                'overfitting': float(result['overfitting'])
            }
        
        json.dump(overall_results, f, indent=2)
    
    print("\nAblation study experiment completed.")
    return results


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Command-line arguments
    """
    parser = argparse.ArgumentParser(description='Ablation study experiment')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    # Model arguments
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_scale', type=float, default=0.01, help='Weight initialization scale')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--loss', type=str, default='softmax_cross_entropy', help='Loss function')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use_gradient_clipping', action='store_true', help='Use gradient clipping')
    parser.add_argument('--clip_value', type=float, default=5.0, help='Gradient clipping value')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Import matplotlib only when needed
    import matplotlib.pyplot as plt
    
    args = parse_arguments()
    run_ablation_study_experiment(args)
