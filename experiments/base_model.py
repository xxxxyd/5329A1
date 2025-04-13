import os
import numpy as np
import time
import argparse
import json

from modules.model import create_mlp
from modules.loss import get_loss
from modules.optimizers import get_optimizer
from utils.data_loader import load_data, split_train_val
from utils.preprocessing import standardize_features, one_hot_encode
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_precision_recall_f1
)


def run_base_model_experiment(args):
    """
    Run base model experiment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "="*50)
    print("Running base model experiment")
    print("="*50)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results', args.name)
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
    
    # Create model
    print("Creating model...")
    model = create_mlp(
        input_dim=train_data.shape[1],
        hidden_dims=[args.n_neurons] * args.n_layers,
        output_dim=args.num_classes,
        activation_name=args.activation,
        use_batch_norm=args.use_batch_norm,
        use_dropout=args.use_dropout,
        dropout_rate=args.dropout_rate,
        weight_scale=args.weight_scale
    )
    
    # Compile model
    loss = get_loss(args.loss)
    optimizer = get_optimizer(
        args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    model.compile(loss=loss, optimizer=optimizer)
    
    # Print model details
    print(f"Model architecture: {args.n_layers} hidden layers with {args.n_neurons} neurons each")
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
        use_learning_rate_scheduler=args.use_lr_scheduler,
        use_gradient_clipping=args.use_gradient_clipping,
        clip_value=args.clip_value,
        save_best=True
    )
    training_time = time.time() - start_time
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(test_data_std, test_labels_one_hot)
    
    # Print results
    print("\nTest results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Training time: {training_time:.2f}s")
    
    # Save results
    print("\nSaving results...")
    # Save model
    model.save_weights(os.path.join(results_dir, 'model_weights.npy'))
    
    # Save metrics
    results = {
        'test_loss': float(metrics['loss']),
        'test_accuracy': float(metrics['accuracy']),
        'test_precision': float(metrics['avg_precision']),
        'test_recall': float(metrics['avg_recall']),
        'test_f1_score': float(metrics['avg_f1_score']),
        'training_time': float(training_time),
        'num_parameters': int(model.count_parameters()),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'args': vars(args)
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    # Plot training history
    plot_training_history(history, save_path=os.path.join(results_dir, 'training_history.png'))
    
    # Plot confusion matrix
    y_true = np.argmax(test_labels_one_hot, axis=1)
    y_pred = model.predict(test_data_std)
    plot_confusion_matrix(y_true, y_pred, args.num_classes, save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Plot class accuracy
    plot_class_accuracy(metrics, args.num_classes, save_path=os.path.join(results_dir, 'class_accuracy.png'))
    
    # Plot precision, recall, and F1 score
    plot_precision_recall_f1(metrics, args.num_classes, save_path=os.path.join(results_dir, 'precision_recall_f1.png'))
    
    print("\nBase model experiment completed.")
    return results


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Command-line arguments
    """
    parser = argparse.ArgumentParser(description='Base model experiment')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    # Model arguments
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--use_dropout', action='store_true', help='Use dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_scale', type=float, default=0.01, help='Weight initialization scale')
    
    # Training arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum (for SGD)')
    parser.add_argument('--loss', type=str, default='softmax_cross_entropy', help='Loss function')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--use_gradient_clipping', action='store_true', help='Use gradient clipping')
    parser.add_argument('--clip_value', type=float, default=5.0, help='Gradient clipping value')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--name', type=str, default='base_model', help='Experiment name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_base_model_experiment(args)
