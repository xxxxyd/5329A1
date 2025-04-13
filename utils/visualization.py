import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy.
    
    Parameters:
    -----------
    history : dict
        Dictionary with training history
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training accuracy')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, num_classes=10, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    num_classes : int
        Number of classes
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentage
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm_percent, annot=True, fmt='.1f',
                cmap='Blues', cbar=True,
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    
    # Set labels
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix (%)')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_class_accuracy(metrics, num_classes=10, save_path=None):
    """
    Plot accuracy for each class.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with evaluation metrics
    num_classes : int
        Number of classes
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Get confusion matrix
    cm = metrics['confusion_matrix']
    
    # Compute accuracy for each class
    class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        class_acc[i] = cm[i, i] / np.sum(cm[i, :]) * 100
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot class accuracy
    plt.bar(range(num_classes), class_acc)
    plt.axhline(y=np.mean(class_acc), color='r', linestyle='--', label=f'Mean: {np.mean(class_acc):.1f}%')
    
    # Set labels
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Class')
    plt.xticks(range(num_classes))
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_f1(metrics, num_classes=10, save_path=None):
    """
    Plot precision, recall, and F1 score for each class.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with evaluation metrics
    num_classes : int
        Number of classes
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Get metrics
    precision = [metrics['precision'][i] for i in range(num_classes)]
    recall = [metrics['recall'][i] for i in range(num_classes)]
    f1_score = [metrics['f1_score'][i] for i in range(num_classes)]
    
    # Plot metrics
    x = np.arange(num_classes)
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1 Score')
    
    # Set labels
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score by Class')
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_ablation_study(results, save_path=None):
    """
    Plot ablation study results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with ablation study results
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Get configurations and metrics
    configs = list(results.keys())
    accuracies = [results[config]['accuracy'] * 100 for config in configs]
    
    # Sort by accuracy
    indices = np.argsort(accuracies)
    configs = [configs[i] for i in indices]
    accuracies = [accuracies[i] for i in indices]
    
    # Plot results
    bars = plt.barh(configs, accuracies)
    
    # Add values
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{accuracies[i]:.1f}%', ha='left', va='center')
    
    # Set labels
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Configuration')
    plt.title('Ablation Study Results')
    plt.grid(axis='x')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_hyperparameter_comparison(param_values, accuracies, param_name, save_path=None):
    """
    Plot hyperparameter comparison.
    
    Parameters:
    -----------
    param_values : list
        List of parameter values
    accuracies : list
        List of corresponding accuracies
    param_name : str
        Name of the parameter
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot results
    plt.plot(param_values, accuracies, 'o-')
    
    # Mark the best value
    best_idx = np.argmax(accuracies)
    best_value = param_values[best_idx]
    best_accuracy = accuracies[best_idx]
    plt.plot(best_value, best_accuracy, 'ro', markersize=10, label=f'Best: {best_value}')
    
    # Set labels
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Effect of {param_name} on Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_network_structure_heatmap(results, save_path=None):
    """
    Plot network structure heatmap.
    
    Parameters:
    -----------
    results : dict
        Dictionary with network structure results
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Get dimensions and accuracies
    depths = sorted(set([config[0] for config in results.keys()]))
    widths = sorted(set([config[1] for config in results.keys()]))
    
    # Create matrix
    matrix = np.zeros((len(widths), len(depths)))
    
    # Fill matrix
    for i, width in enumerate(widths):
        for j, depth in enumerate(depths):
            matrix[i, j] = results.get((depth, width), {}).get('accuracy', 0) * 100
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='viridis',
                xticklabels=depths, yticklabels=widths)
    
    # Set labels
    plt.xlabel('Depth (Number of Hidden Layers)')
    plt.ylabel('Width (Neurons per Hidden Layer)')
    plt.title('Network Structure Comparison (Accuracy %)')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_convergence_speed(results, save_path=None):
    """
    Plot convergence speed.
    
    Parameters:
    -----------
    results : dict
        Dictionary with convergence speed results
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get configurations and metrics
    configs = list(results.keys())
    speeds = [results[config]['convergence_speed'] for config in configs]
    
    # Sort by speed
    indices = np.argsort(speeds)
    configs = [configs[i] for i in indices]
    speeds = [speeds[i] for i in indices]
    
    # Plot results
    bars = plt.barh(configs, speeds)
    
    # Add values
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{speeds[i]:.1f} epochs', ha='left', va='center')
    
    # Set labels
    plt.xlabel('Convergence Speed (epochs)')
    plt.ylabel('Configuration')
    plt.title('Convergence Speed Comparison')
    plt.grid(axis='x')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_comparison_bar_chart(results, metric='accuracy', title=None, save_path=None):
    """
    Plot comparison bar chart.
    
    Parameters:
    -----------
    results : dict
        Dictionary with comparison results
    metric : str
        Metric to compare
    title : str or None
        Title of the plot
    save_path : str or None
        Path to save the plot, if None, the plot is shown
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get configurations and metrics
    configs = list(results.keys())
    values = [results[config][metric] * 100 if metric in ['accuracy', 'precision', 'recall', 'f1_score'] else results[config][metric] for config in configs]
    
    # Sort by value
    indices = np.argsort(values)
    configs = [configs[i] for i in indices]
    values = [values[i] for i in indices]
    
    # Plot results
    bars = plt.barh(configs, values)
    
    # Add values
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{values[i]:.1f}{"%" if metric in ["accuracy", "precision", "recall", "f1_score"] else ""}',
                 ha='left', va='center')
    
    # Set labels
    plt.xlabel(f'{metric.capitalize()}{"(%)" if metric in ["accuracy", "precision", "recall", "f1_score"] else ""}')
    plt.ylabel('Configuration')
    plt.title(title or f'{metric.capitalize()} Comparison')
    plt.grid(axis='x')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
