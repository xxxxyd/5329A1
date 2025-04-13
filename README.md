# Neural Network Hyperparameter Tuning Project

## Project Introduction

This project implements a complete neural network training framework for hyperparameter tuning and model performance evaluation on classification tasks. The project includes various hyperparameter tuning experiments, ablation studies, and best model evaluation, aiming to explore the impact of different hyperparameter configurations on model performance.

## Dataset

The dataset used in this project is contained in the `data` directory, including:
- `train_data.npy`: Training data features
- `train_labels.npy`: Training data labels
- `test_data.npy`: Test data features
- `test_labels.npy`: Test data labels

Detailed information about the dataset:
- **Training data**: Shape (50000, 128), type float64, value range from -23.42 to 25.58
- **Training labels**: Shape (50000, 1), type uint8, containing 10 classes (0-9)
- **Test data**: Shape (10000, 128), type float64, value range from -21.37 to 25.40
- **Test labels**: Shape (10000, 1), type uint8, containing 10 classes (0-9)

The dataset has a balanced class distribution, with 5000 samples per class in the training set and 1000 samples per class in the test set.

### Data Preprocessing

In our experiments, the following preprocessing operations were performed on the data:

1. **Standardization**: In most experiments, we use the `standardize_features` function to standardize features to have zero mean and unit variance. This helps accelerate model convergence and improve performance.
   ```python
   train_data_std, test_data_std, _, _ = standardize_features(train_data, test_data)
   ```

2. **Normalization**: In the best model experiment, we use the `normalize_data` function to normalize features to a specific range (default [0,1]).
   ```python
   train_data_norm, mean, std = normalize_data(train_data)
   test_data_norm = (test_data - mean) / std
   ```

3. **One-hot Encoding**: Convert integer class labels to one-hot encoded vectors for classification model training.
   ```python
   train_labels_one_hot = one_hot_encode(train_labels, num_classes=10)
   ```

4. **Data Splitting**: Further split the training data into training and validation sets for model performance evaluation and early stopping during training.
   ```python
   train_subset, train_subset_labels, val_subset, val_subset_labels = split_train_val(
       train_data_std, train_labels_one_hot, val_ratio=0.1, random_seed=42)
   ```

The dataset contains 10 classes, and you can view dataset information using the following command:

```shell
python simple_view_data.py
```

## Project Structure

```
├── data/                       # Dataset directory
├── experiments/                # Experiment scripts
│   ├── __init__.py
│   ├── ablation_study.py       # Ablation experiments
│   ├── base_model.py           # Base model experiments
│   ├── best_model.py           # Best model experiments
│   └── hyperparameter_tuning.py# Hyperparameter tuning experiments
├── modules/                    # Model components
│   ├── __init__.py
│   ├── activations.py          # Activation functions
│   ├── layers.py               # Network layers
│   ├── loss.py                 # Loss functions
│   ├── model.py                # Model definitions
│   └── optimizers.py           # Optimizers
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_loader.py          # Data loading
│   ├── preprocessing.py        # Data preprocessing
│   └── visualization.py        # Visualization tools
├── results/                    # Experiment results
├── main.py                     # Main script
├── run_hyperparameter_tuning.py# Hyperparameter tuning script
├── collect_results.py          # Results collection script
└── README.md                   # This document
```

## Experimental Approach

### 1. Hyperparameter Tuning Experiments

We conducted the following hyperparameter tuning experiments:

1. **Learning Rate Experiment**: Testing different learning rates and their impact on model performance
   - Test values: 0.0001, 0.001, 0.01, 0.05, 0.1

2. **Weight Decay Experiment**: Exploring different L2 regularization strengths and their effect on the model
   - Test values: 0.0001, 0.0005, 0.001, 0.005, 0.01

3. **Neurons Experiment**: Studying the impact of hidden layer neuron count on model capacity
   - Test values: 32, 64, 128, 256, 512

4. **Hidden Layers Experiment**: Evaluating the effect of network depth on model performance
   - Test values: 1, 2, 3, 4, 5

5. **Batch Size Experiment**: Exploring different batch sizes and their impact on training stability and efficiency
   - Test values: 16, 32, 64, 128, 256

6. **Activation Function Experiment**: Comparing the performance of different activation functions
   - Test functions: relu, leaky_relu, sigmoid, tanh, elu

7. **Momentum Experiment**: Testing different momentum values and their effect on the optimization process
   - Test values: 0.0, 0.5, 0.9, 0.95, 0.99

8. **Dropout Rate Experiment**: Studying different dropout rates and their impact on model generalization
   - Test values: 0.0, 0.1, 0.3, 0.5, 0.7

### 2. Ablation Study

The ablation study evaluates the contribution of each key component to overall performance by removing or replacing model components. We tested the following configurations:

1. **Base model (base)**: Simplest network structure without regularization techniques
2. **Base model + high weight decay (base+high_weight_decay)**: Adding weight decay
3. **Base model + momentum (base+momentum)**: Adding optimizer momentum
4. **Base model + batch normalization (base+batch_norm)**: Adding batch normalization layers
5. **Base model + Dropout (base+dropout)**: Adding Dropout layers
6. **Full model (full)**: Complete model with all components
7. **Full model - batch normalization (full-batch_norm)**: Removing batch normalization
8. **Full model - Dropout (full-dropout)**: Removing Dropout
9. **Full model - high weight decay (full-high_weight_decay)**: Using lower weight decay
10. **Full model - momentum (full-momentum)**: Removing optimizer momentum

### 3. Best Model Experiment

Based on the results of hyperparameter tuning, we select the best performing hyperparameter configuration and train the final model for comprehensive evaluation.

## How to Run Experiments

### 1. Environment Setup

Make sure you have the required Python packages installed:
- numpy
- matplotlib
- seaborn

> **Note**: The original code uses scikit-learn's `confusion_matrix` function, but if third-party libraries are not allowed, you can use the following custom function instead:
> 
> ```python
> def confusion_matrix(y_true, y_pred, num_classes=10):
>     """
>     Calculate confusion matrix
>     
>     Parameters:
>     ------
>     y_true : ndarray
>         True labels
>     y_pred : ndarray
>         Predicted labels
>     num_classes : int
>         Number of classes
>     
>     Returns:
>     ------
>     cm : ndarray
>         Confusion matrix
>     """
>     cm = np.zeros((num_classes, num_classes), dtype=int)
>     for i in range(len(y_true)):
>         cm[y_true[i]][y_pred[i]] += 1
>     return cm
> ```
> 
> Add this function to `utils/visualization.py` and remove the `from sklearn.metrics import confusion_matrix` import statement.

### 2. Running Hyperparameter Tuning Experiments

#### Single Hyperparameter Experiment

To run a specific hyperparameter tuning experiment, use the following commands:

```shell
# Learning rate experiment
python -m experiments.hyperparameter_tuning --experiment learning_rate

# Weight decay experiment
python -m experiments.hyperparameter_tuning --experiment weight_decay

# Neurons experiment
python -m experiments.hyperparameter_tuning --experiment neurons

# Hidden layers experiment
python -m experiments.hyperparameter_tuning --experiment layers

# Batch size experiment
python -m experiments.hyperparameter_tuning --experiment batch_size

# Activation function experiment
python -m experiments.hyperparameter_tuning --experiment activation

# Momentum experiment
python -m experiments.hyperparameter_tuning --experiment momentum

# Dropout rate experiment
python -m experiments.hyperparameter_tuning --experiment dropout_rate
```

You can also use a more concise command line:

```shell
python run_hyperparameter_tuning.py --experiment=learning_rate
```

#### Running All Hyperparameter Experiments

To run all hyperparameter tuning experiments, use the following command:

```shell
python run_hyperparameter_tuning.py --experiment=all
```

Or:

```shell
python -m experiments.hyperparameter_tuning --experiment all
```

You can add additional parameters to adjust experiment settings, for example:

```shell
python run_hyperparameter_tuning.py --experiment=all --epochs=30 --batch_size=32
```

### 3. Running Ablation Study

To run the ablation study, use the following command:

```shell
python -m experiments.ablation_study
```

### 4. Running Best Model Experiment

Based on the hyperparameter tuning results, run the best model:

```shell
python -m experiments.best_model --dropout_rate 0.1 --weight_decay 0.005 --momentum 0.5 --n_layers 1 --batch_size 16 --activation relu --n_neurons 512 --learning_rate 0.1
```

Or simply use:

```shell
python -m experiments.best_model
```

This command will automatically load the saved best hyperparameter configuration.

### 5. Collecting and Analyzing Results

To collect all experiment results and generate summary reports, run:

```shell
python collect_results.py
```

This will generate the following files:
- `all_experimental_results.txt`: Detailed summary of all experiment results
- `all_best_results.txt`: Brief summary of the best configuration for each hyperparameter
- `hyperparameter_summary.txt`: Intuitive summary of hyperparameter tuning results

## Experimental Results

### Hyperparameter Tuning Results Summary

Based on the experimental results, the best hyperparameter configuration is as follows:
- Dropout rate: 0.1 (Accuracy: 0.5381)
- Weight decay: 0.005 (Accuracy: 0.5352)
- Momentum: 0.5 (Accuracy: 0.5344)
- Number of hidden layers: 1 (Accuracy: 0.5336)
- Batch size: 16 (Accuracy: 0.5328)
- Activation function: relu (Accuracy: 0.5328)
- Number of neurons: 512 (Accuracy: 0.5255)
- Learning rate: 0.1 (Accuracy: 0.5139)

### Ablation Study Results Summary

The ablation study shows:
- Removing momentum has the smallest impact on performance, with the full-momentum configuration achieving 53.14% accuracy
- Dropout is a key factor in improving model performance, with removing Dropout causing accuracy to drop to 42.53%
- Batch normalization may be harmful without other regularization techniques, with the base+batch_norm configuration only achieving 19.75% accuracy

## Visualization and Analysis

Each experiment generates various visualization charts, including:
- Training history curves (loss and accuracy)
- Confusion matrix
- Class accuracy distribution
- Hyperparameter vs. performance plots
- Precision, recall, and F1 score for each class

These charts are saved in the corresponding experiment subdirectory under the `results` directory.

## Conclusions and Recommendations

1. Shallower network structures (single hidden layer) perform best on this task
2. Appropriate regularization (Dropout=0.1, weight decay=0.005) significantly improves model performance
3. Smaller batch sizes (16) help the model converge to better solutions
4. ReLU activation function and SGD optimizer with momentum=0.5 are the best choices
5. The model's recognition ability varies across classes, and it's recommended to focus on easily confused class pairs 