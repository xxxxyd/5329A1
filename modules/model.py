import numpy as np
import time
from copy import deepcopy

class Model:
    """Neural network model class."""
    
    def __init__(self, layers=None):
        """
        Initialize neural network model.
        
        Parameters:
        -----------
        layers : list
            List of layers
        """
        self.layers = layers if layers is not None else []
        self.loss = None
        self.optimizer = None
        self.metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_params = None
        self.best_val_loss = float('inf')
        self.training = True
        self.patience_counter = 0
    
    def add(self, layer):
        """
        Add a layer to the model.
        
        Parameters:
        -----------
        layer : Layer
            Layer to add
        """
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """
        Compile the model with loss function and optimizer.
        
        Parameters:
        -----------
        loss : Loss
            Loss function
        optimizer : Optimizer
            Optimizer
        """
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        x : ndarray
            Input data
            
        Returns:
        --------
        out : ndarray
            Output of the model
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self):
        """
        Backward pass through the model.
        
        Returns:
        --------
        dout : ndarray
            Gradient with respect to the input
        """
        dout = self.loss.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def set_train_mode(self, mode=True):
        """
        Set training mode for all layers.
        
        Parameters:
        -----------
        mode : bool
            Whether to set training mode or evaluation mode
        """
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'set_train_mode'):
                layer.set_train_mode(mode)
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
        --------
        num_params : int
            Number of trainable parameters
        """
        num_params = 0
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    num_params += param.size
        return num_params
    
    def predict(self, x):
        """
        Make predictions.
        
        Parameters:
        -----------
        x : ndarray
            Input data
            
        Returns:
        --------
        predictions : ndarray
            Model predictions
        """
        # Set to evaluation mode
        self.set_train_mode(False)
        
        # Forward pass
        scores = self.forward(x)
        
        # Get predictions
        predictions = np.argmax(scores, axis=1)
        
        # Set back to training mode
        self.set_train_mode(self.training)
        
        return predictions
    
    def evaluate(self, x, y):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        x : ndarray
            Input data
        y : ndarray
            True labels
            
        Returns:
        --------
        metrics : dict
            Dictionary with evaluation metrics
        """
        # Set to evaluation mode
        self.set_train_mode(False)
        
        # Forward pass
        scores = self.forward(x)
        
        # Compute loss
        loss = self.loss.forward(scores, y)
        
        # Compute accuracy
        predictions = np.argmax(scores, axis=1)
        if len(y.shape) > 1:
            # Convert one-hot encoded labels to class indices
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        accuracy = np.mean(predictions == y_indices)
        
        # Compute precision, recall, and F1 score for each class
        precision = {}
        recall = {}
        f1_score = {}
        
        for class_idx in range(scores.shape[1]):
            # True positives: predicted class_idx and actual class_idx
            tp = np.sum((predictions == class_idx) & (y_indices == class_idx))
            
            # False positives: predicted class_idx but not actual class_idx
            fp = np.sum((predictions == class_idx) & (y_indices != class_idx))
            
            # False negatives: not predicted class_idx but actual class_idx
            fn = np.sum((predictions != class_idx) & (y_indices == class_idx))
            
            # Compute precision: tp / (tp + fp)
            if tp + fp == 0:
                precision[class_idx] = 0
            else:
                precision[class_idx] = tp / (tp + fp)
            
            # Compute recall: tp / (tp + fn)
            if tp + fn == 0:
                recall[class_idx] = 0
            else:
                recall[class_idx] = tp / (tp + fn)
            
            # Compute F1 score: 2 * (precision * recall) / (precision + recall)
            if precision[class_idx] + recall[class_idx] == 0:
                f1_score[class_idx] = 0
            else:
                f1_score[class_idx] = 2 * (precision[class_idx] * recall[class_idx]) / (precision[class_idx] + recall[class_idx])
        
        # Compute average precision, recall, and F1 score
        avg_precision = np.mean(list(precision.values()))
        avg_recall = np.mean(list(recall.values()))
        avg_f1_score = np.mean(list(f1_score.values()))
        
        # Compute confusion matrix
        num_classes = scores.shape[1]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(predictions)):
            confusion_matrix[y_indices[i], predictions[i]] += 1
        
        # Set back to training mode
        self.set_train_mode(self.training)
        
        # Return metrics
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1_score,
            'confusion_matrix': confusion_matrix
        }
        
        return metrics
    
    def train_on_batch(self, x, y):
        """
        Train the model on a batch of data.
        
        Parameters:
        -----------
        x : ndarray
            Input data
        y : ndarray
            True labels
            
        Returns:
        --------
        metrics : dict
            Dictionary with training metrics
        """
        # Set to training mode
        self.set_train_mode(True)
        
        # Forward pass
        scores = self.forward(x)
        
        # Compute loss
        loss = self.loss.forward(scores, y)
        
        # Backward pass
        self.backward()
        
        # Update parameters
        self.optimizer.update(self)
        
        # Compute accuracy
        predictions = np.argmax(scores, axis=1)
        if len(y.shape) > 1:
            # Convert one-hot encoded labels to class indices
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        accuracy = np.mean(predictions == y_indices)
        
        # Return metrics
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=100, batch_size=32,
              shuffle=True, verbose=1, use_early_stopping=False, patience=10,
              use_learning_rate_scheduler=False, learning_rate_decay=0.5, learning_rate_schedule=None,
              use_gradient_clipping=False, clip_value=5.0, save_best=True):
        """
        Train the model.
        
        Parameters:
        -----------
        x_train : ndarray
            Training data
        y_train : ndarray
            Training labels
        x_val : ndarray or None
            Validation data
        y_val : ndarray or None
            Validation labels
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle the data before each epoch
        verbose : int
            Verbosity level
        use_early_stopping : bool
            Whether to use early stopping
        patience : int
            Number of epochs with no improvement after which training will be stopped
        use_learning_rate_scheduler : bool
            Whether to use learning rate scheduler
        learning_rate_decay : float
            Learning rate decay factor
        learning_rate_schedule : list or None
            List of epochs after which to decay the learning rate
        use_gradient_clipping : bool
            Whether to use gradient clipping
        clip_value : float
            Gradient clipping value
        save_best : bool
            Whether to save the best model
            
        Returns:
        --------
        history : dict
            Dictionary with training history
        """
        # Reset metrics
        self.metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Get number of samples
        n_samples = x_train.shape[0]
        
        # Default learning rate scheduler
        if learning_rate_schedule is None and use_learning_rate_scheduler:
            learning_rate_schedule = [epoch for epoch in range(50, epochs, 50)]
        
        # Initialize best parameters
        if save_best:
            self.best_params = self._get_parameters()
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Learning rate scheduler
            if use_learning_rate_scheduler and epoch in learning_rate_schedule:
                self.optimizer.learning_rate *= learning_rate_decay
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: learning rate = {self.optimizer.learning_rate:.6f}")
            
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                x_train_shuffled = x_train
                y_train_shuffled = y_train
            
            # Mini-batch training
            train_loss = 0
            train_acc = 0
            
            for i in range(0, n_samples, batch_size):
                # Get mini-batch
                x_batch = x_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Train on mini-batch
                batch_metrics = self.train_on_batch(x_batch, y_batch)
                
                # Gradient clipping
                if use_gradient_clipping:
                    self._clip_gradients(clip_value)
                
                # Accumulate metrics
                train_loss += batch_metrics['loss'] * len(x_batch)
                train_acc += batch_metrics['accuracy'] * len(x_batch)
            
            # Compute average metrics
            train_loss /= n_samples
            train_acc /= n_samples
            
            # Evaluate on validation set
            if x_val is not None and y_val is not None:
                val_metrics = self.evaluate(x_val, y_val)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_params = self._get_parameters()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if use_early_stopping and self.patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss = None
                val_acc = None
            
            # Store metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            if val_loss is not None:
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.2f}s] - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.2f}s] - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}")
        
        # Restore best parameters if early stopping or save_best
        if save_best and self.best_params is not None:
            self._set_parameters(self.best_params)
        
        # Print final results
        if verbose:
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f}s")
            print(f"Number of parameters: {self.count_parameters()}")
        
        return self.metrics
    
    def _get_parameters(self):
        """
        Get all parameters of the model.
        
        Returns:
        --------
        params : list
            List of parameters
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                params.append(deepcopy(layer.params))
            else:
                params.append(None)
        return params
    
    def _set_parameters(self, params):
        """
        Set all parameters of the model.
        
        Parameters:
        -----------
        params : list
            List of parameters
        """
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and params[i] is not None:
                layer.params = deepcopy(params[i])
    
    def _clip_gradients(self, clip_value):
        """
        Clip gradients to prevent exploding gradients.
        
        Parameters:
        -----------
        clip_value : float
            Maximum gradient value
        """
        for layer in self.layers:
            if hasattr(layer, 'grads'):
                for param_name in layer.grads:
                    layer.grads[param_name] = np.clip(layer.grads[param_name], -clip_value, clip_value)
    
    def save_weights(self, filepath):
        """
        Save model weights to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the weights
        """
        weights = self._get_parameters()
        np.save(filepath, weights, allow_pickle=True)
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the weights from
        """
        weights = np.load(filepath, allow_pickle=True)
        self._set_parameters(weights)


def create_mlp(input_dim, hidden_dims, output_dim, activation_name='relu', use_batch_norm=False,
               use_dropout=False, dropout_rate=0.5, weight_scale=0.01):
    """
    Create a multi-layer perceptron model.
    
    Parameters:
    -----------
    input_dim : int
        Input dimension
    hidden_dims : list
        List of hidden layer dimensions
    output_dim : int
        Output dimension
    activation_name : str
        Name of the activation function
    use_batch_norm : bool
        Whether to use batch normalization
    use_dropout : bool
        Whether to use dropout
    dropout_rate : float
        Dropout rate
    weight_scale : float
        Scale for weight initialization
        
    Returns:
    --------
    model : Model
        MLP model
    """
    from modules.activations import get_activation
    from modules.layers import FullyConnected, BatchNorm, Dropout
    
    model = Model()
    
    # Add input layer
    model.add(FullyConnected(input_dim, hidden_dims[0], weight_scale=weight_scale))
    
    # Add batch normalization if requested
    if use_batch_norm:
        model.add(BatchNorm(hidden_dims[0]))
    
    # Add activation function
    model.add(get_activation(activation_name))
    
    # Add dropout if requested
    if use_dropout:
        model.add(Dropout(p=1-dropout_rate))
    
    # Add hidden layers
    for i in range(len(hidden_dims) - 1):
        model.add(FullyConnected(hidden_dims[i], hidden_dims[i+1], weight_scale=weight_scale))
        
        # Add batch normalization if requested
        if use_batch_norm:
            model.add(BatchNorm(hidden_dims[i+1]))
        
        # Add activation function
        model.add(get_activation(activation_name))
        
        # Add dropout if requested
        if use_dropout:
            model.add(Dropout(p=1-dropout_rate))
    
    # Add output layer
    model.add(FullyConnected(hidden_dims[-1], output_dim, weight_scale=weight_scale))
    
    return model


# Define MultiLayerNetwork as an alias for Model for compatibility
MultiLayerNetwork = Model
