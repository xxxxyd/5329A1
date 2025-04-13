import numpy as np

class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        """
        self.learning_rate = learning_rate
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        weight_decay : float
            Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # Add L2 regularization gradient
                    if 'W' in param_name and self.weight_decay > 0:
                        layer.grads[param_name] += self.weight_decay * layer.params[param_name]
                    
                    # Update parameters
                    layer.params[param_name] -= self.learning_rate * layer.grads[param_name]


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent with Momentum optimizer.
    
    Momentum helps accelerate gradients in the right direction,
    thus leading to faster convergence.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        """
        Initialize SGD with Momentum optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        momentum : float
            Momentum coefficient
        weight_decay : float
            Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        # Initialize velocity if it doesn't exist
        if not self.velocity:
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    self.velocity[i] = {}
                    for param_name in layer.params:
                        self.velocity[i][param_name] = np.zeros_like(layer.params[param_name])
        
        # Update parameters
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # Add L2 regularization gradient
                    if 'W' in param_name and self.weight_decay > 0:
                        layer.grads[param_name] += self.weight_decay * layer.params[param_name]
                    
                    # Update velocity
                    self.velocity[i][param_name] = self.momentum * self.velocity[i][param_name] - self.learning_rate * layer.grads[param_name]
                    
                    # Update parameters
                    layer.params[param_name] += self.velocity[i][param_name]


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Adam (Adaptive Moment Estimation) combines ideas from RMSprop and Momentum.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize Adam optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        beta1 : float
            Exponential decay rate for the first moment estimates
        beta2 : float
            Exponential decay rate for the second moment estimates
        epsilon : float
            Small constant for numerical stability
        weight_decay : float
            Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        self.t = 0   # Timestep
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        self.t += 1
        
        # Initialize moments if they don't exist
        if not self.m:
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    self.m[i] = {}
                    self.v[i] = {}
                    for param_name in layer.params:
                        self.m[i][param_name] = np.zeros_like(layer.params[param_name])
                        self.v[i][param_name] = np.zeros_like(layer.params[param_name])
        
        # Update parameters
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # Add L2 regularization gradient
                    if 'W' in param_name and self.weight_decay > 0:
                        layer.grads[param_name] += self.weight_decay * layer.params[param_name]
                    
                    # Update biased first moment estimate
                    self.m[i][param_name] = self.beta1 * self.m[i][param_name] + (1 - self.beta1) * layer.grads[param_name]
                    
                    # Update biased second raw moment estimate
                    self.v[i][param_name] = self.beta2 * self.v[i][param_name] + (1 - self.beta2) * (layer.grads[param_name]**2)
                    
                    # Compute bias-corrected first moment estimate
                    m_corrected = self.m[i][param_name] / (1 - self.beta1**self.t)
                    
                    # Compute bias-corrected second raw moment estimate
                    v_corrected = self.v[i][param_name] / (1 - self.beta2**self.t)
                    
                    # Update parameters
                    layer.params[param_name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    RMSprop maintains a moving average of the squared gradient for each parameter.
    """
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize RMSprop optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        decay_rate : float
            Decay rate for moving average
        epsilon : float
            Small constant for numerical stability
        weight_decay : float
            Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        # Initialize cache if it doesn't exist
        if not self.cache:
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    self.cache[i] = {}
                    for param_name in layer.params:
                        self.cache[i][param_name] = np.zeros_like(layer.params[param_name])
        
        # Update parameters
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # Add L2 regularization gradient
                    if 'W' in param_name and self.weight_decay > 0:
                        layer.grads[param_name] += self.weight_decay * layer.params[param_name]
                    
                    # Update cache
                    self.cache[i][param_name] = self.decay_rate * self.cache[i][param_name] + (1 - self.decay_rate) * (layer.grads[param_name]**2)
                    
                    # Update parameters
                    layer.params[param_name] -= self.learning_rate * layer.grads[param_name] / (np.sqrt(self.cache[i][param_name]) + self.epsilon)


class Adagrad(Optimizer):
    """
    Adagrad optimizer.
    
    Adagrad adapts the learning rate to the parameters, performing smaller updates
    for parameters associated with frequently occurring features, and larger updates
    for parameters associated with infrequent features.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize Adagrad optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        epsilon : float
            Small constant for numerical stability
        weight_decay : float
            Weight decay (L2 regularization)
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}
    
    def update(self, model):
        """
        Update model parameters.
        
        Parameters:
        -----------
        model : Model
            Model to update
        """
        # Initialize cache if it doesn't exist
        if not self.cache:
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    self.cache[i] = {}
                    for param_name in layer.params:
                        self.cache[i][param_name] = np.zeros_like(layer.params[param_name])
        
        # Update parameters
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # Add L2 regularization gradient
                    if 'W' in param_name and self.weight_decay > 0:
                        layer.grads[param_name] += self.weight_decay * layer.params[param_name]
                    
                    # Update cache
                    self.cache[i][param_name] += layer.grads[param_name]**2
                    
                    # Update parameters
                    layer.params[param_name] -= self.learning_rate * layer.grads[param_name] / (np.sqrt(self.cache[i][param_name]) + self.epsilon)


def get_optimizer(optimizer_name, **kwargs):
    """
    Get optimizer by name.
    
    Parameters:
    -----------
    optimizer_name : str
        Name of the optimizer
    **kwargs : dict
        Additional arguments for the optimizer
        
    Returns:
    --------
    optimizer : Optimizer
        Optimizer
    """
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': Adagrad
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")
    
    return optimizers[optimizer_name.lower()](**kwargs)
