import numpy as np

class Layer:
    """Base class for neural network layers."""
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, dout):
        """Backward pass."""
        raise NotImplementedError


class FullyConnected(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_dim, output_dim, weight_scale=0.01):
        """
        Initialize fully connected layer.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output features
        weight_scale : float
            Standard deviation for weight initialization
        """
        self.params = {}
        self.params['W'] = weight_scale * np.random.randn(input_dim, output_dim)
        self.params['b'] = np.zeros(output_dim)
        self.grads = {}
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        self.x = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : ndarray
            Input of shape (batch_size, input_dim)
            
        Returns:
        --------
        out : ndarray
            Output of shape (batch_size, output_dim)
        """
        self.x = x
        out = np.dot(x, self.params['W']) + self.params['b']
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from the next layer, of shape (batch_size, output_dim)
            
        Returns:
        --------
        dx : ndarray
            Gradient with respect to input, of shape (batch_size, input_dim)
        """
        # Gradients with respect to parameters
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        
        # Gradient with respect to input
        dx = np.dot(dout, self.params['W'].T)
        return dx


# Define Dense as an alias for FullyConnected for compatibility
Dense = FullyConnected


class Dropout(Layer):
    """
    Dropout layer for regularization.
    
    Randomly sets a fraction of the inputs to zero during training.
    """
    
    def __init__(self, p=0.5):
        """
        Initialize dropout layer.
        
        Parameters:
        -----------
        p : float
            Probability of keeping a neuron active
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : ndarray
            Input
            
        Returns:
        --------
        out : ndarray
            Output with dropout applied
        """
        if not self.training:
            return x
        
        self.mask = (np.random.rand(*x.shape) < self.p) / self.p
        out = x * self.mask
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from the next layer
            
        Returns:
        --------
        dx : ndarray
            Gradient with respect to input
        """
        if not self.training:
            return dout
        
        dx = dout * self.mask
        return dx
    
    def set_train_mode(self, mode=True):
        """
        Set training mode.
        
        Parameters:
        -----------
        mode : bool
            Whether to set training mode or evaluation mode
        """
        self.training = mode


class BatchNorm(Layer):
    """
    Batch Normalization layer.
    
    Normalizes the activations of the previous layer for each mini-batch.
    """
    
    def __init__(self, input_dim, momentum=0.9, eps=1e-5):
        """
        Initialize batch normalization layer.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        momentum : float
            Momentum for running mean and variance
        eps : float
            Small constant for numerical stability
        """
        self.params = {}
        self.params['gamma'] = np.ones(input_dim)
        self.params['beta'] = np.zeros(input_dim)
        self.grads = {}
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
        self.momentum = momentum
        self.eps = eps
        self.training = True
        
        self.x = None
        self.x_norm = None
        self.std = None
        self.batch_mean = None
        self.batch_var = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : ndarray
            Input of shape (batch_size, input_dim)
            
        Returns:
        --------
        out : ndarray
            Normalized output
        """
        if self.training:
            self.x = x
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)
            self.std = np.sqrt(self.batch_var + self.eps)
            self.x_norm = (x - self.batch_mean) / self.std
            out = self.params['gamma'] * self.x_norm + self.params['beta']
            
            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running mean and variance during inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.params['gamma'] * x_norm + self.params['beta']
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from the next layer
            
        Returns:
        --------
        dx : ndarray
            Gradient with respect to input
        """
        if not self.training:
            # Nothing to do for inference mode
            return dout
        
        # Get parameters and dimensions
        N = self.x.shape[0]
        
        # Gradients with respect to gamma and beta
        self.grads['gamma'] = np.sum(dout * self.x_norm, axis=0)
        self.grads['beta'] = np.sum(dout, axis=0)
        
        # Gradient with respect to normalized inputs
        dx_norm = dout * self.params['gamma']
        
        # Gradient with respect to batch variance
        dvar = np.sum(dx_norm * (self.x - self.batch_mean) * -0.5 * self.std**(-3), axis=0)
        
        # Gradient with respect to batch mean
        dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * (self.x - self.batch_mean), axis=0)
        
        # Gradient with respect to input
        dx = dx_norm / self.std + dvar * 2 * (self.x - self.batch_mean) / N + dmean / N
        
        return dx
    
    def set_train_mode(self, mode=True):
        """
        Set training mode.
        
        Parameters:
        -----------
        mode : bool
            Whether to set training mode or evaluation mode
        """
        self.training = mode


class ResidualBlock(Layer):
    """
    Residual block with skip connections.
    
    Allows gradients to flow more easily through deep networks.
    """
    
    def __init__(self, input_dim, hidden_dim, activation, use_batch_norm=True):
        """
        Initialize residual block.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of hidden features
        activation : Activation
            Activation function
        use_batch_norm : bool
            Whether to use batch normalization
        """
        self.use_batch_norm = use_batch_norm
        
        self.fc1 = FullyConnected(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim) if use_batch_norm else None
        self.activation1 = activation
        
        self.fc2 = FullyConnected(hidden_dim, input_dim)  # Output dim must match input dim for residual connection
        self.bn2 = BatchNorm(input_dim) if use_batch_norm else None
        self.activation2 = activation
        
        # Projection in case dimensions differ
        self.projection = None
        if input_dim != hidden_dim:
            self.projection = FullyConnected(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : ndarray
            Input
            
        Returns:
        --------
        out : ndarray
            Output
        """
        identity = x
        
        # First transformation
        out = self.fc1.forward(x)
        if self.use_batch_norm:
            out = self.bn1.forward(out)
        out = self.activation1.forward(out)
        
        # Second transformation
        out = self.fc2.forward(out)
        if self.use_batch_norm:
            out = self.bn2.forward(out)
            
        # Apply projection if needed
        if self.projection is not None:
            identity = self.projection.forward(identity)
        
        # Add residual connection
        out = out + identity
        
        # Final activation
        out = self.activation2.forward(out)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from the next layer
            
        Returns:
        --------
        dx : ndarray
            Gradient with respect to input
        """
        # Backprop through final activation
        dx = self.activation2.backward(dout)
        
        # Save residual gradient
        dresidual = dx.copy()
        
        # Backprop through second transformation
        if self.use_batch_norm:
            dx = self.bn2.backward(dx)
        dx = self.fc2.backward(dx)
        
        # Backprop through first activation
        dx = self.activation1.backward(dx)
        
        # Backprop through first transformation
        if self.use_batch_norm:
            dx = self.bn1.backward(dx)
        dx = self.fc1.backward(dx)
        
        # Add residual gradient
        if self.projection is not None:
            dresidual = self.projection.backward(dresidual)
        
        dx = dx + dresidual
        
        return dx
    
    def set_train_mode(self, mode=True):
        """
        Set training mode for all layers.
        
        Parameters:
        -----------
        mode : bool
            Whether to set training mode or evaluation mode
        """
        if self.use_batch_norm:
            self.bn1.set_train_mode(mode)
            self.bn2.set_train_mode(mode)
