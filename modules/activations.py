import numpy as np

class Activation:
    """Base class for activation functions."""
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, dout):
        """Backward pass."""
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
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
            Output after applying ReLU
        """
        self.x = x
        return np.maximum(0, x)
    
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
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
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
            Output after applying sigmoid
        """
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    
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
        return dout * self.out * (1 - self.out)


class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    
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
            Output after applying tanh
        """
        self.out = np.tanh(x)
        return self.out
    
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
        return dout * (1 - self.out**2)


class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha=0.01):
        """
        Initialize Leaky ReLU.
        
        Parameters:
        -----------
        alpha : float
            Slope for negative inputs
        """
        self.alpha = alpha
    
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
            Output after applying Leaky ReLU
        """
        self.x = x
        return np.maximum(self.alpha * x, x)
    
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
        dx = dout.copy()
        dx[self.x < 0] = self.alpha * dx[self.x < 0]
        return dx


class ELU(Activation):
    """Exponential Linear Unit activation function."""
    
    def __init__(self, alpha=1.0):
        """
        Initialize ELU.
        
        Parameters:
        -----------
        alpha : float
            Coefficient for the negative factor
        """
        self.alpha = alpha
    
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
            Output after applying ELU
        """
        self.x = x
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
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
        dx = dout.copy()
        dx = np.where(self.x > 0, dx, dx * self.alpha * np.exp(self.x))
        return dx


class GELU(Activation):
    """Gaussian Error Linear Unit activation function."""
    
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
            Output after applying GELU
        """
        # Approximation of GELU
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
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
        # Approximate derivative of GELU
        x = self.x
        tanh_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
        
        # Derivative of tanh is (1 - tanh^2)
        sech_squared = 1 - tanh_part**2
        
        # Chain rule
        inside_derivative = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        
        return dout * (0.5 * (1 + tanh_part) + 0.5 * x * sech_squared * inside_derivative)


class Softmax:
    """Softmax function for output layer."""
    
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
            Output after applying softmax
        """
        # Shift values for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Note: This is typically combined with the loss function in practice
        for softmax + cross-entropy loss, so this method is not used directly.
        
        Parameters:
        -----------
        dout : ndarray
            Gradient from the next layer
            
        Returns:
        --------
        dx : ndarray
            Gradient with respect to input
        """
        # This is not typically used separately as it's combined with the cross-entropy loss
        # But we provide it for completeness
        n_samples = self.out.shape[0]
        dx = np.zeros_like(dout)
        
        for i in range(n_samples):
            s = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            dx[i] = np.dot(jacobian, dout[i])
            
        return dx


def get_activation(activation_name):
    """
    Get activation function by name.
    
    Parameters:
    -----------
    activation_name : str
        Name of the activation function
        
    Returns:
    --------
    activation : Activation
        Activation function
    """
    activation_functions = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'leaky_relu': LeakyReLU,
        'gelu': GELU,
        'elu': ELU,
        'softmax': Softmax
    }
    
    if activation_name.lower() not in activation_functions:
        raise ValueError(f"Activation function '{activation_name}' not found.")
    
    return activation_functions[activation_name.lower()]()
