import numpy as np

class Loss:
    """Base class for loss functions."""
    
    def forward(self, y_pred, y_true):
        """
        Compute the loss.
        
        Parameters:
        -----------
        y_pred : ndarray
            Predicted values
        y_true : ndarray
            True values
            
        Returns:
        --------
        loss : float
            Loss value
        """
        raise NotImplementedError
    
    def backward(self):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Returns:
        --------
        dout : ndarray
            Gradient of the loss with respect to the predictions
        """
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Cross-entropy loss with softmax activation.
    
    This implements the combined softmax activation and cross-entropy loss
    for numerical stability.
    """
    
    def forward(self, scores, y_true):
        """
        Compute the cross-entropy loss with softmax activation.
        
        Parameters:
        -----------
        scores : ndarray
            Raw scores (logits) of shape (batch_size, num_classes)
        y_true : ndarray
            True labels, either as one-hot encoded vectors of shape (batch_size, num_classes)
            or as class indices of shape (batch_size,)
            
        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Convert class indices to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true_one_hot = np.zeros((y_true.shape[0], scores.shape[1]))
            y_true_one_hot[np.arange(y_true.shape[0]), y_true.flatten()] = 1
            y_true = y_true_one_hot
        
        self.y_true = y_true
        self.batch_size = scores.shape[0]
        
        # Compute softmax
        # Shift scores for numerical stability
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute cross-entropy loss
        correct_log_probs = -np.log(self.probs[np.arange(self.batch_size), np.argmax(y_true, axis=1)])
        loss = np.sum(correct_log_probs) / self.batch_size
        
        return loss
    
    def backward(self):
        """
        Compute the gradient of the loss with respect to the scores.
        
        Returns:
        --------
        dscores : ndarray
            Gradient of the loss with respect to the scores
        """
        # Gradient of cross-entropy loss with respect to softmax probabilities
        dprobs = self.probs.copy()
        dprobs[np.arange(self.batch_size), np.argmax(self.y_true, axis=1)] -= 1
        dprobs /= self.batch_size
        
        return dprobs


class MSELoss(Loss):
    """Mean squared error loss."""
    
    def forward(self, y_pred, y_true):
        """
        Compute the mean squared error loss.
        
        Parameters:
        -----------
        y_pred : ndarray
            Predicted values of shape (batch_size, output_dim)
        y_true : ndarray
            True values of shape (batch_size, output_dim)
            
        Returns:
        --------
        loss : float
            Mean squared error loss
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]
        
        # Compute MSE loss
        self.diff = y_pred - y_true
        loss = np.sum(self.diff**2) / (2 * self.batch_size)
        
        return loss
    
    def backward(self):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Returns:
        --------
        dout : ndarray
            Gradient of the loss with respect to the predictions
        """
        # Gradient of MSE loss with respect to predictions
        dout = self.diff / self.batch_size
        
        return dout


class L1Loss(Loss):
    """Mean absolute error loss."""
    
    def forward(self, y_pred, y_true):
        """
        Compute the mean absolute error loss.
        
        Parameters:
        -----------
        y_pred : ndarray
            Predicted values of shape (batch_size, output_dim)
        y_true : ndarray
            True values of shape (batch_size, output_dim)
            
        Returns:
        --------
        loss : float
            Mean absolute error loss
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]
        
        # Compute L1 loss
        self.diff = y_pred - y_true
        loss = np.sum(np.abs(self.diff)) / self.batch_size
        
        return loss
    
    def backward(self):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Returns:
        --------
        dout : ndarray
            Gradient of the loss with respect to the predictions
        """
        # Gradient of L1 loss with respect to predictions
        dout = np.sign(self.diff) / self.batch_size
        
        return dout


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax cross-entropy loss for multi-class classification.
    
    This is a more numerically stable implementation.
    """
    
    def forward(self, scores, y_true):
        """
        Compute the softmax cross-entropy loss.
        
        Parameters:
        -----------
        scores : ndarray
            Raw scores (logits) of shape (batch_size, num_classes)
        y_true : ndarray
            True labels, either as one-hot encoded vectors of shape (batch_size, num_classes)
            or as class indices of shape (batch_size,)
            
        Returns:
        --------
        loss : float
            Softmax cross-entropy loss
        """
        self.scores = scores
        self.batch_size = scores.shape[0]
        
        # Convert class indices to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            self.y_class = y_true.flatten()
        else:
            self.y_class = np.argmax(y_true, axis=1)
        
        # Compute shifted logits for numerical stability
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        logits_for_answers = shifted_logits[np.arange(self.batch_size), self.y_class]
        
        # Compute cross-entropy loss
        loss = np.mean(-logits_for_answers + log_sum_exp.flatten())
        
        # Compute softmax probabilities for backward pass
        self.probs = np.exp(shifted_logits - log_sum_exp)
        
        return loss
    
    def backward(self):
        """
        Compute the gradient of the loss with respect to the scores.
        
        Returns:
        --------
        dscores : ndarray
            Gradient of the loss with respect to the scores
        """
        # Make a copy of the probabilities
        dscores = self.probs.copy()
        
        # For the correct class, subtract 1 from the probability
        dscores[np.arange(self.batch_size), self.y_class] -= 1
        
        # Normalize by batch size
        dscores /= self.batch_size
        
        return dscores


def get_loss(loss_name):
    """
    Get loss function by name.
    
    Parameters:
    -----------
    loss_name : str
        Name of the loss function
        
    Returns:
    --------
    loss : Loss
        Loss function
    """
    loss_functions = {
        'cross_entropy': CrossEntropyLoss,
        'mse': MSELoss,
        'l1': L1Loss,
        'softmax_cross_entropy': SoftmaxCrossEntropyLoss
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Loss function '{loss_name}' not found.")
    
    return loss_functions[loss_name.lower()]()
