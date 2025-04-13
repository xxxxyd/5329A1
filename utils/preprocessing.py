import numpy as np

def standardize_features(train_data, test_data=None):
    """
    Standardize features to have zero mean and unit variance.
    
    Parameters:
    -----------
    train_data : ndarray
        Training data of shape (n_samples, n_features)
    test_data : ndarray or None
        Test data of shape (n_samples, n_features)
        
    Returns:
    --------
    train_data_std : ndarray
        Standardized training data
    test_data_std : ndarray or None
        Standardized test data (if provided)
    mean : ndarray
        Mean values used for standardization
    std : ndarray
        Standard deviation values used for standardization
    """
    # Calculate mean and std on training data
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    # Handle zero std values
    std[std == 0] = 1.0
    
    # Standardize training data
    train_data_std = (train_data - mean) / std
    
    # Standardize test data if provided
    if test_data is not None:
        test_data_std = (test_data - mean) / std
        return train_data_std, test_data_std, mean, std
    
    return train_data_std, mean, std


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Parameters:
    -----------
    labels : ndarray
        Integer labels of shape (n_samples, 1) or (n_samples,)
    num_classes : int
        Number of classes
        
    Returns:
    --------
    one_hot_labels : ndarray
        One-hot encoded labels of shape (n_samples, num_classes)
    """
    # Ensure labels are 1D
    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    
    n_samples = labels.shape[0]
    one_hot_labels = np.zeros((n_samples, num_classes))
    
    # Set the appropriate indices to 1
    one_hot_labels[np.arange(n_samples), labels] = 1
    
    return one_hot_labels


def normalize_features(train_data, test_data=None, feature_range=(0, 1)):
    """
    Normalize features to a specific range.
    
    Parameters:
    -----------
    train_data : ndarray
        Training data of shape (n_samples, n_features)
    test_data : ndarray or None
        Test data of shape (n_samples, n_features)
    feature_range : tuple
        Range to normalize to (min, max)
        
    Returns:
    --------
    train_data_norm : ndarray
        Normalized training data
    test_data_norm : ndarray or None
        Normalized test data (if provided)
    data_min : ndarray
        Minimum values used for normalization
    data_max : ndarray
        Maximum values used for normalization
    """
    min_val, max_val = feature_range
    
    # Calculate min and max on training data
    data_min = np.min(train_data, axis=0)
    data_max = np.max(train_data, axis=0)
    
    # Handle case where min and max are the same
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    # Scale to [0, 1]
    train_data_scaled = (train_data - data_min) / data_range
    
    # Scale to [min_val, max_val]
    train_data_norm = train_data_scaled * (max_val - min_val) + min_val
    
    # Normalize test data if provided
    if test_data is not None:
        test_data_scaled = (test_data - data_min) / data_range
        test_data_norm = test_data_scaled * (max_val - min_val) + min_val
        return train_data_norm, test_data_norm, data_min, data_max
    
    return train_data_norm, data_min, data_max


# Define normalize_data as an alias for normalize_features for compatibility
normalize_data = normalize_features
