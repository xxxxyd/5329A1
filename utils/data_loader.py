import numpy as np

def load_data(data_dir='data'):
    """
    Load the training and test data from numpy files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    train_data : ndarray
        Training data of shape (n_samples, n_features)
    train_labels : ndarray
        Training labels of shape (n_samples, 1)
    test_data : ndarray
        Test data of shape (n_samples, n_features)
    test_labels : ndarray
        Test labels of shape (n_samples, 1)
    """
    train_data = np.load(f'{data_dir}/train_data.npy')
    train_labels = np.load(f'{data_dir}/train_labels.npy')
    test_data = np.load(f'{data_dir}/test_data.npy')
    test_labels = np.load(f'{data_dir}/test_labels.npy')
    
    return train_data, train_labels, test_data, test_labels


def split_train_val(train_data, train_labels, val_ratio=0.2, random_seed=42):
    """
    Split the training data into training and validation sets.
    
    Parameters:
    -----------
    train_data : ndarray
        Training data of shape (n_samples, n_features)
    train_labels : ndarray
        Training labels of shape (n_samples, 1)
    val_ratio : float
        Ratio of validation data (between 0 and 1)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_subset : ndarray
        Training subset data
    train_subset_labels : ndarray
        Training subset labels
    val_subset : ndarray
        Validation subset data
    val_subset_labels : ndarray
        Validation subset labels
    """
    np.random.seed(random_seed)
    num_samples = train_data.shape[0]
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_subset = train_data[train_indices]
    train_subset_labels = train_labels[train_indices]
    val_subset = train_data[val_indices]
    val_subset_labels = train_labels[val_indices]
    
    return train_subset, train_subset_labels, val_subset, val_subset_labels


def create_mini_batches(data, labels, batch_size=64, shuffle=True, seed=None):
    """
    Create mini-batches from the data and labels.
    
    Parameters:
    -----------
    data : ndarray
        Data of shape (n_samples, n_features)
    labels : ndarray
        Labels of shape (n_samples, 1)
    batch_size : int
        Size of each mini-batch
    shuffle : bool
        Whether to shuffle the data before creating mini-batches
    seed : int or None
        Random seed for shuffling
        
    Returns:
    --------
    mini_batches : list
        List of tuples (data_batch, labels_batch)
    """
    mini_batches = []
    n_samples = data.shape[0]
    
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]
    else:
        shuffled_data = data
        shuffled_labels = labels
    
    num_complete_batches = n_samples // batch_size
    
    for i in range(num_complete_batches):
        batch_data = shuffled_data[i * batch_size:(i + 1) * batch_size]
        batch_labels = shuffled_labels[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((batch_data, batch_labels))
    
    # Handle the remaining samples (last mini-batch)
    if n_samples % batch_size != 0:
        batch_data = shuffled_data[num_complete_batches * batch_size:]
        batch_labels = shuffled_labels[num_complete_batches * batch_size:]
        mini_batches.append((batch_data, batch_labels))
    
    return mini_batches


# Define train_val_split as an alias for split_train_val for compatibility
train_val_split = split_train_val
