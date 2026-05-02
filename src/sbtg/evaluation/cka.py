import numpy as np

def center_columns(X):
    """
    Centers the columns of a matrix X by subtracting the mean of each column.
    """
    return X - X.mean(axis=0, keepdims=True)

def linear_cka(X, Y):
    """
    Compute Linear Centered Kernel Alignment (CKA) between two matrices X and Y.
    
    Both matrices must have shape (N, D1) and (N, D2), where N is the number of
    samples. This function uses the feature-space formulation, which is much 
    more memory efficient for large N and small D: O(N D^2) instead of O(N^2)
    for constructing the full Gram matrix.
    
    Args:
        X: np.ndarray of shape (N, D1)
        Y: np.ndarray of shape (N, D2)
        
    Returns:
        float: Linear CKA similarity score between X and Y, bounded in [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples (rows).")
    
    Xc = center_columns(X)
    Yc = center_columns(Y)
    
    # Feature-space formulation avoids full NxN gram matrices
    # ||Xc.T Yc||_F^2
    dot_prod = Xc.T @ Yc
    numerator = np.linalg.norm(dot_prod, ord='fro') ** 2
    
    norm_X = np.linalg.norm(Xc.T @ Xc, ord='fro') 
    norm_Y = np.linalg.norm(Yc.T @ Yc, ord='fro')
    
    denom = norm_X * norm_Y
    if denom == 0:
        return 0.0
    return numerator / denom
