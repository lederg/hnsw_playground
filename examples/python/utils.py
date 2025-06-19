import numpy as np
import numpy.typing as npt
import functools





def unique_in_order(k, a: npt.NDArray,) -> npt.NDArray:
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)][:k]


def unique_along_axis(arr: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """
    Returns the number of unique elements along a specified axis of a 2D numpy array.
    
    Parameters:
    - arr: Input 2D numpy array.
    - axis: Axis along which to find unique elements (0 for rows, 1 for columns).
    
    Returns:
    - A 1D numpy array of the counts of unique elements along the specified axis.
    """
    if axis == 0:
        return np.array([len(np.unique(row)) for row in arr])
    elif axis == 1:
        return np.array([len(np.unique(arr[:, col])) for col in range(arr.shape[1])])
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")


def argpartition_top_k(similarity_matrix: npt.NDArray, initial_k: int) -> npt.NDArray:
    """
    Efficiently finds the indices of the top-k elements in each row of a similarity matrix.
    
    Parameters:
    - similarity_matrix: 2D numpy array representing the similarity scores.
    - initial_k: The number of top elements to find.
    
    Returns:
    - A 2D numpy array of indices of the top-k elements in each row.
    """
    if initial_k <= 0 or initial_k > similarity_matrix.shape[1]:
        raise ValueError("k must be a positive integer and less than or equal to the number of columns in the matrix.")
    # Use argpartition to get the indices of the top-k elements efficiently
    if initial_k == 0:
        return np.argsort(similarity_matrix, axis=1)[:, ::-1]
    if initial_k > similarity_matrix.shape[1]:
        raise ValueError("k is too large for the number of elements in the similarity matrix.")
    # Get the indices of the top-k elements using argpartition
    idx_part = np.argpartition(similarity_matrix, -initial_k, axis=1)[:, -initial_k:]
    row_indices = np.arange(similarity_matrix.shape[0])[:, None]
    top_similarities = similarity_matrix[row_indices, idx_part]
    sorted_idx = np.argsort(top_similarities, axis=1)[:, ::-1]
    indices_by_similarity = idx_part[row_indices, sorted_idx]

    return indices_by_similarity


def compute_similarity_and_l2(query, db, k=0):
    similarity_matrix = np.dot(query, db.T)
    if k > 0:
        # Use argpartition for efficiency when k is small
        idx_part = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
        # Gather the top-k similarities and their indices, then sort them
        row_indices = np.arange(similarity_matrix.shape[0])[:, None]
        top_similarities = similarity_matrix[row_indices, idx_part]
        sorted_idx = np.argsort(top_similarities, axis=1)[:, ::-1]
        indices_by_similarity = idx_part[row_indices, sorted_idx]
        vectors_by_similarity = db[indices_by_similarity]
        l2_distances = np.linalg.norm(query[:, np.newaxis] - vectors_by_similarity, axis=2)
        return indices_by_similarity, l2_distances
    else:
        indices_by_similarity = np.argsort(similarity_matrix, axis=1)[:, ::-1]
        vectors_by_similarity = db[indices_by_similarity]
        l2_distances = np.linalg.norm(query[:, np.newaxis] - vectors_by_similarity, axis=2)
        return indices_by_similarity, l2_distances

def compute_similarity_and_l2_with_unique(query, db, docids, k=0):
    similarity_matrix = np.dot(query, db.T)
    horizon = 2 * k  # to ensure we have enough candidates
    candidates_matrix = None

    def is_done(candidates_matrix) -> bool:
        return np.all(np.array([len(np.unique(row)) for row in candidates_matrix]) >= k)
    if k == 0:
        candidates_matrix = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    else:
        while candidates_matrix is None or not is_done(candidates_matrix):
            candidates_matrix = argpartition_top_k(similarity_matrix, horizon)        # Filter unique docids
            horizon *= 2
    # Now we have enough candidates, filter them to get the top k unique docids. Make sure to choose the nearest neighber per unique docid.
    indices_by_similarity = np.apply_along_axis(functools.partial(unique_in_order, k), axis=1, arr=candidates_matrix)
    vectors_by_similarity = db[indices_by_similarity]
    l2_distances = np.linalg.norm(query[:, np.newaxis] - vectors_by_similarity, axis=2)    

    return indices_by_similarity, l2_distances
