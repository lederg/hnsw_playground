import numpy as np

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
