import hnswlib
import numpy as np
import pickle
import time

def test_hnsw_search(
    data,
    ids,
    docids=None,
    space='multivector',
    ef_construction=200,
    M=16,
    ef=50,
    k=3,
):
    """
    Systematic HNSW search testing function.
    """
    use_docids = docids is not None
    dim = data.shape[1]
    num_elements = data.shape[0]
    p = hnswlib.Index(space=space, dim=dim)
    p.set_num_threads(1)
    p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    p.add_items(data, ids, docids_=docids)
    p.set_ef(ef)
    start = time.time()
    labels, distances = p.knn_query(data, k=k, use_docids=use_docids)
    elapsed = time.time() - start
    print(f"knn_query took {elapsed:.4f} seconds")
    if use_docids:
        total_uniqs = np.unique(np.apply_along_axis(lambda row: np.unique(row).size, axis=1, arr=docids[labels]))
        is_uniq_docs = len(total_uniqs) == 1
    p_copy = pickle.loads(pickle.dumps(p))
    print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
    print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
    print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")
    if use_docids:
        print(f"Unique docs: {is_uniq_docs}")
    return labels, distances

if __name__ == "__main__":
    dim = 128
    num_elements = 10000
    data = np.float32(np.random.random((num_elements, dim)))
    ids = np.arange(num_elements)
    docids = np.random.randint(0, 10, num_elements)
    test_hnsw_search(data, ids, docids=docids)
    test_hnsw_search(data, ids)
