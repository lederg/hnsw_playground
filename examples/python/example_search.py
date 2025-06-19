import hnswlib
import numpy as np
import pickle
import fire

from utils import compute_similarity_and_l2, compute_similarity_and_l2_with_unique


def main(use_multivector=True, dim=128, num_elements=10000, k=1, ef=50, numdocs=10):
    if use_multivector:
        spacename = 'multivector'
    else:
        spacename = 'l2'

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))
    # normalize it
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    ids = np.arange(num_elements)
    if use_multivector:
        docids = np.random.randint(0,numdocs,num_elements)
    else:   # if not using multivector, use ids as docids
        docids = None
    # Declaring index
    p = hnswlib.Index(space=spacename, dim=dim)  # possible options are l2, cosine, multivector or ip

    p.set_num_threads(1)


    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Element insertion (can be called several times):
    # p.add_items(data, ids)
    p.add_items(data, ids, num_threads=1, docids_=docids)


    # Controlling the recall by setting ef:
    p.set_ef(ef)  # ef should always be > k

    # rc = p.get_items([0,1])
    # print('Got items [0,1]:')
    # print(rc)

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k, use_docids=use_multivector)


    if use_multivector:
        bf_labels, bf_distances = compute_similarity_and_l2_with_unique(data, data, docids, k=k)
    else:
        # If not using multivector, we can use the simpler function
        bf_labels, bf_distances = compute_similarity_and_l2(data, data, k=k)

    # Compute recall per row, then average
    recall = np.mean([
        np.isin(labels[i], bf_labels[i]).mean()
        for i in range(labels.shape[0])
    ])
    print(f"Recall: {recall:.4f}")

    if use_multivector:
        total_uniqs = np.unique(np.apply_along_axis(lambda row: np.unique(row).size, axis=1, arr=docids[labels]))
        is_uniq_docs = len(total_uniqs) == 1
    else:
        is_uniq_docs = None


    # Index objects support pickling
    # WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
    # Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
    p_copy = pickle.loads(pickle.dumps(p))  # creates a copy of index p using pickle round-trip

    ### Index parameters are exposed as class properties:
    print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
    print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
    print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")
    print(f"Unique docs: {is_uniq_docs}")


if __name__ == "__main__":
    fire.Fire(main)
