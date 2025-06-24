import hnswlib
import numpy as np
import pickle
import fire

from utils import compute_similarity_and_l2, compute_similarity_and_l2_with_unique


def main(use_multivector=True, dim=128, num_elements=10000, k=3, ef=50, numdocs=10, num_query=100):
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
        print (f"docids[0,1]: {docids[0:2]}")
    else:   # if not using multivector, use ids as docids
        docids = None
    # Declaring index
    p = hnswlib.Index(space=spacename, dim=dim)  # possible options are l2, cosine, multivector or ip
    bfp = hnswlib.BFIndex(space="l2", dim=dim)

    p.set_num_threads(1)
    bfp.set_num_threads(1)


    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    bfp.init_index(max_elements=num_elements)

    # Element insertion (can be called several times):
    # p.add_items(data, ids)
    p.add_items(data, ids, num_threads=1, docids_=docids)
    bfp.add_items(data, ids)

    # Controlling the recall by setting ef:
    p.set_ef(ef)  # ef should always be > k


    # query = data[:num_query]  # take first num_query elements as query
    query = np.float32(np.random.random((num_query, dim)))
    # normalize it
    query = query / np.linalg.norm(query, axis=1, keepdims=True)

    # rc = query[0:2]
    # print('Querying with items [0,1]:')
    # print(rc)
    # rc = p.get_items([0,1])
    # print('Got items [0,1]:')
    # print(rc)

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(query, k=k, use_docids=use_multivector)
    bf_labels, bf_distances = bfp.knn_query(query, k=num_elements)

    bf_unique = []

    for i in range(num_query):
        unique_docs = set()
        query_labels = []
        for j in range(num_elements):
            if use_multivector:
                if docids[bf_labels[i][j]] not in unique_docs:
                    unique_docs.add(docids[bf_labels[i][j]])
                    query_labels.append(bf_labels[i][j])
            else:
                query_labels.append(bf_labels[i][j])
            if len(query_labels) >= k:
                break
        bf_unique.append(query_labels)

            
    bf_labels = np.array(bf_unique)

    # if use_multivector:
    #     bf_labels, bf_distances = compute_similarity_and_l2_with_unique(data, data, docids, k=k)
    # else:
    #     # If not using multivector, we can use the simpler function
    #     bf_labels, bf_distances = compute_similarity_and_l2(data, data, k=k)

    # Compute recall per row, then average
    recall = np.mean([
        np.isin(labels[i], bf_labels[i]).mean()
        for i in range(labels.shape[0])
    ])
    print(f"Recall: {recall:.4f}")

    if use_multivector:
        total_uniqs = np.unique(np.apply_along_axis(lambda row: np.unique(row).size, axis=1, arr=docids[labels]))
        is_uniq_docs = len(total_uniqs) == 1
        bf_total_uniqs = np.unique(np.apply_along_axis(lambda row: np.unique(row).size, axis=1, arr=docids[bf_labels]))
        is_bf_uniq_docs = len(bf_total_uniqs) == 1

    else:
        is_uniq_docs = None
        is_bf_uniq_docs = None


    rc = p.get_items([0,1])
    print('Got items [0,1]:')
    print(rc)



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
    print(f"Unique docs in brute force: {is_bf_uniq_docs}")


if __name__ == "__main__":
    fire.Fire(main)
