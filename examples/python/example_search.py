import hnswlib
import numpy as np
import pickle
import fire
import os
import json
from datetime import datetime

from utils import *


def main(
    use_multivector=True,
    dim=64,
    num_elements=0,
    blocks_per_doc=10,
    block_size=100,
    k=3,
    max_candidates=6,
    M=16,
    ef=50,
    numdocs=0,
    num_threads=1,
    num_query=100,
    independent = True,
    experiment_path=None
):
    if use_multivector:
        spacename = 'multivector'
    else:
        spacename = 'l2'

    if numdocs <= 0:
        assert num_elements > 0, "If numdocs is not specified, num_elements must be greater than 0."
        numdocs = int(num_elements / (blocks_per_doc * block_size))
    elif num_elements <= 0:
        assert numdocs > 0, "If num_elements is not specified, numdocs must be greater than 0."
        num_elements = blocks_per_doc * numdocs * block_size
    else:
        print("Both num_elements and numdocs are specified, ignoring blocks_per_doc for data generation.")
        blocks_per_doc = num_elements // (numdocs * block_size)

    # Compose experiment parameter string for filenames
    mv_flag = "mv" if use_multivector else "nomv"
    param_str = f"d{dim}_n{num_elements}_M{M}_{mv_flag}_nd{numdocs}"
    exp_dir = f"{experiment_path}_{param_str}" if experiment_path else None

    hnsw_file = os.path.join(exp_dir, "hnsw.bin")
    bf_file = os.path.join(exp_dir, "bf.bin")
    docids_file = os.path.join(exp_dir, "docids.npy")

    if exp_dir and os.path.exists(exp_dir):
        # Load indices and docids from files
        print(f"Loading HNSW index from {hnsw_file}")
        p = hnswlib.Index(space=spacename, dim=dim)
        p.load_index(hnsw_file)
        print(f"Loading brute-force index from {bf_file}")
        bfp = hnswlib.BFIndex(space="l2", dim=dim)
        bfp.load_index(bf_file)
        if use_multivector and os.path.exists(docids_file):
            docids = np.load(docids_file)
            print(f"Loaded docids from {docids_file}")
        else:
            docids = None
    else:
        # Generate data and save to new directory
        if exp_dir:
            os.makedirs(exp_dir, exist_ok=True)
        data, docids = generate_data(
            blocks_per_doc=blocks_per_doc,
            block_size=block_size,
            num_docs=numdocs,
            dim=dim,
            independent=independent
        )
        perm = np.random.permutation(len(data))
        data = data[perm]
        docids = docids[perm] 

        print("Data generated and permuted.")
        ids = np.arange(num_elements)
        if not use_multivector:
            docids = None

        p = hnswlib.Index(space=spacename, dim=dim)
        bfp = hnswlib.BFIndex(space="l2", dim=dim)

        p.set_num_threads(1)
        bfp.set_num_threads(1)

        p.init_index(max_elements=num_elements, ef_construction=200, M=M)
        bfp.init_index(max_elements=num_elements)

        p.add_items(data, ids, num_threads=num_threads, docids_=docids)
        bfp.add_items(data, ids)

        # Save indices and docids
        if exp_dir:
            if os.path.exists(hnsw_file):
                print(f"Warning: {hnsw_file} already exists. Skipping save.")
            else:
                p.save_index(hnsw_file)
                print(f"Saved HNSW index to {hnsw_file}")
            if os.path.exists(bf_file):
                print(f"Warning: {bf_file} already exists. Skipping save.")
            else:
                bfp.save_index(bf_file)
                print(f"Saved brute-force index to {bf_file}")
            if docids is not None:
                if os.path.exists(docids_file):
                    print(f"Warning: {docids_file} already exists. Skipping save.")
                else:
                    np.save(docids_file, docids)
                    print(f"Saved docids to {docids_file}")

    # Generate queries for evaluation
    query = np.float32(np.random.random((num_query, dim)))
    query = query / np.linalg.norm(query, axis=1, keepdims=True)

    # Controlling the recall by setting ef:
    p.set_ef(ef)  # ef should always be > k

    labels, distances = p.knn_query(query, k=k, max_candidates=max_candidates, use_docids=use_multivector)
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

    metric_hops, metric_distance_computations = p.get_metric_stats()
    avg_query_time = p.get_avg_query_time()
    print(f"Metric hops: {metric_hops}, Metric distance computations: {metric_distance_computations}")
    print(f"Average query time: {avg_query_time:.4f} seconds")
    p_copy = pickle.loads(pickle.dumps(p))

    print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
    print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
    print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")
    print(f"Unique docs: {is_uniq_docs}")
    print(f"Unique docs in brute force: {is_bf_uniq_docs}")

    # Save experiment results to JSON file in the experiment directory
    if exp_dir:
        results = {
            "recall": float(recall),
            "parameters": {
                "use_multivector": use_multivector,
                "dim": dim,
                "num_elements": num_elements,
                "k": k,
                "max_candidates": max_candidates,
                "M": M,
                "ef": ef,
                "numdocs": numdocs,
                "blocks_per_doc": blocks_per_doc,
                "num_query": num_query,
            },
            "unique_docs": is_uniq_docs,
            "unique_docs_bf": is_bf_uniq_docs,
            "metric_hops": metric_hops,
            "metric_distance_computations": metric_distance_computations,
            "avg_query_time": avg_query_time,
            "index_size": int(p_copy.element_count),
            "index_capacity": int(p_copy.max_elements),
            "ef_runtime": int(p_copy.ef),
            "command_line": " ".join(os.sys.argv),
        }
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(exp_dir, f"results_{now}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved experiment results to {results_file}")


if __name__ == "__main__":
    fire.Fire(main)
