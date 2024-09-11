import scipy.sparse
import numpy as np
from typing import *
from ext.data.data_utils import read_graph


def get_adj_mat(kg_file, entity_vocab, rel_vocab):
    adj_mat = read_graph(kg_file, entity_vocab, rel_vocab)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)
    return adj_mat


def get_programs(e: str, ans: str, all_paths_around_e: List[List[str]]):
    """
    Given an entity and answer, get all paths which end at that ans node in the subgraph surrounding e
    """
    all_programs = []
    for path in all_paths_around_e:
        for l, (r, e_dash) in enumerate(path):
            if e_dash == ans:
                # get the path till this point
                all_programs.append([x for (x, _) in path[:l + 1]])  # we only need to keep the relations
    return all_programs


def execute_one_program(sparse_adj_mats: Dict[str, scipy.sparse.csr_matrix], entity_vocab: Dict[str, int], e: str,
                        path: List[str]) -> np.ndarray:
    """
    starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
    max_branch number.
    """
    src_vec = np.zeros((len(entity_vocab), 1), dtype=np.uint32)
    src_vec[entity_vocab[e]] = 1
    ent_vec = scipy.sparse.csr_matrix(src_vec)
    for r in path:
        ent_vec = sparse_adj_mats[r] * ent_vec
    final_counts = ent_vec.toarray().reshape(-1)
    return final_counts


def create_sparse_adj_mats(train_map, entity_vocab, rel_vocab):
    sparse_adj_mats = {}
    csr_data, csr_row, csr_col = {}, {}, {}
    for (e1, r), e2_list in train_map.items():
        _ = csr_data.setdefault(r, [])
        _ = csr_row.setdefault(r, [])
        _ = csr_col.setdefault(r, [])
        for e2 in e2_list:
            csr_data[r].append(1)
            csr_row[r].append(entity_vocab[e2])
            csr_col[r].append(entity_vocab[e1])
    for r in rel_vocab:
        sparse_adj_mats[r] = scipy.sparse.csr_matrix((np.array(csr_data[r], dtype=np.uint32),  # data
                                                      (np.array(csr_row[r], dtype=np.int64),  # row
                                                       np.array(csr_col[r], dtype=np.int64)))  # col
                                                     , shape=(len(entity_vocab), len(entity_vocab)))
    return sparse_adj_mats