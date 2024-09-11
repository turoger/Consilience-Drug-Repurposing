import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
import uuid
from typing import *
import logging
import json
import sys
#import wandb
import time

from ext.data.data_utils import create_vocab, load_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list
from ext.utils import execute_one_program, get_programs, get_adj_mat, create_sparse_adj_mats
from numpy.random import default_rng

rng = default_rng()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_paths(args, train_adj_list, start_node, max_len=3):
    """
    :param start_node:
    :param K:
    :param max_len:
    :return:
    """
    all_paths = set()
    for k in range(args.num_paths_to_collect):
        path = []
        curr_node = start_node
        entities_on_path = set([start_node])
        for l in range(max_len):
            outgoing_edges = train_adj_list[curr_node]
            if args.prevent_loops:
                # Prevent loops
                temp = []
                for oe in outgoing_edges:
                    if oe[1] in entities_on_path:
                        continue
                    else:
                        temp.append(oe)
                outgoing_edges = temp
            if len(outgoing_edges) == 0:
                break
            # pick one at random
            out_edge_idx = rng.integers(0, len(outgoing_edges), size=1)
            out_edge = outgoing_edges[out_edge_idx[0]]
            path.append(out_edge)
            curr_node = out_edge[1]  # assign curr_node as the node of the selected edge
            entities_on_path.add(out_edge[1])
        all_paths.add(tuple(path))

    return all_paths


def combine_path_splits(data_dir, file_prefix=None):
    combined_paths = defaultdict(list)
    # combined_paths = []
    file_names = []
    for f in tqdm(os.listdir(data_dir)):
        if os.path.isfile(os.path.join(data_dir, f)):
            if file_prefix is not None:
                if not f.startswith(file_prefix):
                    continue
            file_names.append(f)
    for f in file_names:
        #logger.info("Reading file name: {}".format(os.path.join(data_dir, f)))
        with open(os.path.join(data_dir, f), "rb") as fin:
            paths = pickle.load(fin)
            # combined_paths.append(paths)
            for k, v in paths.items():
                combined_paths[k] = v
    # import pdb
    # pdb.set_trace()
    return combined_paths


def get_paths_parallel(args, kg_file, out_dir, job_id=0, total_jobs=1):
    """
    :param kg_file:
    :return:
    """
    unique_entities = get_unique_entities(kg_file)
    num_entities = len(unique_entities)
    logger.info("Total num unique entities are {}".format(num_entities))
    num_entities_in_partition = num_entities / total_jobs
    st = job_id * num_entities_in_partition
    en = min(st + num_entities_in_partition, num_entities)
    logger.info("Starting a job with st ind {} and end ind {}".format(st, en))
    logger.info("Creating adj list")
    train_adj_list = create_adj_list(kg_file, args.add_inv_edges)
    logger.info("Done creating...")
    st_time = time.time()
    paths_map = defaultdict(list)
    for ctr, e1 in enumerate(tqdm(unique_entities)):
        if st <= ctr < en:
            paths = get_paths(args, train_adj_list, e1, args.max_len)
            if paths is None:
                continue
            paths_map[e1] = paths
            if args.use_wandb and (ctr - st) % 100 == 0:
                wandb.log({"progress": (ctr - st) / num_entities_in_partition})

    logger.info("Took {} seconds to collect paths for {} entities".format(time.time() - st_time, len(paths_map)))
    out_file_name = "paths_" + str(args.num_paths_to_collect) + "_path_len_" + str(args.max_len) + "_" + str(job_id)
    if args.prevent_loops:
        out_file_name += "_no_loops"
    out_file_name += ".pkl"
    fout = open(os.path.join(out_dir, out_file_name), "wb")
    logger.info("Saving at {}".format(os.path.join(out_dir, out_file_name)))
    pickle.dump(paths_map, fout)
    fout.close()


def combine_precision_maps(args, dir_name, output_dir_name, output_file_name="precision_map.pkl"):
    """
    Combines all the individual maps
    :param dir_name:
    :return:
    """
    all_numerator_maps, all_denominator_maps = [], []
    combined_numerator_map, combined_denominator_map = {}, {}
    logger.info("Combining precision map located in {}".format(dir_name))
    for filename in os.listdir(dir_name):
        if filename.endswith("_precision_map.pkl"):
            logger.info("Reading filename {}".format(filename))
            with open(os.path.join(dir_name, filename), "rb") as fin:
                count_maps = pickle.load(fin)
                all_numerator_maps.append(count_maps["numerator_map"])
                all_denominator_maps.append(count_maps["denominator_map"])
    assert len(all_numerator_maps) == len(all_denominator_maps)
    for numerator_map, denominator_map in zip(all_numerator_maps, all_denominator_maps):
        for e, _ in numerator_map.items():
            c = args.cluster_assignments[e]
            for r, _ in numerator_map[e].items():
                for path, s_c in numerator_map[e][r].items():
                    if c not in combined_numerator_map:
                        combined_numerator_map[c] = {}
                    if r not in combined_numerator_map[c]:
                        combined_numerator_map[c][r] = {}
                    if path not in combined_numerator_map[c][r]:
                        combined_numerator_map[c][r][path] = 0
                    combined_numerator_map[c][r][path] += numerator_map[e][r][path]

        for e, _ in denominator_map.items():
            c = args.cluster_assignments[e]
            for r, _ in denominator_map[e].items():
                for path, s_c in denominator_map[e][r].items():
                    if c not in combined_denominator_map:
                        combined_denominator_map[c] = {}
                    if r not in combined_denominator_map[c]:
                        combined_denominator_map[c][r] = {}
                    if path not in combined_denominator_map[c][r]:
                        combined_denominator_map[c][r][path] = 0
                    combined_denominator_map[c][r][path] += denominator_map[e][r][path]
    # now calculate precision
    ratio_map = {}
    for c, _ in combined_numerator_map.items():
        for r, _ in combined_numerator_map[c].items():
            if c not in ratio_map:
                ratio_map[c] = {}
            if r not in ratio_map[c]:
                ratio_map[c][r] = {}
            for path, s_c in combined_numerator_map[c][r].items():
                try:
                    ratio_map[c][r][path] = s_c / combined_denominator_map[c][r][path]
                except ZeroDivisionError:
                    import pdb
                    pdb.set_trace()

    output_filenm = os.path.join(output_dir_name, output_file_name)
    logger.info("Dumping ratio map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump(ratio_map, fout)
    logger.info("Done...")


def calc_precision_map_parallel(args, dir_name, job_id=0, total_jobs=1):
    """
    Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when executed
    to how many times the path was executed.
    Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
    :return:
    """
    logger.info("Calculating precision map")
    success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
    # not sure why I am getting RuntimeError: dictionary changed size during iteration.
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in args.train_map.items()]
    # sort this list so that every job gets the same list for processing
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in sorted(train_map, key=lambda item: item[0])]
    job_size = len(train_map) / total_jobs
    st = job_id * job_size
    en = min((job_id + 1) * job_size, len(train_map))
    logger.info("Start of partition: {}, End of partition: {}".format(st, en))
    for e_ctr, ((e1, r), e2_list) in tqdm(enumerate(train_map)):
        if e_ctr < st or e_ctr >= en:
            # not this partition
            continue
        if e_ctr % 100 == 0:
            logger.info("Processing entity# {}".format(e_ctr))
        c = args.entity_vocab[e1]  # calculate stats for each entity
        if c not in success_map:
            success_map[c] = {}
        if c not in total_map:
            total_map[c] = {}
        if r not in success_map[c]:
            success_map[c][r] = {}
        if r not in total_map[c]:
            total_map[c][r] = {}
        if r in args.path_prior_map_per_entity[c]:
            paths_for_this_relation = args.path_prior_map_per_entity[c][r]
        else:
            continue  # if a relation is missing from prior map, then no need to calculate precision for that relation.
        for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
            ans_vec = execute_one_program(args.sparse_adj_mats, args.entity_vocab, e1, path)
            ans = [args.rev_entity_vocab[d_e] for d_e in np.nonzero(ans_vec)[0]]
            if len(ans) == 0:
                continue
            # execute the path get answer
            if path not in success_map[c][r]:
                success_map[c][r][path] = 0
            if path not in total_map[c][r]:
                total_map[c][r][path] = 0
            for a in ans:
                if a in e2_list:
                    success_map[c][r][path] += 1
                total_map[c][r][path] += 1
    output_filenm = os.path.join(dir_name, "{}_precision_map.pkl".format(job_id))
    logger.info("Dumping precision map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump({"numerator_map": success_map, "denominator_map": total_map}, fout)
    logger.info("Done...")


def combine_prior_maps(args, dir_name, output_dir, output_file_name="path_prior_map.pkl"):
    all_program_maps = []
    combined_program_maps = {}
    logger.info("Combining prior maps located in {}".format(dir_name))
    for filename in os.listdir(dir_name):
        if filename.endswith("_path_prior_map.pkl"):
            logger.info("Reading {}".format(filename))
            with open(os.path.join(dir_name, filename), "rb") as fin:
                program_maps = pickle.load(fin)
                all_program_maps.append(program_maps)

    for program_map in all_program_maps:
        for e, _ in program_map.items():
            c = args.cluster_assignments[e]
            for r, _ in program_map[e].items():
                for path, s_c in program_map[e][r].items():
                    if c not in combined_program_maps:
                        combined_program_maps[c] = {}
                    if r not in combined_program_maps[c]:
                        combined_program_maps[c][r] = {}
                    if path not in combined_program_maps[c][r]:
                        combined_program_maps[c][r][path] = 0
                    combined_program_maps[c][r][path] += program_map[e][r][path]

    for c, _ in combined_program_maps.items():
        for r, path_counts in combined_program_maps[c].items():
            sum_path_counts = 0
            for p, p_c in path_counts.items():
                sum_path_counts += p_c
            for p, p_c in path_counts.items():
                combined_program_maps[c][r][p] = p_c / sum_path_counts

    output_filenm = os.path.join(output_dir, output_file_name)
    logger.info("Dumping ratio map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump(combined_program_maps, fout)
    logger.info("Done...")


def calc_prior_path_prob_parallel(args, output_dir_name, job_id=0, total_jobs=1):
    """
    Calculate how probable a path is given a query relation, i.e P(path|query rel)
    For each entity in the graph, count paths that exists for each relation in the
    random subgraph.
    :return:
    """
    logger.info("Calculating prior map")
    programs_map = {}
    job_size = len(args.train_map) / total_jobs
    st = job_id * job_size
    en = min((job_id + 1) * job_size, len(args.train_map))
    logger.info("Start of partition: {}, End of partition: {}".format(st, en))
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in args.train_map.items()]
    # sort this list so that every job gets the same list for processing
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in sorted(train_map, key=lambda item: item[0])]
    for e_ctr, ((e1, r), e2_list) in enumerate(tqdm((train_map))):
        if e_ctr < st or e_ctr >= en:
            # not this partition
            continue
        if e_ctr % 100 == 0:
            logger.info("Processing entity #{}".format(e_ctr))
        c = args.entity_vocab[e1]  # calculate stats for each entity
        if c not in programs_map:
            programs_map[c] = {}
        if r not in programs_map[c]:
            programs_map[c][r] = {}
        all_paths_around_e1 = args.all_paths[e1]
        nn_answers = e2_list
        for nn_ans in nn_answers:
            programs = get_programs(e1, nn_ans, all_paths_around_e1)
            for p in programs:
                p = tuple(p)
                if len(p) == 1:
                    if p[0] == r:  # don't store query relation
                        continue
                if p not in programs_map[c][r]:
                    programs_map[c][r][p] = 0
                programs_map[c][r][p] += 1

    output_filenm = os.path.join(output_dir_name, "{}_path_prior_map.pkl".format(job_id))
    logger.info("Dumping path prior pickle at {}".format(output_filenm))

    with open(output_filenm, "wb") as fout:
        pickle.dump(programs_map, fout)

    logger.info("Done...")


def calc_sim(adj_mat: torch.Tensor, query_entities: torch.LongTensor) -> torch.Tensor:
    """
    :param adj_mat: N X R
    :param query_entities: b is a batch of indices of query entities
    :return:
    """
    query_entities_vec = torch.index_select(adj_mat, dim=0, index=query_entities)
    sim = torch.matmul(query_entities_vec, torch.t(adj_mat))
    return sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    # data specific args
    parser.add_argument("--dataset_name", type=str, default="obl2021")
    parser.add_argument("--data_dir", type=str, default="/home/rajarshi/Dropbox/research/Open-BIo-Link/")
    parser.add_argument("--expt_dir", type=str, default="../prob_cbr_expts/")
    parser.add_argument("--subgraph_file_name", type=str, default="combined_paths_10000_len_3_no_loops.pkl")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='',
                        help="Useful to switch between test files for FB122")
    parser.add_argument("--sim_batch_size", type=int, default=128,
                        help="Batch size to use when doing ent-ent similarity")
    parser.add_argument("--k_adj", type=int, default=100,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    # properties of paths
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=4)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1, help="prevent sampling of looped paths")
    parser.add_argument("--add_inv_edges", action="store_true")
    # preprocessing args
    parser.add_argument("--create_vocab", action="store_true")
    parser.add_argument("--combine_paths", action="store_true")
    parser.add_argument("--calculate_precision_map_parallel", action="store_true",
                        help="If on, calculate precision maps")
    parser.add_argument("--calculate_prior_map_parallel", action="store_true",
                        help="If on, calculate precision maps")
    parser.add_argument("--calculate_ent_similarity", action="store_true",
                        help="If on, calculate precision maps")
    parser.add_argument("--get_paths_parallel", action="store_true",
                        help="If on, collect paths around entities...")
    parser.add_argument("--combine_precision_map", action="store_true",
                        help="If on, combine precision maps")
    parser.add_argument("--combine_prior_map", action="store_true",
                        help="If on, combine prior maps")
    parser.add_argument("--do_clustering", action="store_true")
    # parallel jobs
    parser.add_argument("--total_jobs", type=int, default=50,
                        help="Total number of jobs")
    parser.add_argument("--current_job", type=int, default=0,
                        help="Current job id")
    parser.add_argument("--name_of_run", type=str, default="unset")
    # Clustering args
    parser.add_argument("--linkage", type=float, default=0.8,
                        help="Clustering threshold")
    # Wandb
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=1, help="Set to 1 if using W&B")

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='pr-cbr')
    assert 0 <= args.current_job < args.total_jobs and args.total_jobs > 0
    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.expt_dir, "outputs", args.dataset_name, args.name_of_run)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {args.output_dir}")

    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name, "paths_{}".format(args.num_paths_to_collect))
    kg_file = os.path.join(data_dir, "full_graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                   "graph.txt")
    args.dev_file = os.path.join(data_dir, "dev.txt")
    args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
        else os.path.join(data_dir, args.test_file_name)

    args.train_file = os.path.join(data_dir, "graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                      "train.txt")
    if args.get_paths_parallel:
        kg_file = os.path.join(data_dir, "graph.txt")
        if not os.path.exists(subgraph_dir):  # subgraph dir is the output dir
            os.makedirs(subgraph_dir)
        get_paths_parallel(args, kg_file, subgraph_dir, args.current_job, args.total_jobs)
        sys.exit(0)

    logger.info("Loading train map")
    train_map = load_data(kg_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    if args.create_vocab:
        entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(kg_file)
        eval_entities = get_unique_entities(args.dev_file)
        test_entities = get_unique_entities(args.test_file)
        eval_entities = eval_entities | test_entities  # take union of the two.
        eval_vocab, eval_rev_vocab = {}, {}
        e_ctr = 0
        for e in eval_entities:
            if e not in entity_vocab:
                continue
            eval_vocab[e] = e_ctr
            eval_rev_vocab[e_ctr] = e
            e_ctr += 1
        logger.info("Saving vocabs...")
        entity_vocab_file = os.path.join(data_dir, "entity_vocab.json")
        rel_vocab_file = os.path.join(data_dir, "relation_vocab.json")
        eval_vocab_file = os.path.join(data_dir, "eval_vocab.json")
        for file_name, vocab in [(entity_vocab_file, entity_vocab), (rel_vocab_file, rel_vocab),
                                 (eval_vocab_file, eval_vocab)]:
            logger.info("Writing {}".format(file_name))
            with open(file_name, "w") as fin:
                json.dump(vocab, fin)
        sys.exit(0)
    logger.info("Loading vocabs...")
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
    # making these part of args for easier access #hack
    args.entity_vocab = entity_vocab
    args.rel_vocab = rel_vocab
    args.rev_entity_vocab = rev_entity_vocab
    args.rev_rel_vocab = rev_rel_vocab
    args.train_map = train_map
    args.dev_map = dev_map
    args.test_map = test_map
    adj_mat = get_adj_mat(kg_file, entity_vocab, rel_vocab)
    logger.info("Building sparse adjacency matrices")
    args.sparse_adj_mats = create_sparse_adj_mats(args.train_map, args.entity_vocab, args.rel_vocab)
    if args.calculate_ent_similarity:
        logger.info("Calculating entity similarity matrix...")
        adj_mat = torch.from_numpy(adj_mat)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        logger.info('Using device: {}'.format(device.__str__()))
        adj_mat = adj_mat.to(device)
        # Changed
        adj_mat= adj_mat.cpu()
        query_ind = []
        for i in range(len(eval_vocab)):
            query_ind.append(entity_vocab[eval_rev_vocab[i]])
        query_ind = torch.LongTensor(query_ind).to(device)
        # Changed
        query_ind = query_ind.cpu()
        # Calculate similarity
        st = 0
        sim = None
        arg_sim = None
        batch_sim, batch_sim_ind = None, None
        while st < query_ind.shape[0]:
            en = min(st + args.sim_batch_size, query_ind.shape[0])
            logger.info("st: {}, en: {}, query_ind.shape[0]: {}".format(st, en, query_ind.shape[0]))
            batch_sim = calc_sim(adj_mat, query_ind[st:en])  # n X N (n== size of dev_entities, N: size of all entities)
            #batch_sim_sorted = np.sort(-batch_sim, axis=-1)
            #batch_sim = batch_sim.detach().cpu().numpy()
            batch_sim_ind = np.argsort(-batch_sim, axis=-1)
            batch_sim_ind = batch_sim_ind[:, :args.k_adj]
            batch_sim_sorted = None
            for i in range(batch_sim.shape[0]):
                batch_sim_sorted = batch_sim[i, batch_sim_ind[i, :]] if batch_sim_sorted is None else np.vstack(
                    [batch_sim_sorted, batch_sim[i, batch_sim_ind[i, :]]])
            if sim is None:
                sim = batch_sim_sorted
                arg_sim = batch_sim_ind
            else:
                sim = np.vstack([sim, batch_sim_sorted])
                arg_sim = np.vstack([arg_sim, batch_sim_ind])
            st = en
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name)
        ent_sim_dict_file = os.path.join(dir_name, "ent_sim.pkl")
        logger.info("Writing {}".format(ent_sim_dict_file))
        with open(ent_sim_dict_file, "wb") as fout:
            pickle.dump({"sim": sim, "arg_sim": arg_sim}, fout)
        sys.exit(0)
    if args.do_clustering:
        if args.linkage > 0:
            raise NotImplementedError
            # if os.path.exists(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl")):
            #     logger.info("Clustering with linkage {} found, loading them....".format(args.linkage))
            #     fin = open(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl"), "rb")
            #     args.cluster_assignments = pickle.load(fin)
            #     fin.close()
            # else:
            #     logger.info("Clustering entities with linkage = {}...".format(args.linkage))
            #     args.cluster_assignments = cluster_entities(adj_mat, args.linkage)
            #     logger.info("There are {} unique clusters".format(np.unique(args.cluster_assignments).shape[0]))
            #     dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage))
            #     if not os.path.exists(dir_name):
            #         os.makedirs(dir_name)
            #     logger.info("Dumping cluster assignments of entities at {}".format(dir_name))
            #     fout = open(os.path.join(dir_name, "cluster_assignments.pkl"), "wb")
            #     pickle.dump(args.cluster_assignments, fout)
            #     fout.close()
        else:
            args.cluster_assignments = np.zeros(adj_mat.shape[0])
            dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping cluster assignments of entities at {}".format(dir_name))
            fout = open(os.path.join(dir_name, "cluster_assignments.pkl"), "wb")
            pickle.dump(args.cluster_assignments, fout)
            fout.close()

    dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage))
    logger.info(
        "Loading cluster assignments of entities from {}".format(os.path.join(dir_name, "cluster_assignments.pkl")))
    with open(os.path.join(dir_name, "cluster_assignments.pkl"), "rb") as fin:
        args.cluster_assignments = pickle.load(fin)

    if args.calculate_prior_map_parallel:
        logger.info(
            "Calculating prior map. Current job id: {}, Total jobs: {}".format(args.current_job, args.total_jobs))
        logger.info("Loading subgraph around entities:")
        file_prefix = "paths_{}_path_len_{}_".format(args.num_paths_to_collect, args.max_len)
        all_paths = combine_path_splits(subgraph_dir, file_prefix=file_prefix)
        logger.info("Done...")
        args.all_paths = all_paths
        assert args.all_paths is not None
        assert args.train_map is not None
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "per_entity_maps", "prior_maps",
                                "path_{}".format(args.num_paths_to_collect))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        calc_prior_path_prob_parallel(args, dir_name, args.current_job, args.total_jobs)

    if args.combine_prior_map:
        assert args.cluster_assignments is not None
        input_dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                      "per_entity_maps", "prior_maps",
                                      "path_{}".format(args.num_paths_to_collect))
        output_dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage), "prior_maps",
                                       "path_{}".format(args.num_paths_to_collect))
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)
        combine_prior_maps(args, input_dir_name, output_dir_name)

    if args.calculate_precision_map_parallel:
        logger.info(
            "Calculating precision map. Current job id: {}, Total jobs: {}".format(args.current_job, args.total_jobs))
        assert args.train_map is not None
        logger.info("Loading prior map...")
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "per_entity_maps")
        per_entity_prior_map_dir = os.path.join(dir_name, "prior_maps", "path_{}".format(args.num_paths_to_collect))
        args.path_prior_map_per_entity = combine_path_splits(per_entity_prior_map_dir)
        assert args.path_prior_map_per_entity is not None
        dir_name = os.path.join(dir_name, "precision_maps", "path_{}".format(args.num_paths_to_collect))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        calc_precision_map_parallel(args, dir_name, args.current_job, args.total_jobs)

    if args.combine_precision_map:
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "per_entity_maps", "precision_maps",
                                "path_{}".format(args.num_paths_to_collect))
        output_dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage), "precision_maps",
                                       "path_{}".format(args.num_paths_to_collect))
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)
        combine_precision_maps(args, dir_name, output_dir_name)