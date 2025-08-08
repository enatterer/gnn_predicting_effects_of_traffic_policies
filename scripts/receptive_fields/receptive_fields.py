#!/usr/bin/env python3
"""
receptive_field.py

Compute k-hop receptive fields for either individual seeds or one big set of seeds,
or perform layer-wise neighbor sampling (“fanouts”) à la GraphSAGE.

example usage:
python receptive_fields.py /home/abasu/gnn_predicting_effects_of_traffic_policies/data/inductive_data/training_data/aschaffenburg/000001.pt -k 3 --f 5 5 5 --n 128
"""

import torch
import argparse
import random

###########################################################
n_runs = 10000
###########################################################

def load_data(path):
    data = torch.load(path)
    if not hasattr(data, 'edge_index'):
        raise ValueError(f"No edge_index found in {path}")
    return data

def build_adj_mixed(edge_index, num_nodes):
    """
    Build adjacency for a graph where some edges are undirected (reciprocal pairs)
    and others are directed (one-way).
    """
    all_edges = set(map(tuple, edge_index.t().tolist()))
    processed = set()
    adj = {i: set() for i in range(num_nodes)}
    for u, v in all_edges:
        if (u, v) in processed:
            continue
        if (v, u) in all_edges:
            adj[u].add(v)
            adj[v].add(u)
            processed.update({(u, v), (v, u)})
        else:
            adj[u].add(v)
            processed.add((u, v))
    return adj

def bfs_receptive_field(adj, start_node, max_hop):
    visited = {start_node}
    frontier = {start_node}
    receptive = set()
    for _ in range(max_hop):
        nxt = set()
        for u in frontier:
            nxt |= adj[u]
        nxt -= visited
        if not nxt:
            break
        receptive |= nxt
        visited |= nxt
        frontier = nxt
    return receptive

def bfs_receptive_field_multi(adj, start_nodes, max_hop):
    visited = set(start_nodes)
    frontier = set(start_nodes)
    receptive = set()
    for _ in range(max_hop):
        nxt = set()
        for u in frontier:
            nxt |= adj[u]
        nxt -= visited
        if not nxt:
            break
        receptive |= nxt
        visited |= nxt
        frontier = nxt
    return receptive

def average(lst):
    return sum(lst) / len(lst)

def main():
    parser = argparse.ArgumentParser(
        description="Compute k-hop receptive fields or layer-wise sampling on a PyG data.pt graph"
    )
    parser.add_argument('data_path',
        help="Path to the data.pt file containing a PyG Data object")
    parser.add_argument('--hops', '-k',
        type=int, nargs='+', default=[1,2],
        help="List of hop-distances to evaluate, e.g. `-k 1 2 3`")
    parser.add_argument('--print-nodes',
        action='store_true',
        help="If set, list the actual neighbor node IDs for each receptive field or sample")
    parser.add_argument('--seed-nodes', '-s',
        type=int, nargs='+',
        help="Explicit list of node IDs to use as seeds (default: all nodes)")
    parser.add_argument('--num-seeds', '-n',
        type=int,
        help="If set, randomly pick this many seeds from [0..num_nodes-1]")
    parser.add_argument('--min-neighbors', '-m',
        type=int, nargs='+', default=[0],
        help=(
            "List of minimum neighbor-count thresholds, one per hop. "
            "E.g. `-k 1 2 3 -m 5 10 20`, or a single value to apply to all hops."
        ))
    parser.add_argument('--fanouts', '-f',
        type=int, nargs='+',
        help=(
            "If set, do layer-wise sampling instead of full BFS: "
            "one fanout per hop, e.g. `-k 3 -f 15 10 5`"
        ))

    args = parser.parse_args()
    
    if len(args.hops) ==1:
        max_hop = args.hops[0]
        args.hops = list(range(1, max_hop + 1))

    # load graph
    data = load_data(args.data_path)
    E = data.edge_index
    N = getattr(data, 'num_nodes', int(E.max().item()) + 1)
    adj = build_adj_mixed(E, N)

    # pick seeds
    if args.seed_nodes and args.num_seeds:
        parser.error("Cannot use both --seed-nodes and --num-seeds")
    if args.seed_nodes:
        seeds = args.seed_nodes
    elif args.num_seeds is not None:
        if args.num_seeds > N:
            parser.error("--num-seeds > total nodes")
        seeds = random.sample(range(N), args.num_seeds)
        print(f"Selected {len(seeds)} random seeds: {sorted(seeds)}")
    else:
        seeds = list(range(N))

    # normalize min-neighbors thresholds
    th = args.min_neighbors
    if len(th) == 1:
        thresholds = th * len(args.hops)
    elif len(th) == len(args.hops):
        thresholds = th
    else:
        parser.error(f"--min-neighbors needs 1 or {len(args.hops)} values (got {len(th)})")

    # layer-wise sampling if requested
    if args.fanouts:
        if len(args.fanouts) != len(args.hops):
            parser.error(f"--fanouts ({len(args.fanouts)}) must match #hops ({len(args.hops)})")

        def sample_subgraph(adj, seeds, fanouts):
            all_nodes = set(seeds)
            frontier = set(seeds)
            layer_sizes = []
            for fanout in fanouts:
                nxt = set()
                for u in frontier:
                    neigh = list(adj[u])
                    if len(neigh) > fanout:
                        sampled = random.sample(neigh, fanout)
                    else:
                        sampled = neigh
                    nxt.update(sampled)
                frontier = nxt
                all_nodes |= frontier
                layer_sizes.append(len(frontier))
            return all_nodes, layer_sizes

        # collect raw metrics
        subgraph_sizes    = []        # will hold ints
        layer_sizes_runs  = []        # will hold lists of ints

        for _ in range(n_runs):
            subgraph, sizes = sample_subgraph(adj, seeds, args.fanouts)
            subgraph_sizes.append(len(subgraph))
            layer_sizes_runs.append(sizes)

        # 1) average total subgraph size
        avg_subgraph_size = average(subgraph_sizes)

        # 2) average per-layer sizes
        #    assume every sizes list has length = len(args.fanouts)
        num_layers = len(args.fanouts)
        avg_layer_sizes = []
        for layer_idx in range(num_layers):
            # pull out the layer_idx entry from each run’s sizes
            vals = [ run_sizes[layer_idx] for run_sizes in layer_sizes_runs ]
            avg_layer_sizes.append(average(vals))

        print("Mean subgraph size over", n_runs, "runs:", avg_subgraph_size)
        print("Mean layer sizes:", avg_layer_sizes)

        descriptor = " → ".join(f"{h}-hop@{f}" for h, f in zip(sorted(args.hops), args.fanouts))
        print(f"\n=== Layer-wise sampling: {descriptor} ===")
        for hop, sz, thr in zip(sorted(args.hops), avg_layer_sizes, thresholds):
            status = "OK" if sz >= thr else "BELOW"
            print(f"  Layer {hop}: sampled {sz} unique neighbors (threshold {thr}) → {status}")
        print(f"\nTotal nodes in sampled subgraph (including seeds): {avg_subgraph_size}")

        if args.print_nodes:
            print("Nodes:", sorted(subgraph))
        return

    # otherwise full-breadth BFS
    if args.num_seeds is not None:
        # collective BFS from all seeds
        for idx, hop in enumerate(sorted(args.hops)):
            thr = thresholds[idx]
            rf = bfs_receptive_field_multi(adj, seeds, hop)
            cnt = len(rf)
            sub_sz = cnt + len(seeds)
            if cnt < thr:
                print(f"\n=== {hop}-hop: only {cnt} < {thr}, skipping ===")
                continue
            label = f"{hop}-Hop from {len(seeds)} seeds"
            if args.print-nodes:
                print(f"\n{label} → {cnt} neighbors → subgraph size {sub_sz}")
                print(sorted(rf))
            else:
                print(f"{label} → {cnt} neighbors → subgraph size {sub_sz}")
    else:
        # individual BFS per seed
        for idx, hop in enumerate(sorted(args.hops)):
            thr = thresholds[idx]
            print(f"\n=== {hop}-Hop on {len(seeds)} seeds individually ===")
            for node in seeds:
                rf = bfs_receptive_field(adj, node, hop)
                cnt = len(rf)
                sub_sz = cnt + 1
                if cnt < thr:
                    continue
                if args.print_nodes:
                    print(f"Node {node}: {cnt} → subgraph size {sub_sz} → {sorted(rf)}")
                else:
                    print(f"Node {node}: {cnt} → subgraph size {sub_sz}")

if __name__ == '__main__':
    main()
