import pickle
import random
import time
import itertools

import networkx as nx
import numpy as np
import tensorflow as tf

random.seed(42)


def nearest_sampler(src_nodes, num_samples, G):
    """

    Args:
        src_nodes:{list} a list of original(target) nodes
        num_samples:{int} sampling number
        G:{networkx.classes.graph.Graph} graph

    Returns:
        sample_list:{list} a list of sampled nodes
    """
    neighbors = []
    for node in src_nodes:
        # sampling with replacement
        neighbors.append(np.random.choice(list(G.neighbors(node)), num_samples, replace=True))

    return list(np.asarray(neighbors).flatten().astype(str))


def multihop_nearest_sampler(src_nodes, num_samples, G):
    """

    Args:
        src_nodes:{list} a list of original(target) nodes; src - > 0-hop nodes
        num_samples:{list} a list of sampling numbers; the number of depth(k-hops) = len(num_samples)
        G:{networkx.classes.graph.Graph} graph

    Returns:
        sampling_results:{list} a list of multihop sampling results

    """
    # B^K set
    sampling_results = [src_nodes]

    # for k = K...1 do
    # B^(k-1) = B^(k)
    for k, hopk_num in enumerate(num_samples):
        hopk_result = nearest_sampler(sampling_results[k], hopk_num, G)
        sampling_results.append(hopk_result)
    sampling_results.reverse()
    return sampling_results


def nearest_sampler_without_replacement(src_nodes, G):
    neighbors = [list(G.neighbors(node)) for node in src_nodes]

    return neighbors


def init_dataset(G):
    # prepare data for loading mini-batches
    node_ids = list(G.nodes)
    node_attributes = [np.squeeze(G.nodes[idx]['attribute']) for idx in node_ids]

    len_attribute = len(node_attributes[0])

    # create indexing table
    keys_tensor = tf.constant(node_ids, dtype=tf.string)
    values_tensor = tf.constant(node_attributes, dtype=tf.float32)
    lookup_table = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.string,
        value_dtype=tf.float32,
        default_value=[-1] * len_attribute,
        empty_key='',
        deleted_key='$')
    lookup_table.insert(keys_tensor, values_tensor)

    return lookup_table


class TripChainGenerator:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False):
        assert batch_size >= 1
        print("Initialize batch generator")
        print("----Dataset Information----")
        print("Classes:", dataset['others']['classes'])
        print("Number of classes:", len(dataset['others']['classes']))
        print(dataset['others']['area'], 'of', dataset['others']['survey'])
        print("-----Graph Information-----")
        print("Number of nodes:", len(dataset['graph']['nodes']))
        print("Number of edges:", len(dataset['graph']['edges']))
        print("Average degrees:", len(dataset['graph']['edges']) / len(dataset['graph']['nodes']))
        print("-----Chain Information-----")
        print("Number of chains:", len(dataset['chains']))
        print("---------------------------")

        print("Create graph")
        st = time.time()
        self.G = nx.Graph()
        self.G.add_nodes_from(dataset['graph']['nodes'])
        self.G.add_edges_from(dataset['graph']['edges'])
        ed = time.time()
        print("Finished in %fs" % (ed - st))
        print("---------------------------")

        self.indices = np.arange(len(dataset['chains']))
        self.chains = dataset['chains']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.node_lookup_table = init_dataset(self.G)

    def _pack_outputs(self, batch_idx):
        chain_ids = []
        labels = []
        region_ids = []
        max_len = 0
        trip_feat_dims = self.chains[0]['trips'].shape[-1]
        region_feat_dims = self.chains[0]['regions']['attributes'].shape[-1]

        for idx in batch_idx:
            chain_ids.append(self.chains[idx]['chain_id'])
            labels.append(self.chains[idx]['labels'])
            region_ids.append(self.chains[idx]['regions']['region_ids'])
            max_len = max(max_len, len(self.chains[idx]['regions']['region_ids']))

        # tag indicies
        tag_indicies = np.zeros((len(batch_idx), max_len - 1), dtype='int32')
        sequence_lengths = np.zeros((len(batch_idx),), dtype='int32')

        for i in range(len(batch_idx)):
            sequence_lengths[i] = len(labels[i])
            tag_indicies[i, :sequence_lengths[i]] = labels[i]

        chain_ids = tuple(chain_ids)
        dropped_region_ids = list(set(list(itertools.chain.from_iterable(region_ids))))

        # graph sampling
        neighbor_nodes = [list(self.G.neighbors(node)) for node in dropped_region_ids]
        graph_contents = []
        max_neighbor = 0
        num_nodes = []
        for src, neighbors in zip(dropped_region_ids, neighbor_nodes):
            num_nodes.append(len(neighbors) + 1)
            max_neighbor = max(max_neighbor, len(neighbors))
            src_node = self.node_lookup_table.lookup(tf.constant(src, dtype=tf.string))
            neighbor_nodes = self.node_lookup_table.lookup(tf.constant(neighbors, dtype=tf.string))
            graph_contents.append(tf.concat([tf.expand_dims(src_node, 0), neighbor_nodes], 0))

        graph_mask = np.zeros((len(num_nodes), max_neighbor + 1), dtype='float32')
        for i, length in enumerate(num_nodes):
            graph_mask[i, :length] = 1

        # padding and concat -> (NUM_SRC, MAX_NODE, DIMS)
        for i, tensor in enumerate(graph_contents):
            if i == 0:
                paddings = tf.constant([
                    [0, max_neighbor + 1 - tf.shape(tensor).numpy()[0]],
                    [0, 0]
                ])
                tensor = tf.pad(tensor, paddings)
                graph_tensor = tf.expand_dims(tensor, 0)
            else:
                paddings = tf.constant([
                    [0, max_neighbor + 1 - tf.shape(tensor).numpy()[0]],
                    [0, 0]
                ])
                tensor = tf.pad(tensor, paddings)
                graph_tensor = tf.concat([graph_tensor, tf.expand_dims(tensor, 0)], 0)

        trip_tensor = np.zeros((len(batch_idx), max_len - 1, trip_feat_dims), dtype='float32')
        region_tensor = np.zeros((len(batch_idx), max_len, region_feat_dims), dtype='float32')
        arrive_tensor = np.zeros((len(batch_idx), max_len), dtype='float32')
        depart_tensor = np.zeros((len(batch_idx), max_len - 1), dtype='float32')

        trip_mask = np.zeros((len(batch_idx), max_len - 1), dtype='float32')
        region_mask = np.zeros((len(batch_idx), max_len), dtype='float32')

        for i, idx in enumerate(batch_idx):
            length = self.chains[idx]['regions']['attributes'].shape[0]
            trip_tensor[i, :length - 1, :] = self.chains[idx]['trips']
            region_tensor[i, :length, :] = self.chains[idx]['regions']['attributes']
            arrive_tensor[i, :length] = self.chains[idx]['arrive']
            depart_tensor[i, :length - 1] = self.chains[idx]['depart']
            trip_mask[i, :length - 1] = 1
            region_mask[i, :length] = 1

        # build indexing dict
        region_table = {}
        for i, region_id in enumerate(dropped_region_ids):
            region_table[region_id] = i + 1

        region_idx_array = np.zeros(((len(self.idx_batches)), max_len))
        for i, region_seq in enumerate(region_ids):
            for j, region_id in enumerate(region_seq):
                region_idx_array[i, j] = region_table[region_id]
        region_idx_array = region_idx_array.astype('int32')

        masks = (trip_mask, region_mask, graph_mask)
        indicies = (sequence_lengths, tag_indicies, region_idx_array)
        outputs = (trip_tensor, region_tensor, graph_tensor)
        time_en = (arrive_tensor, depart_tensor)
        y = (labels, chain_ids)

        return outputs, masks, indicies, time_en, y

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        self.num_steps = self.indices.shape[0] // self.batch_size
        self.split_lst = [self.batch_size] * self.num_steps
        if self.indices.shape[0] % self.batch_size != 0:
            self.split_lst += [self.indices.shape[0] % self.batch_size]
            self.num_steps += 1

        self.start_p = 0
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.num_steps:
            ed_p = self.start_p + self.split_lst[self.step]
            self.idx_batches = self.indices[self.start_p: ed_p]
            outputs = self._pack_outputs(self.idx_batches)
            self.start_p = ed_p
            self.step += 1
            return outputs
        else:
            raise StopIteration


if __name__ == '__main__':
    # test code

    # BATCH PARAMETERS
    with open('../test_samples/tokyo_23_val.pkl', 'rb') as f:
        dataset = pickle.load(f)

    generator = TripChainGenerator(dataset, batch_size=256, shuffle=False)
    g = iter(generator)

    st = time.time()
    for batch in g:
        ed = time.time()
        print(ed - st)
        st = time.time()
