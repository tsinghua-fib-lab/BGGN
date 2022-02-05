###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm

__all__ = [
    'save_graph_list', 'load_graph_list', 'graph_load_batch',
    'preprocess_graph_list', 'create_graphs'
]


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch_1(data_dir,
                     min_num_nodes=2, # our bundle
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  #  G = nx.Graph()

  # load data
  #  load_data_size
  with open(os.path.join(data_dir, name, '{}_data_size.txt'.format(name)), 'r') as f: 
      num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

  #  load_u_b_interaction
  with open(os.path.join(data_dir, name, 'user_bundle_{}.txt'.format('train')), 'r') as f:
      ub_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
  ub_indice = np.array(ub_pairs, dtype=np.int32) 
  ub_values = np.ones(len(ub_pairs), dtype=np.float32) 
  ub_matrix = sp.coo_matrix( 
      (ub_values, (ub_indice[:, 0], ub_indice[:, 1])), shape=(num_users, num_bundles)).tocsr()

  #  load_bi_affiliation
  with open(os.path.join(data_dir, name, 'bundle_item.txt'), 'r') as f:
      bi_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
  bi_indice = np.array(bi_pairs, dtype=np.int32)
  bi_values = np.ones(len(bi_pairs), dtype=np.float32)
  bi_matrix = sp.coo_matrix( 
      (bi_values, (bi_indice[:, 0], bi_indice[:, 1])), shape=(num_bundles, num_items)).tocsr()

  # too slow
  #  data_b2i = np.loadtxt(
      #  os.path.join(data_dir, name, 'bundle_item.txt'), delimiter='\t').astype(int)

  b_popularity = ub_matrix.sum(axis=0)/ub_matrix.shape[0]
  ib_popularity = (sp.diags(b_popularity.A.ravel()) @ bi_matrix).T
  ib_norm = sp.diags(1/(np.sqrt((ib_popularity.multiply(ib_popularity)).sum(axis=1).A.ravel()) + 1e-8)) @ ib_popularity
  ii_corelation = ib_norm @ ib_norm.T # 对称阵，对角线有0有1（0.98）
  #  ii_corelation = ib_norm @ ib_norm.T - sp.identity(ib_norm.shpae[0], dtype='float32') # avoid self loop 

  #  data_adj = np.loadtxt(
      #  os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  #  if node_attributes:
    #  data_node_att = np.loadtxt(
        #  os.path.join(path, '{}_node_attributes.txt'.format(name)),
        #  delimiter=',')
  #  data_node_label = np.loadtxt(
      #  os.path.join(path, '{}_node_labels.txt'.format(name)),
      #  delimiter=',').astype(int)
  #  data_graph_indicator = np.loadtxt(
      #  os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      #  delimiter=',').astype(int)
  #  if graph_labels:
    #  data_graph_labels = np.loadtxt(
        #  os.path.join(path, '{}_graph_labels.txt'.format(name)),
        #  delimiter=',').astype(int)

  #  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  #  G.add_edges_from(data_tuple)
  # add node attributes
  #  for i in range(data_node_label.shape[0]):
    #  if node_attributes:
      #  G.add_node(i + 1, feature=data_node_att[i])
    #  G.add_node(i + 1, label=data_node_label[i])
  #  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  #  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  #  data_graph_indicator = data_b2i
  data_graph_indicator = bi_indice
  #  graph_num = data_graph_indicator.max()
  graph_num = data_graph_indicator[:,0].max() + 1
  #  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  node_list = data_graph_indicator[:,1]
  #  graphs = []
  u_b_graphs = {}
  max_nodes = 0
  for i in tqdm(range(graph_num)):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator[:,0] == i]

    #  import pdb; pdb.set_trace()

    # 1. complete graph
    if name == 'Clothe' or 'Electro':
        G_sub = nx.complete_graph(nodes)

    # 2. non-complete graph via binomial
    elif name == 'Youshu' or 'NetEase':
        corel_in_bundle = ii_corelation[nodes, :][:, nodes].A
        #  corel_in_bundle[range(corel_in_bundle.shape[0]),range(corel_in_bundle.shape[1])]=0
        np.fill_diagonal(corel_in_bundle, 0)
        #  corel_in_bundle = (corel_in_bundle - np.mean(corel_in_bundle))/np.std(corel_in_bundle)
        corel_in_bundle = (corel_in_bundle - np.min(corel_in_bundle))/(np.max(corel_in_bundle) - np.min(corel_in_bundle) + 1e-8)
        corel_in_bundle = np.random.binomial(1, corel_in_bundle)
        edges = np.nonzero(corel_in_bundle)
        edges = [(nodes[i], nodes[j]) for i,j in zip(edges[0], edges[1])]
        G_sub = nx.Graph()
        G_sub.add_edges_from(edges)
        G_sub.remove_edges_from(G_sub.selfloop_edges())
    else:
        raise ValueError(r"data's name is wrong")


    #  G_sub = G.subgraph(nodes)
    #  if graph_labels:  # we can classyfy the bundle, but the dataset can not supply.
      #  G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      #  graphs.append(G_sub)
      u_b_graphs[i] = G_sub
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  #  return graphs
  return u_b_graphs

def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  import pdb; pdb.set_trace()
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs


def create_graphs(graph_type, data_dir='data', noise=10.0, seed=1234):
  npr = np.random.RandomState(seed)
  ### load datasets
  graphs = []
  # synthetic graphs
  if graph_type == 'grid':
    graphs = []
    for i in range(10, 20):
      for j in range(10, 20):
        graphs.append(nx.grid_2d_graph(i, j))    
  elif graph_type == 'lobster':
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80
    num_graphs = 100

    seed_tmp = seed
    while count < num_graphs:
      G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
      if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
        graphs.append(G)
        if G.number_of_edges() > max_edge:
          max_edge = G.number_of_edges()
        
        count += 1

      seed_tmp += 1
  elif graph_type == 'DD':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230

  elif graph_type == 'Youshu' or 'Steam' or 'NetEase' or 'Douban':
    graphs = graph_load_batch_1(
        data_dir,
        min_num_nodes=2,
        max_num_nodes=1000,
        #  name='Youshu',
        name=graph_type,
        node_attributes=False,
        graph_labels=False)

  elif graph_type == 'FIRSTMM_DB':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=0,
        max_num_nodes=10000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)

  # graphs is dict now.
  #  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_nodes = [gg.number_of_nodes() for gg in graphs.values()]
  #  num_edges = [gg.number_of_edges() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs.values()]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
   
  return graphs

