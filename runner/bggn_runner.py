from __future__ import (division, print_function)
import os
import time
import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import wandb
import nni

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel

from utils.metrics import print_all_metrics, print_all_metrics_new

try:
  ###
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
  ###
except:
  pass

os.environ["WANDB_API_KEY"] = "cdabb5eadbb48428d86c9a7d3e6f03c0fac536f3"
os.environ["WANDB_MODE"] = "dryrun"

logger = get_logger('exp_logger')
__all__ = ['BggnRunner', 'compute_edge_ratio', 'get_graph', 'evaluate']

NPR = np.random.RandomState(seed=1234)

def compute_edge_ratio(G_list):
  num_edges_max, num_edges = .0, .0
  for gg in G_list:
    num_nodes = gg.number_of_nodes()
    num_edges += gg.number_of_edges()
    num_edges_max += num_nodes**2

  ratio = (num_edges_max - num_edges) / num_edges
  return ratio


def get_graph(adj):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  adj = adj[~np.all(adj == 0, axis=1)]
  adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  # add: remove large adj matrix's uselss nodes
  G.remove_nodes_from(list(nx.isolates(G)))
  return G


def evaluate(graph_gt, graph_pred, degree_only=True):
  mmd_degree = degree_stats(graph_gt, graph_pred)

  if degree_only:
    mmd_4orbits = 0.0
    mmd_clustering = 0.0
    mmd_spectral = 0.0
  else:    
    mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
    mmd_clustering = clustering_stats(graph_gt, graph_pred)    
    mmd_spectral = spectral_stats(graph_gt, graph_pred)
    
  return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral


class BggnRunner(object):

  def __init__(self, config):
    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.writer = SummaryWriter(config.save_dir)
    self.is_vis = config.test.is_vis
    self.better_vis = config.test.better_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.is_single_plot = config.test.is_single_plot
    self.num_gpus = len(self.gpus)
    self.is_shuffle = False

    assert self.use_gpu == True

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)
    
    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride
    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))

    # add
    hyperparameter_defaults = dict(
        lr = self.train_conf.lr,
        bpr_weight = self.train_conf.bpr_weight,
        num_GNN_layers = self.model_conf.num_GNN_layers
        )
    if self.config.search:
      hyperparameter_defaults = nni.get_next_parameter()
    wandb.init(config=hyperparameter_defaults, project="BGGN")
    #  self.wandb_config = wandb.config
    self.train_conf.lr = wandb.config.lr
    self.config.train.bpr_weight = wandb.config.bpr_weight
    self.config.model.num_GNN_layers = wandb.config.num_GNN_layers

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      #  self.npr.shuffle(self.graphs)

    #  self.graphs_train = self.graphs[:self.num_train]
    #  self.graphs_dev = self.graphs[:self.num_dev]
    #  self.graphs_test = self.graphs[self.num_train:]
    
    #  self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs.values()) # self.graphs is a dict
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    #  self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs.values()]) # self.graphs is a dict
    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()
    
    ### save split for benchmarking
    if config.dataset.is_save_split:      
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)
      
      save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

  def train(self):
    ### create data loader
    #  train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs, tag='train')
    train_dataset = eval(self.dataset_conf.train_loader_name)(self.config, self.graphs, tag='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    #  import pdb; pdb.set_trace()
    if self.use_gpu:
      model = DataParallel(model, device_ids=self.gpus).to(self.device)

    #  wandb.watch(model)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam': 
      #  optimizer = optim.Adam(params, lr=self.wandb_config.lr, weight_decay=self.train_conf.wd)
      optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_epoch,
        gamma=self.train_conf.lr_decay)
    #  lr_scheduler = optim.lr_scheduler.StepLR(
        #  optimizer,
        #  step_size=self.train_conf.lr_decay_size,
        #  gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch

    # Training Loop
    iter_count = 0    
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      #  lr_scheduler.step()
      train_iterator = train_loader.__iter__()

      #  import pdb; pdb.set_trace()
      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1
        
        
        avg_train_loss = .0        
        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []
          
          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              #  data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['pos_idx_feat'] = batch_data[dd][ff]['pos_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              # add
              #  data['user'] = batch_data[dd][ff]['user'].pin_memory().to(gpu_id, non_blocking=True)
              data['user_idx'] = batch_data[dd][ff]['user_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              batch_fwd.append((data,))

          if batch_fwd:
            train_loss = model(*batch_fwd).mean()              
            avg_train_loss += train_loss              
            # add
            #  if inner_iter == 0:
                #  self.writer.add_graph(model, (*batch_fwd,))

            # assign gradient
            train_loss.backward()
        
        # clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        # add: restrict
        #  model.named_parameters()[size_penalty].data.clamp_(0, 1)

        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
        
        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())
        #  train_loss = float(avg_train_loss)
        
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
          logger.info("NLL Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, train_loss))

          # add: log
          self.writer.add_scalar('train_loss', train_loss, iter_count)
          wandb.log({'train_loss': train_loss})
          if self.config.search:
            nni.report_intermediate_result(train_loss)

      # change position 
      lr_scheduler.step()

       # add: log
      self.writer.add_scalar('train_loss_epoch_final', train_loss, epoch)
      self.writer.add_scalar('train_loss_epoch_mean', np.mean(results['train_loss']), epoch)
      wandb.log({'train_loss_epoch_final': train_loss})
      wandb.log({'train_loss_epoch_mean': np.mean(results['train_loss'])})
      #  best_loss = max(np.mean(results['train_loss']), best_loss)
      if self.config.search:
        nni.report_final_result(np.mean(results['train_loss']))

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
    
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    
    return 1

  def test(self):
    # add
    #  test_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs, tag='train')
    test_dataset = eval(self.dataset_conf.test_loader_name)(self.config, self.graphs, tag='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self.test_conf.batch_size,
        #  shuffle=self.train_conf.shuffle,
        num_workers=self.test_conf.num_workers,
        collate_fn=test_dataset.collate_fn,
        drop_last=False)

    self.config.save_dir = self.test_conf.test_model_dir

    ### Compute Erdos-Renyi baseline    
    if self.config.test.is_test_ER:
      p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / sum([aa.number_of_nodes() ** 2 for aa in self.graphs_train])      
      graphs_gen = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in range(self.num_test_gen)]
    else:
      ### load model
      model = eval(self.model_conf.name)(self.config)
      model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
      load_model(model, model_file, self.device)

      if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

      model.eval()

      ### Generate Graphs
      A_pred = []
      num_nodes_pred = []
      num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

      #  gen_run_time = []
      #  for ii in tqdm(range(num_test_batch)):
      #    with torch.no_grad():
      #      start_time = time.time()
      #      input_dict = {}
      #      input_dict['is_sampling']=True
      #      input_dict['batch_size']=self.test_conf.batch_size
      #      input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
      #      #  A_tmp = model(input_dict)
      #      graphs_gen = model(input_dict)
      #      gen_run_time += [time.time() - start_time]
      #      #  A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
      #      #  num_nodes_pred += [aa.shape[0] for aa in A_tmp]
      
      test_res_path = os.path.join(self.config.save_dir, 'test_result.p')
      if not os.path.exists(test_res_path):

        # add
        gen_run_time = []
        res = []
        all_graphs = []
        #  for metric in metrics:
            #  metric.start()
        with torch.no_grad():
            #  xx = 0
            for users, ground_truth in tqdm(test_loader):
                #  xx += 1
                if len(ground_truth[0]) == 0:
                    continue
                #  if xx == 645:
                    #  import pdb; pdb.set_trace()

                input_dict = {}
                input_dict['is_sampling']=True
                input_dict['batch_size']=self.test_conf.batch_size
                input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
                input_dict['user_for_test']=users
                start_time = time.time()
                graphs_gen, nodes_gen = model(input_dict)
                #  graphs_gen, nodes_gen, candidates_gen = model(input_dict)
                gen_run_time += [time.time() - start_time]
                #  for metric in metrics:
                    #  metric(graphs_gen, ground_truth_u_b.to(device))
                #  import pdb; pdb.set_trace()
                res += [(ground_truth[0], nodes_gen)] # if batch_size = 1
                #  all_graphs += graphs_gen
                all_graphs += [graphs_gen]
                #  if len(res)==2:
                    #  break
        #  for metric in metrics:
            #  metric.stop()
            #  print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
        #  print('')

        logger.info('Average test time per mini-batch = {}'.format(
            np.mean(gen_run_time)))

        pickle.dump((res, all_graphs), open(test_res_path, 'wb'))

      else:
        res, all_graphs = pickle.load(open(test_res_path, 'rb'))

      #  import pdb; pdb.set_trace()
      #  flag, k, prec, recall, ndcg, mrr, jacc, total = print_all_metrics_new(
      flag, k, prec, recall, ndcg, mrr, jacc, total, best_prec, best_index = print_all_metrics_new(
          self.dataset_conf.name, res, self.test_conf.topk)
      logger.info('%s\tPre@%d: %.4f%%\tRecall@%d: %.4f%%\tNDCG@%d: %.4f%%\tMRR@%d: %.4f%%\tDiv: %.4f' 
                  % (flag, k, prec*100/total, k, recall*100/total, 
                     k, ndcg*100/total, k, mrr*100/total, -jacc/total)
                  )

      res_5 = [(re[0], re[1][:self.test_conf.topk1]) for re in res] 
      #  flag, k, prec, recall, ndcg, mrr, jacc, total = print_all_metrics_new(
      flag, k, prec, recall, ndcg, mrr, jacc, total, _, _ = print_all_metrics_new(
          self.dataset_conf.name, res_5, self.test_conf.topk1)
      logger.info('%s\tPre@%d: %.4f%%\tRecall@%d: %.4f%%\tNDCG@%d: %.4f%%\tMRR@%d: %.4f%%\tDiv: %.4f' 
                  % (flag, k, prec*100/total, k, recall*100/total, 
                     k, ndcg*100/total, k, mrr*100/total, -jacc/total)
                  )
 
      
      graphs_gen = all_graphs
          
      #  graphs_gen = [get_graph(aa) for aa in A_pred]

    ### Visualize Generated Graphs
    if self.is_vis:
      num_col = self.vis_num_row
      num_row = int(np.ceil(self.num_vis / num_col))
      test_epoch = self.test_conf.test_model_name
      test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
      save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(self.config.test.test_model_name[:-4], test_epoch, self.block_size, self.stride))

      # remove isolated nodes for better visulization
      graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:self.num_vis]]
      #  print(best_prec)
      #  import pdb; pdb.set_trace()
      #  graphs_pred_vis = [copy.deepcopy(graphs_gen[gg]) for gg in best_index[:self.num_vis]]
      #  best_truth = [res[i][0] for i in best_index[:self.num_vis]]
      #  best_nodes = [res[i][1] for i in best_index[:self.num_vis]]
      # best_user has 10 graphs
      graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[best_index[0]]]
      best_truth = res[best_index[0]][0] 
      best_nodes = res[best_index[0]][1]
      #  best_candidates = [candidates_gen[i] for i in best_index[:self.num_vis]]
      #  best_labels = [{i:best_candidates[i]} for i in range(len(best_candidates))]
      #  best_labels = [{i:best_nodes[i]}  i for i in range(len(best_nodes))]
      best_labels = []
      #  for best_node in best_nodes:
      #  for best_node in best_nodes:
          #  labels_dict = {}
          #  for node in range(len(best_node)):
              #  labels_dict[node] = best_node[node]
          #  best_labels += [labels_dict]
      for best_graph,best_node in zip(graphs_pred_vis, best_nodes):
          labels_dict = {}
          assert len(best_node) == len(best_graph.nodes)
          for tmp_ii,tmp_jj in zip(best_node, best_graph.nodes):
              labels_dict[tmp_jj] = tmp_ii
          best_labels += [labels_dict]

      #  if self.better_vis:
        #  for gg in graphs_pred_vis:
          #  gg.remove_nodes_from(list(nx.isolates(gg)))
      # Error: networkx.exception.NetworkXError: Frozen graph can't be modified
      # already have done once in graph_gen, so now comment them temporaryly.

      # display the largest connected component for better visualization
      #  import pdb; pdb.set_trace()
      vis_graphs = []
      #  vis_labels = []
      #  for gg in graphs_pred_vis:
      for ii,gg in enumerate(graphs_pred_vis):        
        CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        vis_graphs += [CGs[0]]
        #  labels_dict = {}
        #  if len(CGs[0].nodes) != 0:
            #  for node in CGs[0].nodes:
                #  labels_dict[node] = best_labels[ii][node]
        #  vis_labels += [labels_dict]

      #  import pdb; pdb.set_trace()
      if self.is_single_plot:
        draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
      else:
        #  draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')
        #  draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='shell')
        draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='shell', labels=best_labels)

      save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

      if self.is_single_plot:
        draw_graph_list(
            self.graphs_train[:self.num_vis],
            num_row,
            num_col,
            fname=save_name,
            layout='spring')
      else:      
        draw_graph_list_separate(
            #  self.graphs_train[:self.num_vis],
            list(self.graphs.values())[:self.num_vis], # self.graphs is a dict
            fname=save_name[:-4],
            is_single=True,
            #  layout='spring')
            layout='shell')

#      ### Evaluation
    #  if self.config.dataset.name in ['lobster']:
    #    acc = eval_acc_lobster_graph(graphs_gen)
    #    logger.info('Validity accuracy of generated graphs = {}'.format(acc))
    #
    #  num_nodes_gen = [len(aa) for aa in graphs_gen]
    #
    #  # Compared with Validation Set
    #  num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]  # shape B X 1
    #  mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(
    #      self.graphs_dev, graphs_gen, degree_only=False)
    #  mmd_num_nodes_dev = compute_mmd(
    #      [np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)
    #
    #  # Compared with Test Set
    #  num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]  # shape B X 1
    #  mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(
    #      self.graphs_test, graphs_gen, degree_only=False)
    #  mmd_num_nodes_test = compute_mmd(
    #      [np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)
    #
    #  logger.info("Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
    #      mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev))
    #  logger.info("Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(
    #      mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test))
    #
    #  if self.config.dataset.name in ['lobster']:
    #    return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test, acc
    #  else:
      #  return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test

      return 1
