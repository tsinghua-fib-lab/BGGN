import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
  aa, bb = aa&bb, aa|bb
  #  if len(aa) != 0:
     #  print(aa)
     #  print('a:', a)
     #  print('b:', b)
  if len(bb) == 0: return 0.0
  #  return len(aa & bb) / len(aa | bb)
  return len(aa) / len(bb)

def hit_degree(b, b_list):
    return max(jaccard_sim(b, i) for i in b_list)

def Pre(groundtruth, preds):
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for pred in preds:
    prec += hit_degree(pred, groundtruth)
  prec /= len(preds)
  return prec

def Recall(groundtruth, preds):
  recall = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for grou in groundtruth:
    recall += hit_degree(grou, preds)
  recall /= len(groundtruth)
  return recall

#  def NDCG(groundtruth, preds):
  #  prec = 0.0
  #  if not isinstance(groundtruth[0], list):
    #  groundtruth = [groundtruth]
  #  i = 1
  #  for pred in preds:
    #  prec += (2 ** (hit_degree(pred, groundtruth)) - 1)/np.log2(i+1)
    #  i += 1
  #  prec /= len(preds)
  #  return prec

def DCG(scores):
    return np.sum(
        #  np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(len(scores), dtype=np.float32) + 2)),
        dtype=np.float32)

def NDCG(groundtruth, preds):
    #  relevance = np.ones_like(groundtruth)
    #  it2rel = {it: r for it, r in zip(groundtruth, relevance)}
    #  rank_scores = np.asarray([it2rel.get(it, 0.0) for it in preds], dtype=np.float32)
#
    #  idcg = DCG(np.sort(relevance)[::-1])
    #  dcg = DCG(rank_scores)

    if not isinstance(groundtruth[0], list):
      groundtruth = [groundtruth]

    relevance = [hit_degree(i, groundtruth) for i in preds]
    idcg = DCG(np.sort(relevance)[::-1])
    dcg = DCG(relevance)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

def MRR(groundtruth, preds):
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  i = 1
  for pred in preds:
    prec += hit_degree(pred, groundtruth)/i
    i += 1
  prec /= len(preds)
  return prec

def jaccard(preds):
  jac = 0.0
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      jac += jaccard_sim(preds[i], preds[j])
  jac /= len(preds) * (len(preds) - 1) / 2
  return jac

def precision(groundtruth, preds):
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for grou in groundtruth:
    prec_part = 0.0
    for pred in preds:
      # prec_part = max(prec_part, jaccard_sim(pred, grou))
      prec_part += jaccard_sim(pred, grou)
    prec_part /= len(preds)
    prec += prec_part
  prec /= len(groundtruth)
  return prec

def print_all_metrics(flag, res, k=10):

  total = len(res)
  soft_prec, prec, jacc = 0.0, 0.0, 0.0
  #  for groundtruth, preds, scores in res:
  for groundtruth, preds in res:
    preds = preds[:k]

    prec += precision(groundtruth, preds)
    jacc += jaccard(preds)
  print('%s\tP@%d: %.4f%%\tDiv: %.4f'
      % (flag, k, prec*100/total, -jacc/total))

def print_all_metrics_new(flag, res, k=10):

  #  total = len(res)
  total = 0
  best_index = []
  best_prec = []
  prec, recall, ndcg, mrr, jacc = 0.0, 0.0, 0.0, 0.0, 0.0
  #  for groundtruth, preds, scores in res:
  #  for groundtruth, preds in res:
  for i, (groundtruth, preds) in enumerate(res):
    if len(groundtruth) == 0:
        continue
    total += 1  

    best_index += [i]
    best_prec += [Pre(groundtruth, preds)]

    preds = preds[:k]

    prec += Pre(groundtruth, preds)
    recall += Recall(groundtruth, preds)
    ndcg += NDCG(groundtruth, preds)
    mrr += MRR(groundtruth, preds)
    if len(preds) == 1: # topk=1, no diversity
        jacc += 0
    else:
        jacc += jaccard(preds)

  print('%s\tPre@%d: %.4f%%\tRecall@%d: %.4f%%\tNDCG@%d: %.4f%%\tMRR@%d: %.4f%%\tDiv: %.4f'
      % (flag, k, prec*100/total, k, recall*100/total, k, ndcg*100/total, k, mrr*100/total, -jacc/total))

  best_prec, best_index = zip(*sorted(zip(best_prec, best_index), reverse=True, key=lambda x:x[0]))

  #  return flag, k, prec, recall, ndcg, mrr, jacc, total
  return flag, k, prec, recall, ndcg, mrr, jacc, total, best_prec, best_index


if __name__ == '__main__':
  assert len(sys.argv) == 2
  pkl_path = sys.argv[1]

  with open(pkl_path, 'rb') as f:
    res = pickle.load(f)

  k = 10
  flag = "clo"
  if 'ele' in pkl_path:
    flag = 'ele'
  print_all_metrics(flag, res, 10)
  print_all_metrics(flag, res, 5)
