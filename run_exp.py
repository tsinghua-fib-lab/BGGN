import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')

import setproctitle
import resource
import gc
import socket

def main():
  args = parse_arguments()
  config = get_config(args.config_file, is_test=args.test)

  config.search = args.search
  config.test.test_model_dir = args.testfile
  if not config.search:
    #  os.environ["CUDA_VISIBLE_DEVICES"] = config.available_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  if args.test:
    config.model.num_GNN_layers = int(args.num_GNN_layers)

  # for device-side assert triggered
  os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  # log info
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  print(">" * 80)
  pprint(config)
  print("<" * 80)

  # Run the experiment
  try:
    runner = eval(config.runner)(config)
    if not args.test:
      setproctitle.setproctitle("train@changjianxin")
      runner.train()
    else:
      setproctitle.setproctitle("inference@changjianxin")
      runner.test()
  except:
    logger.error(traceback.format_exc())

  sys.exit(0)

def restric_memory():
  maxsize = 50*(1024**3) # 50GB
  soft, hard = resource.getrlimit(resource.RLIMIT_AS)
  resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def memory_leak():
  gc.collect()
  max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print("{:.2f} MB".format(max_mem_used / 1024))

if __name__ == "__main__":
  #  restric_memory()
  #  memory_leak()
  main()
