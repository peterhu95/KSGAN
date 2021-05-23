import os
import logging
import datetime
import torch
from random import sample, random

from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import heads_tails, inplace_shuffle, batch_by_num
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from simple_avg import SimplE_avg
from logger_init import logger_init
from select_gpu import select_gpu
from corrupter import BernCorrupterMulti


logger_init()
torch.cuda.set_device(select_gpu())
overwrite_config_with_args()
dump_config()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx, 'SimplE_avg':SimplE_avg}
gen_config = config()[config().g_config]
dis_config = config()[config().d_config]
gen = models[config().g_config](n_ent, n_rel, gen_config)
dis = models[config().d_config](n_ent, n_rel, dis_config)

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
filt_heads, filt_tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

dis.load(os.path.join(config().task.dir, "gan_dis_0331071735.mdl"))
dis.test_link(test_data, n_ent, filt_heads, filt_tails,write_data=False)

