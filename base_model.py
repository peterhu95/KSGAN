import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging
import numpy as np

import os


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst):
        raise NotImplementedError

    def dist(self, src, rel, dst):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst):
        raise NotImplementedError

    def prob(self, src, rel, dst):
        return nnf.softmax(self.prob_logit(src, rel, dst))

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, src_bad, dst_bad, rel_smpl):
        d_good = self.dist(src, rel, dst)
        d_bad = self.dist(src_bad, rel_smpl, dst_bad,False)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, truth):
        probs = self.prob(src, rel, dst)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth] + 1e-30)
        return -truth_probs
        
    def prob_loss(self,src,rel,dst,src_bad,dst_bad,rel_smpl):
        n,m=src_bad.data.shape
        one_bad_score=self.score(src_bad,rel_smpl,dst_bad,False)
        one_good_score=self.score(src,rel,dst)
        # src_cat=torch.cat((src.resize(n,1),src_bad),1)
        # rel_cat=torch.cat((rel.resize(n,1),rel_smpl),1)
        # dst_cat=torch.cat((dst.resize(n,1),dst_bad),1)
        label_truth=torch.ones(n,1)
        label_fault=torch.zeros(n,1)  # m
        label_fault-=1
        label=torch.cat((label_truth,label_fault),1)
        good_bad_scores=torch.cat((one_good_score.resize(n,1),one_bad_score.resize(n,1)),1)
        return nnf.softplus(Variable(label.cuda())*good_bad_scores)  # ,False # self.score(src_cat,rel_cat,dst_cat)


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

    def gen_step(self, src, rel, dst, n_sample=15, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = dst.size()
        rel_var = Variable(rel.cuda())
        src_var = Variable(src.cuda())
        dst_var = Variable(dst.cuda())

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=False)
        sample_srcs = src[row_idx, sample_idx.data.cpu()]
        sample_dsts = dst[row_idx, sample_idx.data.cpu()]
        
        rel_numpy=rel.numpy()
        rel_numpy=rel_numpy[:,:n_sample]
        sample_rels=torch.from_numpy(rel_numpy)
        #print(sample_rels)
        #sample_rels=rel.unsqueeze(1).expand(n,n_sample)
        
        #print()
        #print("sample_rels",sample_rels)
        #print()
        rewards = yield sample_srcs, sample_dsts, sample_rels
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards.cpu()) * log_probs[row_idx.cuda(), sample_idx.data].cpu())# add cpu()
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, rel_smpl, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        rel_smpl_var=Variable(rel_smpl.cuda())
        
        if hasattr(self.mdl,"margin"):
            losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var, rel_smpl_var)
        else:
            losses = self.mdl.prob_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var, rel_smpl_var)
        
        
        fake_scores = self.mdl.score(src_fake_var, rel_smpl_var, dst_fake_var) ##,False
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails,filt=True,write_data=True,epo=0):
        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
            src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
            dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
            all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                               .type(torch.LongTensor).cuda(), volatile=True)
            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var).data
            for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                if filt:
                    if tails[(s, r)]._nnz() > 1:
                        tmp = dst_scores[t]
                        dst_scores += tails[(s, r)].cuda() * 1e30
                        dst_scores[t] = tmp
                    if heads[(t, r)]._nnz() > 1:
                        tmp = src_scores[s]
                        src_scores += heads[(t, r)].cuda() * 1e30
                        src_scores[s] = tmp
                mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                mrr, mr, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit10_tot / count)
        
        if write_data:
            f=open(os.path.join(os.getcwd(),"plotdata",config().task.dir+".txt"),"a")
            write_str=str(mrr_tot / count)+" "+str(hit10_tot / count)+" "+str(epo)+'\n'
            f.write(write_str)
            f.close()
            print("write"+'\n'+write_str+"to "+"plotdata/"+config().task.dir+".txt")
        
        return mrr_tot / count
