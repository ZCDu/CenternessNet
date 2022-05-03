import os
import pdb
import torch
import importlib
import torch.nn as nn

from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)


class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss = loss

    def forward(self, xs, ys, gt_bboxes=None, gt_labels=None, **kwargs):
        preds = self.model(*xs, **kwargs)
        centerness = preds.pop()
        loss_kp = self.loss(preds, ys, **kwargs)
        return loss_kp, centerness


# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)


class NetworkFactory(object):
    def __init__(self, db):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)

        self.model = DummyModule(nnet_module.model(db))
        self.loss = nnet_module.loss
        self.centerness_loss = nnet_module.centerness_loss
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(
            self.network, chunk_sizes=system_configs.chunk_sizes).cuda()

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()))
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                    self.model.parameters()),
                                             lr=system_configs.learning_rate,
                                             momentum=0.9,
                                             weight_decay=0.0001)
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def fix_params(self, m):
        classname = m.__class__.__name__
        #print('m:', m.__class__)
        if classname.find('BatchNorm') != -1:
            m.eval()
        #elif classname.find('CenterNessNet') != -1:
        #    m.requires_grad = True
        #else:
        #    m.requires_grad = False

    def train_mode(self):
        self.network.train()
        #for m in self.network.modules():
        #    print('m:', m)
        #    if m.find('BatchNorm') != -1:
        #        m.eval
        #    elif m.find('CenterNessNet') != -1:
        #        m.requires_grad = True
        #    else:
        #        m.requires_grad = False
        for m in self.network.named_parameters():
            if m[0].find('CenterNessNet') != -1:
                m[1].requires_grad = True
            elif m[0].find('tl') != -1:
                m[1].requires_grad = False
            elif m[0].find('br') != -1:
                m[1].requires_grad = False
            elif m[0].find('ct') != -1:
                m[1].requires_grad = False
            else:
                m[1].requires_grad = True
        self.network.apply(self.fix_params)
        for m in self.network.named_parameters():
            print("0:%s,1:%s" % (m[0], m[1].requires_grad))

    def eval_mode(self):
        self.network.eval()

    def train(self, iteration, xs, ys, gt_bboxes, gt_labels, **kwargs):
        xs = [x for x in xs]
        ys = [y for y in ys]
        #gt_bboxes = [box for box in gt_bboxes]
        #gt_labels = [label for label in gt_labels]
        #print('gt_bboxes[0] shape:', gt_labels[0].shape) # 就是由于gt_bboxes这些引起的问题
        #print('gt_bboxes len:', len(gt_labels)) # 就是由于gt_bboxes这些引起的问题
        #print('nnet xs:', xs[0].type())

        if (iteration % 3 == 0):
            self.optimizer.step()
            self.optimizer.zero_grad()
        #output = self.model(*xs, **kwargs)
        #output1 = output.pop()
        #loss_kp = self.loss(output, ys)
        #centerness_loss = self.centerness_loss(output1, gt_bboxes, gt_labels)
        loss_kp, centerness = self.network(xs, ys)
        centerness_loss = self.centerness_loss(centerness, gt_bboxes,
                                               gt_labels)
        loss = loss_kp[0]
        focal_loss = loss_kp[1]
        pull_loss = loss_kp[2]
        push_loss = loss_kp[3]
        regr_loss = loss_kp[4]
        #centerness_loss = loss_kp[5]
        loss = loss + centerness_loss
        loss = loss.mean()
        focal_loss = focal_loss.mean()
        pull_loss = pull_loss.mean()
        push_loss = push_loss.mean()
        regr_loss = regr_loss.mean()
        centerness_loss = centerness_loss.mean()
        #centerness_loss = 0
        loss.backward()
        return loss, focal_loss, pull_loss, push_loss, regr_loss, centerness_loss

    def validate(self, xs, ys, gt_bboxes, gt_labels, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss_kp, centerness = self.network(xs, ys)
            centernesness_loss = self.centerness_loss(centerness, gt_bboxes,
                                                      gt_labels)
            loss = loss_kp[0]
            focal_loss = loss_kp[1]
            pull_loss = loss_kp[2]
            push_loss = loss_kp[3]
            regr_loss = loss_kp[4]
            centerness_loss = centernesness_loss
            loss = loss + centerness_loss
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # 按需载入
    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        #pretrained_model = system_configs.snapshot_file.format(
        #    pretrained_model)
        with open(pretrained_model, "rb") as f:
            pretrained_dict = torch.load(f)
            net_state_dict = self.model.state_dict()
            pretrained_dict1 = {
                k: v
                for k, v in pretrained_dict.items() if k in net_state_dict
            }
            net_state_dict.update(pretrained_dict1)
            self.model.load_state_dict(net_state_dict)

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)
        # add
        #cache_og = system_configs.snapshot_file.format(1)
        #cache_og = system_configs.snapshot_file.format(iteration)
        #print('add origin param {}'.format(cache_og))
        #with open(cache_og, "rb") as f:
        #    param_og = torch.load(f)
        #    net_state_dict = self.model.state_dict()
        #    pretrained_dict1 = {
        #        k: v
        #        for k, v in param_og.items() if k in net_state_dict
        #    }
        #    net_state_dict.update(pretrained_dict1)
        #    self.model.load_state_dict(net_state_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)
