# Future Long-Term Active Days Prediction
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from collections import Counter
from scipy.ndimage import convolve1d
from torch.nn.parameter import Parameter, UninitializedParameter
from Model.FDS import FDS
from sklearn import preprocessing
from Model.Tool import setMaskFLTADP

Tensor = torch.Tensor


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= (weights + 0.1).expand_as(loss)
    loss = torch.mean(loss)
    return loss


def BCE(
        input: Tensor,
        target: Tensor):
    # input:   [batch_size, future_day, label_size]
    # target:  [batch_size,future_day, label_size]
    # loss:    [batch_size, future_day,label_size]
    loss = F.binary_cross_entropy(input, target, reduction='none')
    return loss.mean()


class MLADP(torch.nn.Module):
    def __init__(self, model_param):
        super(MLADP, self).__init__()
        print('run MLTADP')
        self.epoch = 0
        self.coefficient = torch.nn.Parameter(torch.Tensor([1.]))
        self.day = model_param["day"]
        self.future_day = model_param["future_day"]
        self.day_numpy_train = model_param["day_numpy_train"]
        self.device = model_param["device"]
        self.week_embedding_size = model_param["week_embedding_size"]
        self.week_num = model_param["week_num"]
        self.day_num = model_param["day_num"]
        self.a_feat_size = model_param['a_feat_size']
        self.u_feat_size = model_param['u_feat_size']
        self.a_field_size = model_param['a_field_size']
        self.u_field_size = model_param['u_field_size']
        self.batch_size = model_param['batch_size']
        self.imbalance_stratage_enable = model_param["imbalance_stratage_enable"]
        self.multi_task_enable = model_param['multi_task_enable']
        self.fine_grained = model_param['fine_grained']
        self.hidden_period_size = model_param['hidden_period_size']
        self.period_num = model_param['period_num']
        self.bce_weight = model_param['bce_weight']
        # Embedding Layer
        self.embedding_size = model_param['embedding_size']
        self.u_embedding_size = model_param['u_embedding_size']
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.u_embedding_size)
        #self.day_embedding_matrix = Parameter(torch.empty((self.future_day, self.embedding_size)))
        # week_embedding
        self.week_embeddings = torch.nn.Embedding(self.week_num, self.week_embedding_size)
        # day_embedding
        self.day_embeddings = torch.nn.Embedding(self.day_num, self.week_embedding_size)
        # user_MLP
        self.context_size = model_param['context_size']
        self.fc1 = torch.nn.Linear(self.u_field_size * self.u_embedding_size, self.u_embedding_size)
        # period
        self.fc2 = torch.nn.Linear(self.period_num, 1)

        # Attention
        self.num_attention_head=model_param['num_attention_head']
        self.multihead_attn_user = torch.nn.MultiheadAttention(num_heads=self.num_attention_head,
                                                          embed_dim=self.u_embedding_size)

        self.multihead_attn=torch.nn.MultiheadAttention(num_heads=self.num_attention_head,
                              embed_dim=self.embedding_size)
        # LSTM
        self.hidden_size = model_param['hidden_size']
        self.lstm = torch.nn.LSTM(input_size=self.a_field_size * self.embedding_size, hidden_size=self.hidden_size,
                                  bidirectional=True)
        self.lstm_period = torch.nn.LSTM(input_size=self.week_embedding_size, hidden_size=self.hidden_period_size,
                                  bidirectional=True)
        # Attention
        self.multihead_attn_1 = torch.nn.MultiheadAttention(num_heads=self.num_attention_head,
                                                          embed_dim=self.week_embedding_size,
                                                            kdim=self.week_embedding_size,
                                                            vdim=2 * self.hidden_size)

        if self.fine_grained == 1:
            self.logic_layer_pred_1 = torch.nn.Linear(
                2 * self.hidden_size +  self.u_embedding_size + 2 * self.hidden_period_size, self.a_field_size)
            self.maxpool_1 = torch.nn.MaxPool1d(kernel_size=self.a_field_size)
        else:
            self.logic_layer_pred_1 = torch.nn.Linear(
                self.day * 2 * self.hidden_size +  self.u_field_size * self.u_embedding_size + self.week_embedding_size, 1)
            self.maxpool_1 = torch.nn.MaxPool1d(kernel_size=1)
        self.maxpool_2 = torch.nn.MaxPool1d(kernel_size=self.week_embedding_size)

        if (self.multi_task_enable != 0):
            if self.fine_grained == 1:
                #a1 = self.future_day * self.a_field_size
                a1 = self.future_day
            else :
                a1 = self.future_day
            a2 = self.future_day * 2 * self.hidden_size
            #a3 = self.future_day * self.week_embedding_size
            a3 = self.future_day
            a4 =  self.u_embedding_size
            # print(a1)
            # print(a2)
            # print(a3)
            # print(a4)
            self.logic_layer_2 = torch.nn.Linear(a1+a2+a3+a4, 1)
        else:
            a1 = self.day * 2 * self.hidden_size
            a2 = self.u_embedding_size
            self.logic_layer_2 = torch.nn.Linear(a1+a2, 1)
        # Attention
        self.multihead_attn_period = torch.nn.MultiheadAttention(num_heads=self.num_attention_head,
                                                          embed_dim=self.week_embedding_size)

        self.multihead_attn_pf = torch.nn.MultiheadAttention(num_heads=self.num_attention_head,
                                                            embed_dim=2 * self.hidden_period_size,
                                                            kdim=2 * self.hidden_period_size,
                                                            vdim=2 * self.hidden_size)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # sigmoid
        self.sig = torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        # create time_interval
        days = torch.tensor([i for i in range(1, self.day + 1)])
        futures = torch.tensor([i for i in range(self.day + 1, self.day + self.future_day + 1)])
        # time_intervel [future_day,day]
        self.time_intervel = torch.stack([i.item() - days for i in futures])
        # time_intervel [batch_size, future_day, day]
        self.time_intervel = torch.unsqueeze(self.time_intervel, 0).repeat(self.batch_size, 1, 1).to(self.device)
        self.logic_layer_s = torch.nn.Linear(2*self.hidden_period_size,1)
        # MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)
        # init
        self.init_embeddings()
        self.init_param()
    def init_embeddings(self):
        print("init_embeddings!")
        nn.init.kaiming_normal_(self.a_embeddings.weight)
        nn.init.kaiming_normal_(self.u_embeddings.weight)
        #nn.init.kaiming_normal_(self.day_embedding_matrix)
        nn.init.kaiming_normal_(self.week_embeddings.weight)
        nn.init.kaiming_normal_(self.day_embeddings.weight)
    def init_param(self):
        print("init_param!")
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.logic_layer_pred_1.weight)
        nn.init.constant_(self.logic_layer_pred_1.bias, 0)
        nn.init.kaiming_normal_(self.logic_layer_2.weight)
        nn.init.constant_(self.logic_layer_2.bias, 0)
        nn.init.kaiming_normal_(self.logic_layer_s.weight)
        nn.init.constant_(self.logic_layer_s.bias,0)
    def forward(self, ui, uv, ai, av, y_1=None, y_2=None, epoch = 0,time = None):
        # batch_size: 32
        # embedding_size: 64
        # time ：   [batch_size,past_day+future_day,4(year,month,day,week)]
        # ai :    [batch_size, day, a_field_size]
        # av :    [batch_size, day, a_field_size]
        # ui :    [batch_size, u_field_size]
        # uv :    [batch_size, u_field_size]
        # y_1 : [batch_size,future_day,a_field_size || 1]

        if self.fine_grained==0:
            y_1 = y_1.sum(dim=2)
            # print(y_1)
            one = torch.ones_like(y_1)
            zero = torch.zeros_like(y_1)
            y_1 = torch.where(y_1 == 0, zero, one)
            y_1 = y_1.reshape((self.batch_size,self.future_day,1))

        y_2_labels = torch.argmax(y_2,1)/(y_2.shape[1])
        # user_embedding
        # u_emb : [batch_size, u_field_size, embedding_size]
        u_emb = self.u_embeddings(ui)
        # u_emb: [batch_size, u_field_size, embeding_size]
        u_emb = torch.multiply(u_emb, uv.reshape(-1, self.u_field_size, 1))
        # self-Attention
        # [u_field_size,batch_size,,u_embeding_size]
        u_emb = u_emb.permute(1, 0, 2)
        u_inter,_ = self.multihead_attn_user(u_emb,u_emb,u_emb)
        # [batch_size ,u_field_size,,u_embeding_size]
        u_emb = u_emb.permute(1, 0, 2)
        # [batch_size, u_field_size * u_embeding_size]
        u_inter = u_inter.reshape(self.batch_size,-1)
        # [batch_size, u_embeding_size]
        u_inter = self.relu(self.fc1(u_inter))

        # day and week emb :
        # weeks  [batch_size , day+future_day ]
        weeks = (time[:, :, 3] - 1).cpu().numpy()
        days = (time[:, :, 2] - 1).cpu().numpy()
        # week and days embedding
        weeks = torch.tensor(weeks).long()
        weeks = weeks.to(self.device)
        days = torch.tensor(days).long()
        days = days.to(self.device)
        # weeks_emb: [batch_size,day+future_day,week_embeddings]
        weeks_emb = self.week_embeddings(weeks)
        # days_emb: [batch_size,day+future_day,week_embeddings]
        days_emb = self.day_embeddings(days)
        # days_emb: [batch_size,day+future_day,period_num,day_embeddings]
        period_emb = torch.cat((weeks_emb.unsqueeze(2), days_emb.unsqueeze(2)), dim=2)
        # Attention
        # period_emb : [batch_size * (day+future_day),period_num, week_embeddings]
        period_emb = period_emb.reshape(-1, self.period_num, self.week_embedding_size)
        # period_emb : [period_num,batch_size * (day+future_day), week_embeddings]
        period_emb = period_emb.permute(1, 0, 2)
        period_mul, period_weight = self.multihead_attn_period(period_emb, period_emb, period_emb)
        # [batch_size * (day + future_day), week_embeddings,period_num ]
        period_mul = period_mul.permute(1, 2, 0)
        # [batch_size , (day + future_day), week_embeddings]
        period_X = self.relu(self.fc2(period_mul)).squeeze(-1).reshape(self.batch_size, -1, self.week_embedding_size)
        # LSTM
        # period_X : [(day + future_day),batch_size, week_embeddings]
        period_X = period_X.permute((1, 0, 2))
        # period_output:[day+future_day, batch_size, num_directions * self.hidden_period_size]
        period_output, (h1_n, c1_n) = self.lstm_period(period_X)
        # period_output:[batch_size , (day+future_day), num_directions * self.hidden_period_size]
        period_output = period_output.permute((1, 0, 2))

        # self-Attention + LSTM
        # a_emb : [batch_size, day , a_field_size, embedding_size]
        a_emb = self.a_embeddings(ai)
        # a_emb: [batch_size, day, a_field_size, embeding_size]
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))
        # a_emb: [batch_size*day, a_field_size, embeding_size]
        a_emb = a_emb.reshape(-1, self.a_field_size, self.embedding_size)
        # a_emb: [a_field_size, batch_size*day,  embeding_size]
        a_emb = a_emb.permute(1,0,2)
        #a_output:[a_field_size,batch_size*day,  embeding_size]
        a_output, a_weights=self.multihead_attn(a_emb,a_emb,a_emb)
        # a_output:[batch_size*day,a_field_size,  embeding_size]
        a_output = a_output.permute(1,0,2)
        # a_output:[batch_size,day, a_field_size*embeding_size]
        a_output=a_output.reshape(self.batch_size,self.day,-1)
        # LSTM
        # a_output:[self.day, batch_size, self.a_field_size * self.embedding_size]
        a_output = a_output.permute(1, 0, 2)
        # a_output:[self.day, batch_size, num_directions * self.hidden_size]
        a_output, (h_n, c_n) = self.lstm(a_output)
        # output:[batch_size，self.day, num_directions * self.hidden_size]
        a_output = a_output.permute(1, 0, 2)

        if (self.multi_task_enable != 0):

            # period_past:[batch_size,past_day, num_directions * self.hidden_period_size]
            period_past = period_output[:, :self.day, :]
            # period_future:[batch_size ,future_day, num_directions * self.hidden_period_size]
            period_future = period_output[:, self.day:, :]
            # period_past:[past_day,batch_size, num_directions * self.hidden_period_size]
            period_past = period_past.permute((1,0,2))
            # period_future:[future_day, batch_size , num_directions * self.hidden_period_size]
            period_future = period_future.permute((1,0,2))
            # output:[self.day,batch_size， num_directions * self.hidden_size]
            a_output = a_output.permute(1, 0, 2)
            # y_weights : [batch_size , future_day ,day]
            _, y_weights = self.multihead_attn_pf(period_future, period_past, a_output)
            # output:[batch_size，self.day, num_directions * self.hidden_size]
            a_output = a_output.permute(1, 0, 2)
            # period_future:[batch_size,future_day, num_directions * self.hidden_period_size]
            period_future = period_future.permute((1,0,2))
            # S: [batch_size , future_day, self.day]
            S = torch.sigmoid(self.logic_layer_s(period_future)).repeat(1,1,self.day)
            #time_intervel_weight [batch_size , future_day, day]
            time_intervel_weight = torch.exp((self.time_intervel*S))
            # all_weight [batch_size, future_day, day]
            all_weight = y_weights + time_intervel_weight
            # y_deep_1 [batch_size , future_day, num_directions * self.hidden_size]
            y_deep_1 = torch.bmm(all_weight,a_output)
            # u_inter_us [batch_size, future_day,  u_embeding_size]
            u_inter_us = torch.unsqueeze(u_inter,1).repeat(1,self.future_day,1)
            # input_1 [batch_size, future_day, num_directions * self.hidden_size + u_embeding_size+num_directions * self.hidden_period_size]
            input_1_1 = torch.cat((y_deep_1,u_inter_us,period_future),dim=2)
            # pred_1 [batch_size, future_day, a_feat_size || 1]
            pred_1 = torch.sigmoid(self.logic_layer_pred_1(input_1_1))

            # input_2_1 [batch_size, future_day]
            input_2_1 = self.maxpool_1(pred_1).squeeze(-1)
            # input_2_2  [batch_size, future_day * num_directions * self.hidden_size]
            input_2_2 = y_deep_1.reshape((self.batch_size, -1))
            # input_2_3 [batch_size , future_day , 1 ]
            input_2_3 = self.maxpool_2(period_future).reshape(self.batch_size, -1)
            #input_2_4 [batch_size,  u_embeding_size]
            input_2_4 = u_inter
            input_cat=torch.cat((input_2_1,input_2_2,input_2_3,input_2_4),dim=1)
            pred_2 = torch.sigmoid(self.logic_layer_2(input_cat)).squeeze(-1)
        else:
            if self.fine_grained == 1:
                pred_1 = torch.ones((self.batch_size,self.future_day,self.a_field_size))
            else:
                pred_1 = torch.ones((self.batch_size, self.future_day, 1))
            # a_output:[batch_size，self.day * num_directions * self.hidden_size]
            a_output = a_output.reshape(self.batch_size, -1)
            y_deep = torch.cat((a_output,u_inter),dim=1)
            pred_2 = torch.sigmoid(self.logic_layer_2(y_deep)).squeeze(-1)

        if y_1 is not None:
            loss = self.criterion_1(pred_2, y_2_labels)
            if (self.multi_task_enable != 0):
                #loss += self.bce_weight * BCE(pred_1, y_1[:,self.day:,:])
                loss += self.bce_weight * BCE(pred_1, y_1[:, :, :])

            filtered_y_1, filtered_y_2, filtered_pred1, filtered_pred2 = [], [], [], []
            return loss, pred_1, pred_2, filtered_y_1, filtered_y_2, filtered_pred1, filtered_pred2