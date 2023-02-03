import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from Model.Tool import setMask
class CFIN(torch.nn.Module):
    def __init__(self, model_param):
        super(CFIN, self).__init__()
        self.device = model_param['device']
        # criterion
        #BCE
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        #MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)
        self.dropout_p = model_param['dropout_p']
        # action user : one_hot_num
        self.a_feat_size = model_param['a_feat_size']
        self.u_feat_size = model_param['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_param['a_field_size']
        self.u_field_size = model_param['u_field_size']
        # conv
        self.conv_size = model_param['conv_size']
        self.context_size = model_param['context_size']
        self.embedding_size = model_param['embedding_size']
        self.kernel_size=model_param['kernel_size']
        self.stride=model_param['stride']
        # DNN layer
        self.deep_layers = model_param['deep_layers']
        # use attn or not
        self.attn_enable = model_param['attn_enable']
        self.attn_size = model_param['attn_size']
        self.batch_norm = model_param['batch_norm']
        self.batch_norm_decay = model_param['batch_norm_decay']
        # use day_num
        self.day=model_param['day']
        # embedding_layer
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)
        self.conv1 = torch.nn.Conv1d(self.embedding_size, self.conv_size, kernel_size=self.kernel_size, stride=self.stride)
        # attention
        self.fc1 = torch.nn.Linear(self.u_field_size * self.embedding_size, self.context_size)
        self.fc2 = torch.nn.Linear(self.conv_size + self.context_size, self.attn_size)
        self.fc3 = torch.nn.Linear(self.attn_size, 1)
        # Relu
        self.relu = torch.nn.ReLU()
        self.dnn = torch.nn.ModuleList()
        # DNN
        self.dnn.append(torch.nn.Linear(self.conv_size + self.context_size, self.deep_layers[0]))
        for i in range(1, len(self.deep_layers)):
            self.dnn.append(torch.nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
        self.logic_layer = torch.nn.Linear(self.deep_layers[-1], 1)
        # batch_norm
        self.batch_norm = torch.nn.BatchNorm1d(self.a_field_size, eps=1e-05, momentum=1 - self.batch_norm_decay,
                                               affine=True,
                                               track_running_stats=True)
        # Softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # dropout
        self.dropout = torch.nn.Dropout(self.dropout_p)
        #param init
        self.init_params()
        self.init_embeddings()
    def init_params(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        for i in range(len(self.deep_layers)):
            nn.init.kaiming_normal_(self.dnn[i].weight)
        nn.init.kaiming_normal_(self.logic_layer.weight)

        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        for i in range(len(self.deep_layers)):
            nn.init.constant_(self.dnn[i].bias,0)
        nn.init.constant_(self.logic_layer.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.a_embeddings.weight)
        nn.init.kaiming_normal_(self.u_embeddings.weight)

    def forward(self, ui, uv, ai, av, y=None,lossFun='BCE'):
        # batch_size: 32
        # embedding_size: 32
        # pre_ai :    [batch_size, day, a_field_size]
        # pre_av :    [batch_size, day, a_field_size]
        # ai : [batch_size,  a_field_size]
        # av : [batch_size,  a_field_size]
        # a_emb : [batch_size, a_field_size, embedding_size]
        ai = ai[:,0,:]
        av = av.sum(axis=1)
        # ui :    [batch_size , u_field_size]
        # uv :    [batch_size , u_field_size]
        # u_emb : [batch_size , u_field_size , embedding_size]
        a_emb = self.a_embeddings(ai)
        u_emb = self.u_embeddings(ui)
        #a_emb: [batch_size , field_size , embeding_size]
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.a_field_size, 1))
        # u_emb: [batch_size , field_size , embeding_size]
        u_emb = torch.multiply(u_emb, uv.reshape(-1, self.u_field_size, 1))

        if self.batch_norm:
            a_emb = self.batch_norm_layer(a_emb)

        # [batch_size, embeding_size, field_size]
        a_emb = a_emb.permute([0, 2, 1])

        # [batch_size, conv_size, field_size//5]
        a_emb = self.conv1(a_emb)
        a_emb = self.relu(a_emb)
        # [batch_size, field_size//5, conv_size]
        a_emb = a_emb.permute([0, 2, 1])

        # u_inter:[batch_size, context_size]
        u_inter = self.relu(self.fc1(u_emb.reshape(-1, self.u_field_size * self.embedding_size)))
        # a: [batch_size, a_field_size // 5, context_size]
        a = torch.unsqueeze(u_inter, dim=1).repeat(1, self.a_field_size // 5, 1)
        # [batch_size, a_field_size // 5, conv_size + context_size]
        ua_inter = torch.cat((a_emb, a), dim=2)
        # [batch_size, a_field_size // 5, attn_size]
        attn_logit = self.relu(self.fc2(ua_inter.reshape(-1, self.conv_size + self.context_size)))

        # attn_w:[ batch_size , (day//5) * (a_field_size//5)]
        attn_w = self.fc3(attn_logit).reshape(-1, self.a_field_size // 5)
        # 归一化
        attn_w = self.softmax(attn_w)
        # a_weight_emb  [batch_size , field_size//5 , conv_size]
        if self.attn_enable:
            a_weight_emb = torch.multiply(torch.unsqueeze(attn_w, dim=2), a_emb)
        else:
            a_weight_emb = a_emb

        # [batch_size , conv_size]
        deep_input = torch.sum(a_weight_emb, dim=1)
        # [batch_size,conv_size + context_size]
        deep_input = torch.cat((deep_input, u_inter), dim=1)

        y_deep = deep_input
        y_deep=self.dropout(y_deep)
        for i in range(len(self.deep_layers)):
            y_deep = self.relu(self.dnn[i](y_deep))
            y_deep = self.dropout(y_deep)
        # out [batch_size , 1]
        out = torch.sigmoid(self.logic_layer(y_deep))
        # y_true_bool (Active or not):[batch_size , 1]
        esp=1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)
        filtered_y,filtered_pred_y=setMask(y,out)
        if y is not None:
            if lossFun=='BCE':
                loss = self.criterion(out, y_true_bool)
            else:
                loss = self.criterion_1(out, y)
            return loss,out,filtered_y,filtered_pred_y
        else:
            return out,filtered_y,filtered_pred_y
    def batch_norm_layer(self, x):
        bn = self.batch_norm(x)
        return bn