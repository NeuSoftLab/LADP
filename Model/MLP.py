import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Model.Tool import setMask

class MLP(torch.nn.Module):
    def __init__(self, model_param):
        super(MLP, self).__init__()
        # device
        self.device = model_param['device']

        # criterion
        # BCE
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        # MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)

        # batch_size
        self.batch_size = model_param['batch_size']
        # use day num
        self.day = model_param['day']

        # action user : one_hot_num
        self.a_feat_size = model_param['a_feat_size']
        self.u_feat_size = model_param['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_param['a_field_size']
        self.u_field_size = model_param['u_field_size']

        # embedding size
        self.embedding_size = model_param['embedding_size']
        # DNN layer
        self.deep_layers = model_param['deep_layers']
        # batch_norm
        self.batch_norm = model_param['batch_norm']
        self.batch_norm_decay = model_param['batch_norm_decay']
        # setting embedding layers
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)
        # setting relu function
        self.relu = torch.nn.ReLU()
        self.dnn = torch.nn.ModuleList()
        # DNN network
        input_size = self.day * self.a_field_size * self.embedding_size + self.u_field_size * self.embedding_size

        self.dnn.append(torch.nn.Linear(input_size, self.deep_layers[0]))
        for i in range(1, len(self.deep_layers)):
            self.dnn.append(torch.nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
        self.logic_layer = torch.nn.Linear(self.deep_layers[-1], 1)

        self.batch_norm = torch.nn.BatchNorm2d(self.embedding_size, eps=1e-05, momentum=1 - self.batch_norm_decay,
                                               affine=True,
                                               track_running_stats=True)
        # Softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # dropout
        self.dropout = torch.nn.Dropout(model_param['dropout_p'])

    def forward(self, ui, uv, ai, av, y=None, lossFun='BCE'):
        # batch_size: 32
        # embedding_size: 32

        # ai : batch_size * day * u_field_size
        # av : batch_size * day * u_field_size
        # ui : batch_size * a_field_size
        # uv : batch_size * a_field_size
        # a_emb : batch_size * day * u_field_size * embedding_size
        # u_emb : batch_size * a_field_size * embedding_size

        a_emb = self.a_embeddings(ai)
        u_emb = self.u_embeddings(ui)
        # a_emb: batch_size * day * field_size * embedding_size
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))
        # u_emb: batch_size * field_size * embedding_size
        u_emb = torch.multiply(u_emb, uv.reshape(-1, self.u_field_size, 1))

        a_emb = a_emb.reshape(self.batch_size, -1)
        u_emb = u_emb.reshape(self.batch_size, -1)
        # deep_input:batch_size*(day * a_field_size * embedding_size + u_field_size * embedding_size)
        deep_input = torch.cat((a_emb, u_emb), 1)
        # print(deep_input.shape)
        y_deep = deep_input
        y_deep = self.dropout(y_deep)
        for i in range(len(self.deep_layers)):
            y_deep = self.relu(self.dnn[i](y_deep))
            y_deep = self.dropout(y_deep)
        # out: batch_size * 1
        out = torch.sigmoid(self.logic_layer(y_deep))

        # y_true_bool (Active or not):batch_size * 1
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)
        filtered_y, filtered_pred_y = setMask(y, out)
        if y is not None:
            if lossFun == 'BCE':
                loss = self.criterion(out, y_true_bool)
            else:
                loss = self.criterion_1(out, y)
            return loss, out,filtered_y,filtered_pred_y
        else:
            return out,filtered_y,filtered_pred_y

    def batch_norm_layer(self, x):
        bn = self.batch_norm(x)
        return bn
