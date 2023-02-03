import torch
from Model.Tool import setMask

class LR(torch.nn.Module):
    def __init__(self, model_param):
        super(LR, self).__init__()
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

        # setting embedding layers
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)

        # input_size
        input_size = self.day * self.a_field_size * self.embedding_size + self.u_field_size * self.embedding_size
        # linear layer
        self.fc = torch.nn.Linear(input_size, 1)

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
        # deep_input:batch_size*(day * a_field_size * embedidng_size + u_field_size * embedding_size)
        deep_input = torch.cat((a_emb, u_emb), 1)
        y_deep = deep_input
        # out batch_size * 1
        out = self.fc(y_deep)

        # y_true_bool (Active or not):batch_size * 1"
        esp = 1e-5
        y_true_bool = y.clone()
        y_true_bool[y >= esp] = 1.0
        y_true_bool[y < esp] = 0.0
        y_true_bool = y_true_bool.to(self.device)
        filtered_y, filtered_pred_y = setMask(y, out)
        if y is not None:
            if lossFun == 'BCE':
                out = torch.sigmoid(out)
                loss = self.criterion(out, y_true_bool)
            else:
                loss = self.criterion_1(out, y)
            return loss, out,filtered_y,filtered_pred_y
        else:
            return out,filtered_y,filtered_pred_y
