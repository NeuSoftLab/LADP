import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Model.Tool import setMask

class LSCNN(torch.nn.Module):
    def __init__(self, model_param):
        super(LSCNN, self).__init__()
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

        # embedding layer
        self.embedding_size = model_param['embedding_size']
        self.a_embeddings = torch.nn.Embedding(self.a_feat_size, self.embedding_size)
        self.u_embeddings = torch.nn.Embedding(self.u_feat_size, self.embedding_size)

        # hidden_size, input_size
        self.hidden_size = model_param['hidden_size']
        self.lstm_2_input_size = model_param['lstm_2_input_size']

        # max pool
        self.lscnn_conv2_kernel = model_param['lscnn_conv2_kernel']  # 3 * 3
        self.lscnn_conv2_outputsize = model_param['lscnn_conv2_outputsize']  # 20
        self.lscnn_pool_kernel = model_param['lscnn_pool_kernel']  # 2

        #  conv2d
        self.conv2 = torch.nn.Conv2d(1, self.lscnn_conv2_outputsize, self.lscnn_conv2_kernel, padding='same',
                                     stride=1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.lscnn_pool_kernel, stride=2)

        # linear
        h2 = (self.day - self.lscnn_conv2_kernel + 3 - self.lscnn_pool_kernel) // 2 + 1
        w2 = (self.a_field_size - self.lscnn_conv2_kernel + 3 - self.lscnn_pool_kernel) // 2 + 1
        liner_1_input_size = self.lscnn_conv2_outputsize * h2 * w2
        # fc
        self.fc1 = torch.nn.Linear(liner_1_input_size, self.day * self.lstm_2_input_size)

        #  LSTM
        self.lstm_1 = torch.nn.LSTM(input_size=self.embedding_size * self.a_field_size, hidden_size=self.hidden_size)
        self.lstm_2 = torch.nn.LSTM(input_size=self.lstm_2_input_size, hidden_size=self.hidden_size)

        # logic layer
        self.logic_layer = torch.nn.Linear(self.hidden_size, 1, bias=False)

        # relu
        self.relu = torch.nn.ReLU()
        # sigmoid
        self.sig = torch.nn.Sigmoid()
        # dropout
        self.dropout = torch.nn.Dropout(model_param['dropout_p'])

        self.wd = 0.5

    def forward(self, ui, uv, ai, av, y=None, lossFun='BCE'):
        # batch_size: 32
        # ai : batch_size * day * u_field_size
        # av : batch_size * day * u_field_size
        # ui : batch_size * a_field_size
        # uv : batch_size * a_field_size

        # LSTM
        # a_emb: batch_size * day * [a_field_size * embedding_size]
        a_emb = self.a_embeddings(ai)
        a_emb = torch.multiply(a_emb, av.reshape(-1, self.day, self.a_field_size, 1))
        a_emb = a_emb.reshape(self.batch_size, self.day, -1)

        # a_emb: day * batch_size * [a_field_size * embedding_size]
        a_emb = a_emb.permute((1, 0, 2))
        output_1, (h_n, c_n) = self.lstm_1(a_emb)
        # output_1: batch_size * day * hidden_size
        output_1 = output_1.permute((1, 0, 2))
        # y_deep_1: batch_size * hidden_size
        y_deep_1 = torch.sum(output_1, dim=1)
        # out_1: batch_size * 1
        out_1 = self.sig(self.logic_layer(y_deep_1))

        # CNN + LSTM
        # av: batch_size * 1 * day * a_field_size
        av = torch.unsqueeze(av, dim=1)
        # av_conv: batch_size * lscnn_conv2_output_size * h1 * w1
        # h1: self.day - self.lscnn_conv2_kernel + 3
        # w1: self.a_field_size - self.lscnn_conv2_kernel + 3
        av_conv = self.conv2(av)
        av_conv = self.relu(av_conv)
        # av_maxpool: batch_size * lscnn_conv2_output_size * h2 * w2
        # h2: (h1 - self.lscnn_pool_kernel) // 2 + 1
        # w2: (w1 - self.lscnn_pool_kernel) // 2 + 1
        av_maxpool = self.maxpool(av_conv)
        # av_fc: batch_size * [lscnn_conv2_outputsize * h2 * w2]
        av_fc = av_maxpool.reshape(self.batch_size, -1)

        # linear
        # av_fc: batch_size * [day * lstm_2_input_size]
        av_fc = self.fc1(av_fc)
        av_fc = self.dropout(av_fc)
        av_fc = self.relu(av_fc)

        # av_fc: batch_size * day * lstm_2_input_size
        av_fc = av_fc.reshape(self.batch_size, self.day, -1)
        # av_fc: day * batch_size * lstm_2_input_size
        av_fc = av_fc.permute((1, 0, 2))
        # output_2: day * batch_size * hidden_size
        output_2, (h_n, c_n) = self.lstm_2(av_fc)
        # output_2: batch_size * day * hidden_size
        output_2 = output_2.permute((1, 0, 2))
        # output_2: batch_size * hidden_size
        y_deep_2 = torch.sum(output_2, dim=1)
        # out_2: batch_size * 1
        out_2 = self.sig(self.logic_layer(y_deep_2))

        # add sum
        out = ((out_1 + out_2) * self.wd)

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
        return
