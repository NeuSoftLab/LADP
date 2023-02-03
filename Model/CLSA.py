import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Model.Tool import setMask

class CLSA(torch.nn.Module):
    def __init__(self, model_param):
        super(CLSA, self).__init__()
        self.device = model_param['device']
        # BCE
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        # MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)

        # parameters
        self.clas_conv2_kernel = model_param['clas_conv2_kernel']
        self.clsa_conv2_output_size = model_param['clsa_conv2_output_size']
        self.clas_pool_kernel = model_param['clas_pool_kernel']
        self.batch_size = model_param['batch_size']

        self.day = model_param['day']
        self.a_field_size = model_param['a_field_size']

        if (((self.day - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 < 0):
            self.clas_conv2_kernel = 1

        # conv2d
        self.conv2 = torch.nn.Conv2d(1, self.clsa_conv2_output_size, self.clas_conv2_kernel, stride=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.clas_pool_kernel, stride=2)

        # linear
        h2 = ((self.day - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 + 1
        w2 = ((self.a_field_size - self.clas_conv2_kernel + 1) - self.clas_pool_kernel) // 2 + 1
        input_size = self.clsa_conv2_output_size * h2 * w2

        # lstm
        self.lstm_input_size = model_param['lstm_input_size']
        self.hidden_size = model_param['hidden_size']

        # relu function
        self.relu = torch.nn.ReLU()
        # LSTM
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, bidirectional=True)

        # attention
        self.attention_input_size = model_param['attention_input_size']
        self.fc2 = torch.nn.Linear(self.attention_input_size, self.attention_input_size)
        self.tanh = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(self.attention_input_size, 1, bias=False)
        self.sig = torch.nn.Sigmoid()

        # fc
        self.fc1 = torch.nn.Linear(input_size, self.day * self.lstm_input_size)
        self.fc4 = torch.nn.Linear(self.attention_input_size, 1)

    def forward(self, ui, uv, ai, av, y=None, lossFun='BCE'):
        # av: batch_size * 1 * day * a_field_size
        av = torch.unsqueeze(av, dim=1)
        # av_conv: batch_size * clsa_conv2_output_size * h1 * w1
        # h1: self.day - self.clas_conv2_kernel + 1
        # w1: self.a_field_size-self.clas_conv2_kernel + 1
        av_conv = self.conv2(av)
        av_conv = self.relu(av_conv)
        # av_conv:batch_size * clsa_conv2_output_size * h2 * w2
        # h2: (h1 - self.clas_pool_kernel) // 2 + 1
        # w2: (w1 - self.clas_pool_kernel) // 2 + 1
        av_maxpool = self.maxpool(av_conv)
        # av_fc: av_conv:batch_size * [clsa_conv2_output_size * h2 * w2]
        av_fc = av_maxpool.reshape(self.batch_size, -1)

        # linear
        # av_fc: batch_size * [day * lstm_input_size]
        av_fc = self.fc1(av_fc)
        av_fc = self.relu(av_fc)
        # av_fc: batch_size * day * lstm_input_size
        av_fc = av_fc.reshape(self.batch_size, self.day, -1)
        # av_fc: day , batch_size ,action_emb
        av_fc = av_fc.permute((1, 0, 2))
        # output: seq_len, batch_size, num_directions * hidden_size
        # num_directions: 2
        # hidden_size: 14
        output, (h_n, c_n) = self.lstm(av_fc)
        # output:　batch_size，seq_len,num_directions * hidden_size
        output = output.permute((1, 0, 2))
        # attention
        # m: batch_size，seq_len,num_directions * hidden_size
        m = self.tanh(self.fc2(output))
        # s: batch_size，seq_len,1
        s = self.sig(self.fc3(m))
        # r: batch_size，seq_len,num_directions * hidden_size
        r = output * s
        # out: batch_size，num_directions * hidden_size
        y_deep = torch.sum(r, dim=1)
        # out: batch_size，1
        out = torch.sigmoid(self.fc4(y_deep))

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
