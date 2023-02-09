import torch
from torch.autograd import Variable
from Model.Tool import setMask

class RNN(torch.nn.Module):
    def __init__(self, model_param):
        super(RNN, self).__init__()
        # device
        self.device = model_param['device']

        # criterion
        # BCE
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        # MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)

        # action user : one_hot_num
        self.a_feat_size = model_param['a_feat_size']
        self.u_feat_size = model_param['u_feat_size']
        # action user : feature_num
        self.a_field_size = model_param['a_field_size']
        self.u_field_size = model_param['u_field_size']

        # batch_size = 32, seq_length = day, input_size = action_type_num, hidden_size = 64
        self.batch_size = model_param['batch_size']
        self.seq_length = model_param['day']
        self.input_size = model_param['input_size']
        self.hidden_size = model_param['hidden_size']

        self.rnn_cell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        self.logic_layer = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, ui, uv, ai, av, y=None, lossFun='BCE'):
        self.batch_size = av.shape[0]

        # av: seq_length, batch_size, input_size
        av = av.permute([1, 0, 2])
        # RNN
        hidden = self.init_hidden()
        hidden = hidden.to(self.device)
        for index, input in enumerate(av):
            hidden = self.rnn_cell(input, hidden)

        out = torch.sigmoid(self.logic_layer(hidden))

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

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
