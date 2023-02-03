import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Model.Tool import setMask

class DPCNN(torch.nn.Module):
    def __init__(self, model_param):
        super(DPCNN, self).__init__()
        self.device = model_param['device']
        # criterion
        # BCE
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        # MSE
        self.criterion_1 = torch.nn.MSELoss()
        self.criterion_1.to(self.device)

        # parameters
        self.dpcnn_conv2_kernel_1 = model_param['dpcnn_conv2_kernel_1']             # 3 * 3
        self.dpcnn_conv2_output_size_1 = model_param['dpcnn_conv2_output_size_1']     # 32
        self.dpcnn_conv2_kernel_2 = model_param['dpcnn_conv2_kernel_2']             # 3 * 3
        self.dpcnn_conv2_output_size_2 = model_param['dpcnn_conv2_output_size_2']     # 64
        self.batch_size = model_param['batch_size']

        self.day = model_param['day']
        self.a_field_size = model_param['a_field_size']

        # conv2d
        self.conv2_1 = torch.nn.Conv2d(1, self.dpcnn_conv2_output_size_1, self.dpcnn_conv2_kernel_1, padding='same',
                                       stride=1, bias=True)
        self.conv2_2 = torch.nn.Conv2d(self.dpcnn_conv2_output_size_1, self.dpcnn_conv2_output_size_2,
                                       self.dpcnn_conv2_kernel_2, padding='same', stride=1, bias=True)
        # linear
        h2 = (self.day - self.dpcnn_conv2_kernel_1 + 3) - self.dpcnn_conv2_kernel_2 + 3
        w2 = (self.a_field_size - self.dpcnn_conv2_kernel_1 + 3) - self.dpcnn_conv2_kernel_2 + 3
        input_size = self.dpcnn_conv2_output_size_2 * h2 * w2
        self.liner_1_output_size = model_param['liner_1_output_size']
        # fc
        self.fc1 = torch.nn.Linear(input_size, self.liner_1_output_size)
        # relu function
        self.relu = torch.nn.ReLU()
        # logic layer
        self.logic_layer = torch.nn.Linear(self.liner_1_output_size, 1, bias=False)
        # sigmoid
        self.sig = torch.nn.Sigmoid()
        # dropout
        self.dropout = torch.nn.Dropout(model_param['dropout_p'])

    def forward(self, ui, uv, ai, av, y=None, lossFun='BCE'):
        # av: batch_size * 1 * day * a_field_size
        av = torch.unsqueeze(av, dim=1)
        # av_conv_1: batch_size * dpcnn_conv2_output_size_1 * h1 * w1
        # h1: self.day - self.dpcnn_conv2_kernel_1 + 3
        # w1: self.a_field_size - self.dpcnn_conv2_kernel_1 + 3
        av_conv_1 = self.conv2_1(av)
        av_conv_1 = self.relu(av_conv_1)
        # av_conv_1:batch_size * dpcnn_conv2_output_size_2 * h2 * w2
        # h2: h1 - self.dpcnn_conv2_kernel_2 + 3
        # w2: w1 - self.dpcnn_conv2_kernel_2 + 3
        av_conv_2 = self.conv2_2(av_conv_1)
        av_conv_2 = self.relu(av_conv_2)
        # av_fc: av_conv:batch_size * [dpcnn_conv2_output_size_2 * h2 * w2]
        av_fc = av_conv_2.reshape(self.batch_size, -1)
        # linear
        # av_fc: batch_size * liner_1_output_size
        av_fc = self.dropout(av_fc)
        av_fc = self.fc1(av_fc)
        av_fc = self.relu(av_fc)
        # out: batch_size * 1
        out = self.sig(self.logic_layer(av_fc))

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
