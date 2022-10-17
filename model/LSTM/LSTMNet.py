import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()

        #f_t
        self.U_f = nn.parameter(torch.tensor(input_size, hidden_size))
        self.V_f = nn.parameter(torch.tensor(input_size, hidden_size))
        self.b_f = nn.parameter(torch.tensor(hidden_size))

        #o_t
        self.U_o = nn.parameter(torch.tensor(input_size, hidden_size))
        self.V_o = nn.parameter(torch.tensor(input_size, hidden_size))
        self.b_o = nn.parameter(torch.tensor(hidden_size))

        #i_t
        self.U_i = nn.parameter(torch.tensor(input_size, hidden_size))
        self.V_i = nn.parameter(torch.tensor(input_size, hidden_size))
        self.b_i = nn.parameter(torch.tensor(hidden_size))

        #g_t
        self.U_g = nn.parameter(torch.tensor(input_size, hidden_size))
        self.V_g = nn.parameter(torch.tensor(input_size, hidden_size))
        self.b_g = nn.parameter(torch.tensor(hidden_size))



    def forward(self, input, h_last, c_last):


        f_t = nn.functional.sigmoid(torch.mm(self.U_f, input) +torch.mm(self.V_f , input) + self.b_f)
        g_t = nn.functional.tanh(torch.mm(self.U_g, input) + torch.mm(self.V_g, input) + self.b_g)
        i_t = nn.functional.sigmoid(torch.mm(self.U_i, input) + torch.mm(self.V_i, input) + self.b_i)
        o_t = nn.functional.sigmoid(torch.mm(self.U_o, input) + torch.mm(self.V_o,  input) + self.b_o)

        c_now = torch.mm(f_t, c_last) + torch.mm(i_t, g_t)
        h_now = torch.mm(o_t, nn.functional.tanh(c_now))

        output = None
        return output, (h_now, c_now)
