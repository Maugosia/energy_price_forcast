from torch import nn, Tensor, transpose, eye, zeros, ones, rand, matmul, flatten, distributions
from globals import DEVICE
import math


class LipschitzNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, beta_a, beta_w, gamma_a, gamma_w, dt=0.001):
        super(LipschitzNet, self).__init__()
        # matrix construction params
        self.beta_a = beta_a
        self.beta_w = beta_w
        self.gamma_a = gamma_a
        self.gamma_w = gamma_w
        self.hidden_dim = hidden_dim

        # trainable weight matrices
        # m_a = Tensor(hidden_dim, hidden_dim)
        # self.M_A = nn.Parameter(m_a)
        # nn.init.kaiming_uniform_(self.M_A, a=math.sqrt(5))
        #
        # m_w = Tensor(hidden_dim, hidden_dim)
        # self.M_W = nn.Parameter(m_w)
        # nn.init.kaiming_uniform_(self.M_W, a=math.sqrt(5))
        self.M_W = nn.Parameter(gaussian_init_(hidden_dim, std=1))
        self.M_A = nn.Parameter(gaussian_init_(hidden_dim, std=1))

        # b = Tensor([1])
        # self.b = nn.Parameter(b)
        # nn.init.zeros_(self.b)
#
        # u = Tensor(hidden_dim, input_dim)
        # self.U = nn.Parameter(u)
        # nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

        # d = Tensor(output_dim, hidden_dim)
        # self.D = nn.Parameter(d)
        # nn.init.kaiming_uniform_(self.D, a=math.sqrt(5))
        self.D = nn.Linear(hidden_dim, output_dim)
        self.E = nn.Linear(input_dim, hidden_dim)
        # constructed matrices
        self.W = zeros((hidden_dim, hidden_dim), device=DEVICE)
        self.A = zeros((hidden_dim, hidden_dim), device=DEVICE)

        # extra
        self.I = eye(hidden_dim)
        self.hidden_state = ones([hidden_dim, 1])
        self.dt = dt
        self.tan_h = nn.Tanh()

    def forward(self, x):
        t = x.shape[1]
        self.hidden_state = zeros(x.shape[0], self.hidden_dim).to(DEVICE)

        for i in range(t):
            # print("2", x[:, i, :].shape)
            z = self.E(transpose(x[:, i, :], 1, 0))

            if i == 0:
                self.A = (1 - self.beta_a) * (self.M_A + transpose(self.M_A, 1, 0)) + \
                         self.beta_a * (self.M_A - transpose(self.M_A, 1, 0)) - \
                         self.gamma_a * self.I

                self.W = (1 - self.beta_w) * (self.M_W + transpose(self.M_W, 1, 0)) + \
                         self.beta_w * (self.M_W - transpose(self.M_W, 1, 0)) - \
                         self.gamma_w * self.I

            hidden_state_delta = self.dt * matmul(self.hidden_state, self.A) + \
                                 self.dt * self.tan_h(matmul(self.hidden_state, self.W) +
                                                      z)

            self.hidden_state = self.hidden_state + hidden_state_delta

        return self.D(self.hidden_state)

    def init_hidden(self):
        self.hidden_state = ones([self.hidden_dim, 1])


def gaussian_init_(n_units, std=1):
    sampler = distributions.Normal(Tensor([0]),
                                   Tensor([std / n_units]))
    a_init = sampler.sample((n_units, n_units))[..., 0]
    return a_init
