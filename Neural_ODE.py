import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import time

class NeuralODE(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, y0, t, solver):
        solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        solution[0] = y0

        j = 1
        for t0, t1 in zip(t[:-1], t[1:]):
            dy = solver(self.func, t0, t1 - t0, y0)
            y1 = y0 + dy
            solution[j] = y1
            j += 1
            y0 = y1
        return solution


# generate spiral dataset
data_size = 500
true_y0 = torch.tensor([[2., 0.]])
t = torch.linspace(0., 25., data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def visualize(true_y, pred_y=None):
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_title('Trajectory Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], color='green', label='sampled points', s=3)
    #ax.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], color='black', label='true trajectory')
    if pred_y is not None:
        ax.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'red', label='learned trajectory')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    plt.legend()
    plt.grid(True)
    plt.show()


visualize(true_y)


batch_time = 10
batch_size = 16


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (batch_size, 1, emb)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (time, batch_size, 1, emb)
    return batch_y0, batch_t, batch_y


# define dynamic function
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 50),
                             nn.Tanh(),
                             nn.Linear(50,2))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        output = self.net(y)
        return output


# Train
niters = 150

func = ODEFunc()
optimizer = optim.RMSprop(func.parameters(), lr=0.001)

start = time.time()
losses = []
iters = []
for iter in range(niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint_adjoint(func=func, y0=batch_y0, t=batch_t, rtol=1e-7, atol=1e-9, method='euler')

    loss = torch.mean((pred_y - batch_y)**2)
    iters.append(iter)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    if iter % 50 == 0:
        with torch.no_grad():
            pred_y = odeint_adjoint(func, true_y0, t, rtol=1e-7, atol=1e-9, method='euler')
            loss = torch.mean((pred_y - true_y) ** 2)
            print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))
            visualize(true_y, pred_y)


end = time.time()

