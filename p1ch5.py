import torch
import numpy as np

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = t_u*(1e-2)
def model(w1, w2, t_u, b):
    return (w2 * t_u**2 + w1 * t_u + b)

def loss_fn(t_p, t_c):
    squared_diffs = (t_p-t_c)**2
    return squared_diffs.mean()

w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(())
params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = torch.optim.SGD([params], lr = learning_rate)

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        
        t_p = model(params[0], params[1], t_u, params[2])
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if epoch%500==0:
            print('Epoch %d, Loss%f' % (epoch, float(loss)),'\n', params)
    return params

params = training_loop(n_epochs = 5000, optimizer=optimizer, params = params, t_u= t_un, t_c= t_c)


