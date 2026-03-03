import torch
import numpy as np

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(w1, w2, t_u, b):
    return (w2 * t_u**2 + w1 * t_u + b)

def loss_fn(t_p, t_c):
    squared_diffs = (t_p-t_c)**2
    return squared_diffs.mean()

w1 = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(())

t_p = model(w1, w2, t_u, b)
loss = loss_fn(t_p, t_c)

#print(loss,'\n', t_p)

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs

def dmodel_dw1(w1, w2, t_u, b):
    return (t_u**2)

def dmodel_dw2(w1, w2, t_u, b):
    return(t_u)

def dmodel_db(w1, w2, t_u, b):
    return 1.0

grad = dmodel_dw1(w1, w2, t_u, b)
print(sum(grad))

def grad_fn(w1, w2, t_u, t_c, t_p, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dmodel_dw1(w1, w2, t_u, b)
    dloss_dw2 = dmodel_dw2(w1, w2, t_u, b)
    dloss_db = dmodel_db(w1, w2, t_u, b)
    return torch.stack ([sum(dloss_dw1), sum(dloss_dw2), dloss_db])

grad = grad_fn(w1, w2, t_u, t_c, t_p, b)
print(sum(grad))