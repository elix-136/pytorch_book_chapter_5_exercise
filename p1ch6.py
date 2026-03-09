import torch
import torch.nn as nn
import torch.optim as optim

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)
t_un = t_u *0.1
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr = 1e-2)

def training_loop(n_epohs, optimizer,model, loss_fn, t_u, t_c):
    for epoch in range (1, n_epohs-1):
        t_p = model(t_u)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

training_loop(n_epohs=1000, 
              optimizer=optimizer, 
              model=linear_model, 
              loss_fn=nn.MSELoss(), 
              t_u=t_un, 
              t_c= t_c)

print(linear_model.weight, linear_model.bias)

