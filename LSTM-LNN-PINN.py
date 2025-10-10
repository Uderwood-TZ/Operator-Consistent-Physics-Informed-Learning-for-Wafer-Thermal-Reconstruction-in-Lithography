import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


epochs = 50000
N_interior = 3000
N_boundary = 500
radius = 0.15
k = 159.0
h_ = 50.0
T_inf = 800.0
Q = 2000.0


d_ref = radius
t_ref = 1.0
Q_ref = Q
k_ref = k
h_ref = h_


k_nd = 1.0
h_nd = h_ref * d_ref / k_ref
Q_nd = Q_ref * d_ref**2 / (k_ref * t_ref)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)



def sample_interior(n):
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x.requires_grad_(True), y.requires_grad_(True)



def sample_boundary(n):
    theta = 2 * torch.pi * torch.rand(n, 1)
    x = torch.cos(theta)
    y = torch.sin(theta)
    nx = x
    ny = y
    return x.requires_grad_(True), y.requires_grad_(True), nx, ny


class GatedLiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        i = self.sigmoid(self.W_i(combined))
        f = self.sigmoid(self.W_f(combined))
        o = self.sigmoid(self.W_o(combined))
        c_tilde = self.activation(self.W_c(combined))
        c = f * c_prev + i * c_tilde
        h = o * self.activation(c)
        return h, c


class GatedLiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.neuron = GatedLiquidNeuron(input_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h, c = self.neuron(x, h, c)
        return h


class GatedLNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GatedLiquidLayer(2, 64)
        self.layer2 = GatedLiquidLayer(64, 64)
        self.layer3 = GatedLiquidLayer(64, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, xy):
        x = self.layer1(xy)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.output_layer(x)
        return out




def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x, 1), x, order=order - 1)


def loss_pde(model):
    x, y = sample_interior(N_interior)
    xy = torch.cat([x, y], dim=1)
    dT = model(xy)

    d2T_dx2 = gradients(dT, x, order=2)
    d2T_dy2 = gradients(dT, y, order=2)

    res = k_nd * (d2T_dx2 + d2T_dy2) + Q_nd
    return torch.mean(res**2)


def loss_bc(model):
    x, y, nx, ny = sample_boundary(N_boundary)
    xy = torch.cat([x, y], dim=1)
    dT = model(xy)  # ΔT=0 即 T=T_inf
    dT_dx = gradients(dT, x, order=1)
    dT_dy = gradients(dT, y, order=1)

    dT_dn = dT_dx * nx + dT_dy * ny
    flux = -k_nd * dT_dn
    bc = h_nd * dT
    return torch.mean((flux - bc)**2)


model = GatedLNN()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
history_pde = []
history_bc = []
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    lpde = loss_pde(model)
    lbc = loss_bc(model)
    loss_total = lpde + lbc
    loss_total.backward()
    optimizer.step()

    history_pde.append(lpde.item())
    history_bc.append(lbc.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss PDE: {lpde.item():.16f}, Loss BC: {lbc.item():.16f}, Total Loss: {loss_total.item():.16f}")


model.eval()
x_plot = torch.linspace(-1, 1, 400)
y_plot = torch.linspace(-1, 1, 400)
xm, ym = torch.meshgrid(x_plot, y_plot, indexing='ij')
x_flat = xm.reshape(-1, 1)
y_flat = ym.reshape(-1, 1)
xy_plot = torch.cat([x_flat, y_flat], dim=1)

with torch.no_grad():
    dT_pred_nd = model(xy_plot).numpy()

T_pred = dT_pred_nd * t_ref + T_inf
x_real = x_flat * d_ref
y_real = y_flat * d_ref


T_full = T_pred.reshape(xm.shape)


circle_mask = (x_real**2 + y_real**2 <= d_ref**2).numpy().reshape(xm.shape)
T_full[~circle_mask] = np.nan


end_time = time.time()
elapsed_time = end_time - start_time


with open('CC.txt', 'w') as f_time:
    f_time.write(f"CC: {elapsed_time:.10f} s\n")

plt.figure(figsize=(6, 5))
cf = plt.contourf((xm * d_ref).numpy(), (ym * d_ref).numpy(), T_full, levels=200, cmap='rainbow')
cbar = plt.colorbar(cf)
cbar.ax.tick_params(labelsize=10)
cbar.ax.yaxis.set_offset_position('left')
cbar.set_label('Temperature (K)', fontsize=12)
cbar.formatter = FormatStrFormatter("%.4f")
cbar.update_ticks()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"Temperature at {learning_rate:.0e}.png")
plt.show()





plt.figure(figsize=(10, 4))
plt.plot(history_pde, label='Loss PDE')
plt.plot(history_bc, label='Loss BC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"The loss function at {learning_rate:.0e}.png")
plt.show()


plt.figure(figsize=(10, 4))
plt.semilogy(history_pde, label='Loss PDE')
plt.semilogy(history_bc, label='Loss BC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Logarithmic loss function at {learning_rate:.0e}.png")
plt.show()

mask_inside = (x_real**2 + y_real**2 <= d_ref**2).squeeze()

x_valid = x_real[mask_inside]
y_valid = y_real[mask_inside]
T_valid = torch.from_numpy(T_pred)[mask_inside]

data_out = torch.cat([x_valid, y_valid, T_valid], dim=1).numpy()
np.savetxt('Pre.txt', data_out, fmt='%.6f', comments='')



import subprocess
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
error_analysis_script = os.path.join(current_dir, 'DA.py')

if os.path.exists(error_analysis_script):
    print("DA.py ...")
    subprocess.run(['python', error_analysis_script], check=True)
else:
    print("NO DA.py")













