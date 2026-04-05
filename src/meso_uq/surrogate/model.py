import pickle

import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dims, output_dims, hl_dims):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dims, hl_dims[0], bias=True))
        self.layers.append(torch.nn.Tanh())
        for hl_dim0, hl_dim1 in zip(hl_dims[:-1], hl_dims[1:]):
            self.layers.append(torch.nn.Linear(hl_dim0, hl_dim1, bias=True))
            self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(hl_dims[-1], output_dims, bias=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def save_model_states(model, *, xshift, xscale, yshift, yscale, path):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "xshift": xshift, "xscale": xscale, "yshift": yshift, "yscale": yscale}, f, pickle.HIGHEST_PROTOCOL)


def load_model_states(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["xshift"], data["xscale"], data["yshift"], data["yscale"]
