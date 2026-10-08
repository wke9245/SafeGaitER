import torch


def identity_contribution(features, target, strips=8, create_graph=False):
    # Gradients are local to the frame feature tensor during inference.
    (gradient,) = torch.autograd.grad(
        target.sum(), features, create_graph=create_graph, retain_graph=True
    )
    b, t, channels, height, width = features.shape
    activation_gradient = (features * gradient).reshape(
        b, t, channels, strips, height // strips, width
    )
    value = activation_gradient.mean(dim=(2, 4, 5)).relu()
    return value / (value.sum(dim=(1, 2), keepdim=True) + 1e-8)
