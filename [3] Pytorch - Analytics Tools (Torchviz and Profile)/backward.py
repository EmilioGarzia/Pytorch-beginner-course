# Pytorch: how to compute gradient
#
# #author Emilio Garzia, 2024

from torchviz import make_dot
import torch

# Input tensor features
x = torch.tensor([2.0,1.0,3.0], requires_grad=True, dtype=torch.float32)
y = torch.tensor([1.,3.,2.], requires_grad=True, dtype=torch.float32)

# Output tensor
z = (x**3)+(y**2)

# Reduce the output tensor as a scalar value
z = z.sum()

# Compute the gradient of x and y
z.backward()

# Print the gradient of x and y
print(x.grad)
print(y.grad)

# Render the graph of model
"""
graph = make_dot(z, params={"x": x, "y": y}, show_attrs=True, show_saved=True)
graph.render("graph", format="png")
"""