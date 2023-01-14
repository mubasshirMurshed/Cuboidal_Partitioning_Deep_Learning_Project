# %% Imports
import matplotlib.pyplot as plt
from datasets import MNISTGraphDataset
from torchvision.datasets import MNIST

# %% Dataset
ds1 = MNIST(r"data\mnistPytorch", train=False)
ds2 = MNISTGraphDataset(root="data/mnistNew", filename="mnistTest128", length=10)

# %% Plot graph using feature values of x-y pos and colour
idx = 1

fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]
i = 0
# for i, (ax1, ax2) in enumerate(axes):
img = ds1[idx + i][0]
data = ds2[idx + i].x
values, _ = ds2[idx + i].edge_index.t().sort()
edges = values.unique(dim=0)

ax1.imshow(img, cmap="gray")

ax2.invert_yaxis()
ax2.set_facecolor("black")
ax2.set_xlim([0, 27])
ax2.xaxis.set_ticks(range(0, 28, 3))
ax2.set_ylim([27, 0])
ax2.yaxis.set_ticks(range(27, -1, -3))

# Add nodes
for i in range(len(data)):
    # ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.25, facecolor=str(data[i][2].item()), edgecolor="white", linewidth=0.2))
    ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.25, facecolor="red", edgecolor="white", linewidth=0.2))

# Add edges
for i in range(len(edges)):
    # Get source cuboid information
    src_cuboid = data[edges[i][0]]
    x_src = src_cuboid[0]*28
    y_src = src_cuboid[1]*28
    col_src = src_cuboid[2]

    # Get destination cuboid information
    dst_cuboid = data[edges[i][1]]
    x_dst = dst_cuboid[0]*28
    y_dst = dst_cuboid[1]*28
    col_dst = dst_cuboid[2]

    # Get average colour
    colour = (col_src + col_dst)/2

    # Plot line from (x_src, y_src) to (x_dst, y_dst)
    # ax2.plot([x_src, x_dst], [y_src, y_dst], color=str(colour.item()), linestyle='-', linewidth=0.5)
    ax2.plot([x_src, x_dst], [y_src, y_dst], color="red", linestyle='-', linewidth=0.5)

ax2.set_box_aspect(1)
plt.show()

# # %% Do NetworkX graph
# graph = ds2.dataset[idx]
# G = to_networkx(graph, to_undirected=True)
# positions = {key: (x, -y) for (key, (x, y)) in enumerate(graph.pos.cpu().detach().numpy())}
# nx.draw_networkx(G, pos=positions)

# %%
