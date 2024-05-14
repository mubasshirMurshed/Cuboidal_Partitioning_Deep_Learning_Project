# %% Imports
from datasets.graph_dataset import MNISTGraphDataset_Auto_Parallel, MNISTGraphDataset_Auto, MNISTGraphDataset_CSV
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

# %% Dataset
def func():
    ds1 = MNIST(r"data\mnistPytorch", train=False)
    ds2 = MNISTGraphDataset_CSV(root="data/", mode="CP", num_cuboids=64, split="Test", x_centre=True, y_centre=True, colour=True, num_pixels=True, angle=True)

    idx = 1
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax1 = axes[0]
    ax2 = axes[1]
    img = ds1[idx][0]
    data = ds2[idx].x
    # data[:, 2] -= data[:, 2].min()
    # data[:, 2] /= data[:, 2].max()

    # data[:, 0] -= data[:, 0].min()
    # data[:, 0] /= data[:, 0].max()
    # data[:, 0] *= 28

    # data[:, 1] -= data[:, 1].min()
    # data[:, 1] /= data[:, 1].max()
    # data[:, 1] *= 28

    values, _ = ds2[idx].edge_index.t().sort()
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
        # ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.5, facecolor=str(data[i][2].item()), edgecolor="white", linewidth=0.4))
        ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.25, facecolor=str(data[i][2].item()), edgecolor="white", linewidth=0.2))
        # ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.25, facecolor="red", edgecolor="white", linewidth=0.2))
        # ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.05*data[i][3]*28*data[i][4]*28, facecolor=str(data[i][2].item()), edgecolor="white", linewidth=0.5))
        # ax2.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.25, facecolor="red", edgecolor="white", linewidth=0.5))

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
        # ax2.plot([x_src, x_dst], [y_src, y_dst], color=str(colour.item()), linestyle='-', linewidth=1)
        ax2.plot([x_src, x_dst], [y_src, y_dst], color=str(colour.item()), linestyle='-', linewidth=0.5)
        # ax2.plot([x_src, x_dst], [y_src, y_dst], color="red", linestyle='-', linewidth=0.5)

    ax2.set_box_aspect(1)
    plt.show()

    # # %% Do NetworkX graph
    # graph = ds2.dataset[idx]
    # G = to_networkx(graph, to_undirected=True)
    # positions = {key: (x, -y) for (key, (x, y)) in enumerate(graph.pos.cpu().detach().numpy())}
    # nx.draw_networkx(G, pos=positions)

# %%
if __name__ == "__main__":
    func()
# %%
