# %% Imports
import matplotlib.pyplot as plt
from datasets import MNISTGraphDataset_V3

# %% Dataset
ds = MNISTGraphDataset_V3(root="data/logo", filename="logo")

# %% Plot graph using feature values of x-y pos and colour
idx = 1
fig, ax1 = plt.subplots(nrows=1, ncols=1)
data = ds[idx].x
values, _ = ds[idx].edge_index.t().sort()
edges = values.unique(dim=0)

ax1.invert_yaxis()
ax1.set_facecolor("maroon")
ax1.set_xlim([0, 63])
ax1.xaxis.set_ticks(range(0, 64, 8))
ax1.set_ylim([43, 0])
ax1.yaxis.set_ticks(range(43, -1, -4))

# Add nodes
for i in range(len(data)):
    ax1.add_patch(plt.Circle((data[i][0]*28, data[i][1]*28), 0.4, facecolor=str(data[i][2].item()), edgecolor="white", linewidth=0.4))
    
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
    ax1.plot([x_src, x_dst], [y_src, y_dst], color=str(colour.item()), linestyle='-', linewidth=1)

ax1.set_box_aspect(44/64)
plt.show()

# %%
