from dataset import default_box_generator
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
sys.path.append('./')


def plot_boxes(boxes, name='bounding_box'):
  fig, ax = plt.subplots(figsize=(20, 15))
  cmap = cm.get_cmap('hsv', 5)  # Colormap with 4 colors

  # Loop over boxes with a step of 4 (considering 4 boxes per color)
  for i in range(0, len(boxes), 4):
    # Get the current colormap
    colors = cmap(np.linspace(0, 1, 5))

    for j, box in enumerate(boxes[i:i+4]):
      x_min, y_min, x_max, y_max = box[4:]
      rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=1, edgecolor=colors[j], facecolor='none')
      ax.add_patch(rect)
  plt.axis('scaled')
  # plt.show()
  plt.savefig(f'./anchorboxes/{name}.png')


os.makedirs('./anchorboxes', exist_ok=True)

lsizes = [0.2, 0.4, 0.6, 0.8]
ssizes = [0.1, 0.3, 0.5, 0.7]
num_boxes = [10, 5, 3, 1]

for i, num in enumerate(num_boxes):
    boxes = default_box_generator([num], [lsizes[i]], [ssizes[i]])
    plot_boxes(boxes, name=f"bb_{num}")

boxes = default_box_generator(num_boxes, lsizes, ssizes)
plot_boxes(boxes, name=f'bb_all')
