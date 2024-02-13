import argparse
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "dir",
    help="Dir of the images.")

args = parser.parse_args()

path_scores = os.listdir(args.dir)

scores = []
for file in path_scores:
    array_npy = np.load(os.path.join(args.dir,file))
    scores.append(array_npy)

mean = np.mean(scores, axis=0) * 255

matriz_heatmap = cv2.applyColorMap(mean.astype('uint8'), cv2.COLORMAP_JET)
cv2.imwrite(f"mean_images/image_{args.dir.replace('/','_')}.png", matriz_heatmap)