import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "path_with_images",
    help="CFG file with settings of the ViZDoom scenario.")
parser.add_argument(
    "path_with_scores",
    help="Path to save imagens of the game")
parser.add_argument(
    "path_to_save_blend",
    help="Path to save imagens of the game")


if __name__ == '__main__':
    args = parser.parse_args()
    images = os.listdir(args.path_with_images)
    scores = os.listdir(args.path_with_scores)

    alpha = 1.0
    beta = 0.5
    gamma = 0.5

    for image, score in zip(images,scores):
        data = np.load(f"{args.path_with_scores}/{score}") * 255
        img = cv2.imread(f"{args.path_with_images}/{image}")
        matriz_heatmap = cv2.applyColorMap(data.astype('uint8'), cv2.COLORMAP_JET)
        blended_image = cv2.addWeighted(img, alpha, matriz_heatmap, beta, gamma)
        print(f"{args.path_to_save_blend}/{image}")
        cv2.imwrite(f"{args.path_to_save_blend}/{image}", blended_image)

