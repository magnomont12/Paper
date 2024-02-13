import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap

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
parser.add_argument(
    "method",
    help="Path to save imagens of the game")

alpha = 1.0
beta = 0.5
gamma = 0.5

def blend_grad_cam(images, scores):
    for image, score in zip(images,scores):
        data = np.load(f"{args.path_with_scores}/{score}") * 255
        img = cv2.imread(f"{args.path_with_images}/{image}")
        matriz_heatmap = cv2.applyColorMap(data.astype('uint8'), cv2.COLORMAP_JET)
        blended_image = cv2.addWeighted(img, alpha, matriz_heatmap, beta, gamma)
        print(f"{args.path_to_save_blend}/{image}")
        cv2.imwrite(f"{args.path_to_save_blend}/{image}", blended_image)


def normalize_data_positive(data):
    arr_positive = np.maximum(data, 0)
    arr_positive = (arr_positive-np.min(arr_positive))/(np.max(arr_positive)-np.min(arr_positive))
    return arr_positive

def generate_heatmap(data):
    green = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0)
    ]
    green_map = LinearSegmentedColormap.from_list('verde', green)

    matriz_heatmap = green_map(data)
    matriz_heatmap = (matriz_heatmap*255).astype(np.uint8)
    return cv2.cvtColor(matriz_heatmap, cv2.COLOR_RGBA2RGB)

def blend_ig(images, scores):
    for image, score in zip(images,scores):
        img = cv2.imread(f"{args.path_with_images}/{image}")
        data = np.load(f"{args.path_with_scores}/{score}") * 255
        data = normalize_data_positive(data)
        matriz_heatmap = generate_heatmap(data)
        blended_image = cv2.addWeighted(img, alpha, matriz_heatmap, beta, gamma)
        cv2.imwrite(f"{args.path_to_save_blend}/{image}", blended_image)
        


if __name__ == '__main__':
    args = parser.parse_args()
    images = os.listdir(args.path_with_images)
    scores = os.listdir(args.path_with_scores)

    if args.method == "grad-cam":
        blend_grad_cam(images, scores)
    elif args.method == "ig" or args.method == "shap":
        blend_ig(images, scores)


