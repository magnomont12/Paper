import argparse
import os
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "dir",
    help="Dir of the images.")

parser.add_argument(
    "method",
    help="Dir of the images.")

args = parser.parse_args()

path_scores = os.listdir(args.dir)

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

def mean_grad_cam(mean):
    return cv2.applyColorMap(mean.astype('uint8'), cv2.COLORMAP_JET)

def mean_ig(mean):
    mean = normalize_data_positive(mean)
    return generate_heatmap(mean)

if __name__ == '__main__':
    scores = []
    for file in path_scores:
        array_npy = np.load(os.path.join(args.dir,file))
        scores.append(array_npy)

    mean = np.mean(scores, axis=0)
    np.save(f"mean_images_normal/{args.method}/image_{args.dir.replace('/','_')}.npy", mean)
    mean = mean*255

    if args.method == "grad-cam":
        matriz_heatmap = mean_grad_cam(mean)
    elif args.method == "ig" or args.method == "shap":
        matriz_heatmap = mean_ig(mean)
    cv2.imwrite(f"mean_images_normal/{args.method}/image_{args.dir.replace('/','_')}.png", matriz_heatmap)