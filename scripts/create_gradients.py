import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision.specific.gradcam import GradCAM

import tensorflow as tf
from tensorflow import keras

import os

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "model_folder",
    help="Path to the folder containing the model to be loaded.")
parser.add_argument(
    "path_with_images",
    help="CFG file with settings of the ViZDoom scenario.")
parser.add_argument(
    "path_to_save_images",
    help="Path to save imagens of the game")


if __name__ == '__main__':
    args = parser.parse_args()
    images = os.listdir(args.path_with_images)
    model = keras.models.load_model(args.model_folder)

    dictionary_actions = [f"action {action}" for action in range(model.output[0].shape[0])]

    explainer = GradCAM(
        model=model,
        target_layer=model.layers[2],
        preprocess_function=None
    )

    for image in images:
        img = Resize((48, 64)).transform(Image(PilImage.open(f"{args.path_with_images}/{image}")))
        explanations = explainer.explain(img)
        scores = explanations.get_explanations()[0]['scores']
        np.save(f"{args.path_to_save_images}/{image.replace('.png','.npy')}", scores)