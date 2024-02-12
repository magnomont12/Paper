from __future__ import absolute_import, print_function, division, unicode_literals
from PIL import Image
import curses


import argparse

parser = argparse.ArgumentParser(
        description="Load a trained agent and run it in a given scenario. Be careful: if the scenario given is not the one in which the agent was trained, it will run as usual, but agent's performance will be poor.")

parser.add_argument(
    "model_folder",
    help="Path to the folder containing the model to be loaded.")
parser.add_argument(
    "config_file",
    help="CFG file with settings of the ViZDoom scenario.")
parser.add_argument(
    "-n", "--num-games",
    type=int,
    metavar="N",
    default=1,
    help="Number of games to play. [default=5]")
parser.add_argument(
    "-show", "--show-model",
    metavar="FILENAME",
    help="Print the model architecture on screen and save a PNG image.",
    default="")
parser.add_argument(
    "-d", "--disable-window",
    action="store_true",
    help="Disable rendering of the game, effectively showing only the score obtained.")
parser.add_argument(
    "-log", "--log-file",
    metavar="FILENAME",
    help="File path to save the results.",
    default="temp_log_file.txt")

args = parser.parse_args()
    

import vizdoom
import itertools as it
import numpy as np
import skimage.color, skimage.transform
import os

from PIL import Image as PilImage
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision.specific.gradcam import GradCAM
from omnixai.explainers.vision import IntegratedGradientImage
from omnixai.explainers.vision import ShapImage
from omnixai.explainers.vision import LimeImage

from cv2 import resize
from tqdm import trange
from time import time, sleep

import tensorflow as tf
from tensorflow import keras

# use the same maps to ensure a fair comparison
import test_maps

import cv2

# limit gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# training settings
resolution = (48, 64)
frame_repeat = 4


# Converts and down-samples the input image
def preprocess(img):
    img = resize(img, (resolution[1], resolution[0]))
    img = img.astype(np.float32)
    img = img / 255.0
    return img

def get_q_values(model, state):
    return model.predict(state)

def get_best_action(model, state):
    s = state.reshape([1, resolution[0], resolution[1], 1])
    return tf.argmax(get_q_values(model, s)[0])

def encontrar_arquivos_cfg(pasta_raiz):
    arquivos_cfg = []

    for pasta_atual, subpastas, arquivos in os.walk(pasta_raiz):
        for arquivo in arquivos:
            if arquivo.endswith(".cfg"):
                caminho_completo = os.path.join(pasta_atual, arquivo)
                arquivos_cfg.append(caminho_completo)

    return arquivos_cfg
            
def initialize_vizdoom(config_file):
    print("[1.] Initializing ViZDoom...")
    game = vizdoom.DoomGame()
    print(config_file)
    game.load_config(config_file)
    game.set_window_visible(False)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("[.1] ... ViZDoom initialized.")
    return game
import cv2
import curses

def render_image(state):
    # Inicializa a tela usando curses
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.keypad(True)

    # Renderiza a imagem redimensionada
    cv2.imshow('Rendered Image', state)
    cv2.waitKey(0)  # Aguarda até que uma tecla seja pressionada

    # Atualiza a tela do curses
    stdscr.refresh()

    # Continue o loop após a tecla Q ser pressionada
    return

def render_many_images(states):
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.keypad(True)
    states.extend([np.zeros_like(states[0]), np.zeros_like(states[0])])

    # Crie um array numpy empilhando as imagens
    state = np.vstack([np.hstack(states[i:i+6]) for i in range(0, 36, 6)])

    cv2.imshow('Rendered Image', state)
    cv2.waitKey(0)  # Aguarda até que uma tecla seja pressionada

    # Atualiza a tela do curses
    stdscr.refresh()

    # Continue o loop após a tecla Q ser pressionada
    return

def calculate_grad_cam(model, image):
    explainer = GradCAM(
        model=model,
        target_layer=model.layers[2],
        preprocess_function=None
    )
    img = PilImage.fromarray(np.uint8((image*255)))
    img = Resize((48, 64)).transform(Image(img))
    explanations = explainer.explain(img)
    scores = explanations.get_explanations()[0]['scores']
    return scores

def calculate_ig(model, image):
    explainer = IntegratedGradientImage(
        model=model,
        target_layer=model.layers[2],
        preprocess_function=None,
    )
    img = PilImage.fromarray(np.uint8((image*255)))
    img = Resize((48, 64)).transform(Image(img))
    explanations = explainer.explain(img)
    scores = explanations.get_explanations()[0]['scores']
    scores = np.clip(scores,0,scores.max())
    return scores

def calculate_shap(model, image):
    explainer = ShapImage(
        model=model,
        preprocess_function=None,
    )
    img = PilImage.fromarray(np.uint8((image*255)))
    img = Resize((48, 64)).transform(Image(img))
    explanations = explainer.explain(img)
    scores = explanations.get_explanations()[0]['scores']
    scores = np.clip(scores,0,scores.max())
    return scores

def blend_images(score, image):
    alpha = 1.0
    beta = 0.5
    gamma = 0.5
    matriz_heatmap = cv2.applyColorMap((score*255).astype('uint8'), cv2.COLORMAP_JET)
    
    s = (image * 255)
    img = np.zeros((48,64,3))
    img[:,:,0] = s
    img[:,:,1] = s
    img[:,:,2] = s
    img = img.astype('uint8')
    blended_image = cv2.addWeighted(img, alpha, matriz_heatmap, beta, gamma)
    blended_image = cv2.resize(blended_image, (blended_image.shape[1]*2, blended_image.shape[0]*2))
    return blended_image

def verify_all_games_end():
    for game in games:
        if not game.is_episode_finished():
            return False
    return True


if __name__ == "__main__":
    num_of_scenarios = 34

    # load model
    if (os.path.isdir(args.model_folder)):
        print("Loading model from " + args.model_folder + ".")
        model = keras.models.load_model(args.model_folder)
    else:
        print("No folder was found in " + args.model_folder + ".")
        quit()

    if args.show_model:
        model.summary()        
        keras.utils.plot_model(model, args.show_model + ".png", show_shapes=True)
    
    games = []
    config_files = encontrar_arquivos_cfg(args.config_file)
    for config_file in config_files:
        games.append(initialize_vizdoom(config_file))

    num_actions = games[0].get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    for i in trange(args.num_games, leave=True):
        for game in games:
            game.set_seed(test_maps.TEST_MAPS[i])
            game.new_episode()
        
        while not verify_all_games_end():
            images = []
            for game in games:
                if not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(model, state)
                    score = calculate_grad_cam(model, state)
                    blend_image = blend_images(score, state)
                    images.append(blend_image)
                    game.set_action(actions[best_action_index])
                else:
                    images.append(np.zeros((resolution[0]*2, resolution[1]*2,3), dtype=np.uint8))
            render_many_images(images)

            for _ in range(frame_repeat):
                images = []
                for game in games:
                    game.advance_action()
                    if game.get_state() != None:
                        state = preprocess(game.get_state().screen_buffer)
                        blend_image = blend_images(score, state)
                        images.append(blend_image)
                    else:
                        images.append(np.ones((resolution[0]*2, resolution[1]*2,3), dtype=np.uint8))      
                render_many_images(images)
    
    for game in games:
        game.close()