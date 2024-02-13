from __future__ import absolute_import, print_function, division, unicode_literals
from PIL import Image


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
    "path_to_save_images",
    help="Path to save imagens of the game",
    default="")
parser.add_argument(
    "-n", "--num-games",
    type=int,
    metavar="N",
    default=5,
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

from cv2 import resize
from tqdm import trange
from time import time, sleep

import tensorflow as tf
from tensorflow import keras

# use the same maps to ensure a fair comparison
import test_maps


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
            

def initialize_vizdoom(args):
    print("[1.] Initializing ViZDoom...")
    game = vizdoom.DoomGame()
    print(args.config_file)
    game.load_config(args.config_file)
    game.set_window_visible(not args.disable_window)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("[.1] ... ViZDoom initialized.")
    return game

def create_image(state,args,count,best_action_index,episode):
    s = state * 255
    imagem = Image.fromarray(s.astype('uint8'))
    imagem.save(f'{args.path_to_save_images}/image_{episode+1}_{count}_{best_action_index}.png')


if __name__ == "__main__":

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
    
    
    # create vizdoom game
    game = initialize_vizdoom(args)

    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]


    # open file to save resultsMap: line
    if args.log_file:
        log_file = open(args.log_file, "w", buffering=1)
        print("Map,{}".format(args.config_file), file=log_file)
        print("Resolution,{}".format(resolution), file=log_file)
        print("Frame Repeat,{}".format(frame_repeat), file=log_file)
        print("Number of Games,{}".format(args.num_games), file=log_file)
        print("Elapsed time,Score min,Score max,Score mean,Score std", file=log_file)
    

    scores = []
    episode_time = []
    start_time = time()
    for i in trange(args.num_games, leave=True):
        print(test_maps.TEST_MAPS[i])
        game.set_seed(test_maps.TEST_MAPS[i])
        game.new_episode()
        
        ep_start_time = time()
        count = 1
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(model, state)
            #create_image(state, args, count, best_action_index,i)
            count+=1
            game.set_action(actions[best_action_index])
            
            for _ in range(frame_repeat):
                game.advance_action()
                
        ep_time = time() - ep_start_time
        episode_time.append(ep_time)
        
        score = game.get_total_reward()
        scores.append(score)
        
        if args.log_file:
            print("{:.2f},{}".format(ep_time, score), file=log_file)

        if (not args.disable_window):
            print("Score ep. " + str(i) + ": " + "{:2.0f}".format(score))
            sleep(1.0)
    
    total_elapsed_time = time() - start_time
    scores = np.array(scores)
    episode_time = np.array(episode_time)
    
    print("Results: mean: %.1f±%.1f," % (scores.mean(), scores.std()), "min: %.1f" % scores.min(), "max: %.1f" % scores.max())
    
    if args.log_file:
        print("{:.2f},{:.2f},{},{},{},{:.2f}".format(
            total_elapsed_time, episode_time.sum(), scores.min(), scores.max(), scores.mean(), scores.std()), file=log_file)
                
    game.close()
