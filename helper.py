# coding: utf8

import cv2
import numpy as np
import sys


def dumyshape(img):

    # color invert - cv2 goofyness
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_NEAREST) # 128,128,3

    img = img / 255.

    img = np.transpose(img, (2,0,1)) # (3, 128, 128)

    # img = np.expand_dims(img, 0) # (1, 3, 128, 128)

    return img

def reward_calculation(reward_dict, info, reset=False):
    if reset:
        reward_dict = {
            "current_score": 0,
            "current_x": 0,
            "current_rings": 0,
            "reward_flow": 0,
            "lives": None
        }
        return 0

    if reward_dict["current_score"] is None:
        reward_dict["current_score"] = info["score"]
    if reward_dict["current_score"] != info["score"]:
        delta_score = info["score"] - reward_dict["current_score"]
        reward_dict["current_score"] = info["score"]
    else:
        delta_score = 0

    if reward_dict["current_x"] == 0:
        reward_dict["current_x"] = info["x"]
    if reward_dict["current_x"] != info["x"]:
        delta_x = info["x"] - reward_dict["current_x"]
        reward_dict["current_x"] = info["x"]
        delta_x /= 6.
    else:
        delta_x = 0

    if reward_dict["current_rings"] is None:
        reward_dict["current_rings"] = info["rings"]
    if reward_dict["current_rings"] != info["rings"]:
        delta_rings = info["rings"] - reward_dict["current_rings"]
        delta_rings *= 200
        reward_dict["current_rings"] = info["rings"]
    else:
        delta_rings = 0

    if reward_dict["lives"] is None:
        reward_dict["lives"] = info["lives"]
    if reward_dict["lives"] != info["lives"]:
        delta_lives = info["lives"] - reward_dict["lives"]
        reward_dict["lives"] = info["lives"]
        delta_lives *= 800
    else:
        delta_lives = 0

    reward_dict["reward_flow"] = reward_dict["reward_flow"] + delta_score + delta_x + delta_rings + delta_lives


def dumyshape_shrink_expand(img, scale=None):

    # color invert - cv2 goofyness
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_NEAREST) # 128,128,3

    # shrink expand
    if scale and scale>1:
        size = int(128/scale)

        img = cv2.resize( img, (size,size) ) # shrink

        img = cv2.resize( img, ( 128,128 ) ) # expand


    img = img / 255.

    img = np.transpose(img, (2,0,1)) # (3, 128, 128)

    # img = np.expand_dims(img, 0) # (1, 3, 128, 128)

    return img

def dumyshape_gray_edges(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128)) # 128,128


    # edge detection
    img = cv2.Canny(img, 100, 200) # 128,128

    img = img / 255.
    img = np.expand_dims(img, axis=0) # (1, 128, 128)
    # img = np.expand_dims(img, axis=0)  # (1, 1, 128, 128)

    return img

def dumyshape_gray(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128)) # 128,128


    # edge detection
    # img = cv2.Canny(img, 100, 200) # 128,128

    # img = np.interp(img, (0,255), (0,1))
    img = img / 255.
    img = np.expand_dims(img, axis=0) # (1, 128, 128)
    # img = np.expand_dims(img, axis=0)  # (1, 1, 128, 128)

    return img

def reverse_dumyshape(deconv_img):
    deconv_img = deconv_img.squeeze(0) # [3, 128, 128]
    deconv_img = deconv_img.cpu().data.numpy() # (3, 128, 128)
    deconv_img = np.transpose(deconv_img, (1,2,0)) # (128, 128, 3)
    # deconv_img = deconv_img * 255

    return deconv_img

