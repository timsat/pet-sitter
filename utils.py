import glob
import os
import random
import uuid

import cv2
import vlc


player = None

def warn():
    global player
    if player is not None and player.is_playing():
        return
    warns = glob.glob("./warn*.ogg")
    i = random.randrange(len(warns))
    absname = os.path.abspath(warns[i])
    player = vlc.MediaPlayer('file://' + absname)
    player.play()


def save_img_with_prefix(img, prefix):
    unique_filename = prefix + str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(unique_filename, img)