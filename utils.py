import glob
import os
import random
import uuid
import time
import cv2
import vlc

player = None
last_saved_at = time.time()


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
    global last_saved_at
    t = time.time()
    if t > (last_saved_at + 2):
        last_saved_at = t
        unique_filename = prefix + str(uuid.uuid4()) + ".jpg"
        cv2.imwrite(unique_filename, img)
