import logging
import time
import cv2
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import handlers

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

CHECK_PERIOD = 2

data_dir = '/home/tim/datasets/pets/train'


def init_freetype2():
    global ft2
    ft2 = cv2.freetype.createFreeType2()
    ft2.loadFontData('/usr/share/fonts/cantarell/Cantarell-Light.otf', 0)
    #ft2.loadFontData('ComicSansMSRegular.ttf', 0)


def predict_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(im, mode="RGB")
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def put_text(img, text, fontheight, text_pos):
    margin = 5
    baseline_factor = 9/41  # for Cantarelle Light
    thickness = -1
    ts = ft2.getTextSize(text, fontheight, thickness)
    text_pos2 = tuple([sum(x)+margin for x in zip(ts[0], text_pos)])
    text_pos2 = (text_pos2[0], text_pos2[1] + int(baseline_factor * fontheight))
    cv2.rectangle(img, (text_pos[0] - margin, text_pos[1] - margin), text_pos2, (50, 50, 50), cv2.FILLED)
    ft2.putText(img, text, (text_pos[0], text_pos[1] + ts[0][1]), fontheight, (200, 200, 200), thickness, cv2.LINE_AA
                , True)


test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                      ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint = torch.load('./model_best-pets-35e-87a.pth.tar')
checkpoint = torch.load('./model_best-pets-50e-88a.pth.tar')
model = checkpoint['model']
data = datasets.ImageFolder(data_dir, transform=test_transforms)
classes = data.classes
# TODO implement NV12 color pipelineing to RGB via gstreamer
#cv2.cvtColor(cv2.COLOR_BGR2RGB, code=cv2.CV_8U)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

init_freetype2()

last_checked = 0
last_collected = 0
cv2.imshow("preview", frame)

last_frame_time = 1
frame_num = 0

while rval:
    start_time = time.time()
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:     # exit on ESC
        break

    if (time.time() - last_checked) > CHECK_PERIOD:
        frame_num += 1
        last_checked = time.time()
        index = predict_image(frame)
        text1 = "FPS: {:05.2f}".format(1. / last_frame_time)
        text2 = classes[index]
        handler = getattr(handlers, "handle_" + text2, lambda x: None)
        handler(frame, frame_num)
        put_text(frame, text1, fontheight=41, text_pos=(10, 7))
        put_text(frame, text2, fontheight=41, text_pos=(10, 53))

        cv2.imshow("preview", frame)
    last_frame_time = time.time() - start_time

cv2.destroyWindow("preview")
