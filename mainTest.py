import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')
a = str(np.random.randint(0,60))
filename='F:\BrainTumorCLassification\pred\pred' + a + '.jpg'
image=cv2.imread(filename)

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict_step(input_img)
if (result[0][1]==1.0):

    print("pred"+a+":Tumor is present")
else:
    print("pred"+a+":Tumor is not present")




