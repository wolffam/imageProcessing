from pyemd import emd
import numpy as np
from PIL import Image

im = Image.open("8-bitelemy.png")
pix = im.load()

h1 = [1.0/64] * 64
h2 = [0.0] * 64
hist1 = np.array(h1)

w,h = im.size
print(w)
print(h)
print((im.getpixel((1500,100))))
'''
for ind,item in enumerate(list(im.getdata())):
    if ind < 1920:
        print(item)
'''
print(im.histogram())
print(im.getextrema())
im.show()