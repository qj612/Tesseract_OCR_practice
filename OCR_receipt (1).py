
# coding: utf-8

# In[20]:


import cv2
import pytesseract
from PIL import Image
import cv2
import argparse
import re
import os
import tempfile
import logging
import numpy as np
import nltk
from autocorrect import spell
from matplotlib import pyplot as plt
import PyPDF2
import io
from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io


# In[38]:


file_path=r"I:\James Jiang\Optical Character Recognition\receipt.png"


# In[22]:


try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract as tes


# ## 1. Data Visualization through adjusting different filters and see which one perform better

# In[6]:


img = cv2.imread(file_path,0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])


# In[7]:


plt.imshow(thresh1,'gray')


# In[8]:


img = cv2.imread(file_path)

kernel = np.ones((1, 1), np.uint8)
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


# In[9]:


img = cv2.imread(file_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,45,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# In[10]:


plt.imshow(thresh,'gray')


# ## 2. create image noise-cleaning tools and dpi adjustment functions as below
# <p> smooth image, remove noise and change dpi based on data visualization above in order to achieve better OCR performance.</p>

# In[24]:


IMAGE_SIZE = 1600
BINARY_THREHOLD = 80

size = None


def get_size_of_scaled_image(im):
    global size
    if size is None:
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
    return size


def process_image_for_ocr(file_path):
    logging.info('Processing image for text Extraction')
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new

def set_image_dpi(file_path):
    im = Image.open(file_path).convert('RGB')
    # size = (1800, 1800)
    size = get_size_of_scaled_image(im)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))  # best for OCR
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    logging.info('Removing noise and smoothening image')
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


# In[28]:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


# In[12]:


def auto_correction(string):
    token=nltk.word_tokenize(string)
    string1=" ".join(str(spell(x)) for x in token)
    return string1


# ## 3. Example of a restaurant receipt
# <p> Use an example of a restaurant receipt after adjusting parameters above.</p>

# In[39]:


results = tes.image_to_string(process_image_for_ocr(file_path), lang='eng', boxes=None, config='-psm 6')
print(results)

