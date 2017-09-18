# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:32:04 2016

@author: fangx
"""

from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

bimg = imread("D:\\PDM\\2.5\\贾宝玉.png")

wordcloud = WordCloud(
    background_color="white", 
    mask=bimg, font_path='D:\\PDM\\2.5\\simhei.ttf'
)

wordcloud = wordcloud.fit_words(words['计数'])

bimgColors = ImageColorGenerator(bimg)

plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()



bimg = imread("D:\\PDM\\2.5\\贾宝玉2.png")

wordcloud = WordCloud(
    background_color="white", 
    mask=bimg, font_path='D:\\PDM\\2.5\\simhei.ttf'
)

wordcloud = wordcloud.fit_words(words['计数'])

plt.figure(
    num=None, 
    figsize=(8, 6), dpi=80, 
    facecolor='w', edgecolor='k'
)

bimgColors = ImageColorGenerator(bimg)

plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()
