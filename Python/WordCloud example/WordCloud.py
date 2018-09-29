from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Convert all the required text into a single string here 
# and store them in word_string

# you can specify fonts, stopwords, background color and other options

file = open('speech.txt', 'r', encoding='utf-8')
word_string = file.read()

wordcloud = WordCloud(font_path='C:/MyriadPro-Bold.otf',
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(word_string)


plt.imshow(wordcloud)
plt.axis('off')
plt.show()








from collections import Counter
from konlpy.tag import Hannanum
from konlpy.tag import Twitter
import pytagcloud

file = open('speech.txt', 'r', encoding='utf-8')
data = file.read().split('\n');
data
word = []
for t in data:
    T = t.split(' ')
    for t2 in T:
        word.append(t2)

word
# nouns = [t.decode('utf-8') for t in word]
count = Counter(word)
count

hannanum = Hannanum()
nouns = hannanum.nouns(data)

nlp = Twitter()
nouns = nlp.nouns(data)

count = Counter(nouns)

tag2 = count.most_common(40)
taglist = pytagcloud.make_tags(tag2, maxsize=80)

pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600), fontname='Nobile', rectangular=False)


#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from collections import Counter
import urllib
import random
import webbrowser

from konlpy.tag import Hannanum
from lxml import html
import pytagcloud # requires Korean font support
import sys

if sys.version_info[0] >= 3:
    urlopen = urllib.request.urlopen
else:
    urlopen = urllib.urlopen


r = lambda: random.randint(0,255)
color = lambda: (r(), r(), r())

def get_bill_text(billnum):
    url = 'http://pokr.kr/bill/%s/text' % billnum
    response = urlopen(url).read().decode('utf-8')
    page = html.fromstring(response)
    text = page.xpath(".//div[@id='bill-sections']/pre/text()")[0]
    return text

def get_tags(text, ntags=50, multiplier=10):
    h = Hannanum()
    nouns = h.nouns(text)
    count = Counter(nouns)
    return [{ 'color': color(), 'tag': n, 'size': c*multiplier }\
                for n, c in count.most_common(ntags)]

def draw_cloud(tags, filename, fontname='Noto Sans CJK', size=(800, 600)):
    pytagcloud.create_tag_image(tags, filename, fontname=fontname, size=size)
    webbrowser.open(filename)


bill_num = '1904882'
text = get_bill_text(bill_num)
tags = get_tags(text)
print(tags)
draw_cloud(tags, 'wordcloud.png')