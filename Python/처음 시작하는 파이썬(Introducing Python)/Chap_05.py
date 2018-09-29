'''import sys
import Chap_4

print('Program arguments: ', sys.argv)

description = Chap_4.weather()
print(description);

'''
from sources import daily, weekly

print("Daily forecast: ", daily.forecast())
print("Weekly forecast: ")
for number, outlook in enumerate(weekly.forecast(), 1):
    print(number, outlook)

help(dict.get)

from collections import Counter
help(Counter)
from collections import OrderedDict, defaultdict
