
import collections

Card = collections.namedtuple('Card', ['rank','suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                      for rank in self.ranks]
    def __len__(self):
        return len(self._cards)
    def __getitem__(self, position):
        return self._cards[position]

beer_card = Card('7','diamonds')
beer_card

deck = FrenchDeck()
len(deck)
deck[0]
deck[-1]
from random import choice
choice(deck)
deck[:3]
for card in reversed(deck):
    print(card)
Card('Q', 'hearts') in deck
Card('Q', 'imgoos') in deck

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)
def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]
for card in sorted(deck, key=spades_high):
    print(card)




from math import hypot

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'Vector(%r, %r)' % self.x, self.y
    def __abs__(self):
        return hypot(self.x, self.y)
    def __bool__(self):
        return bool(abs(self))
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def __bool__(self):
        return bool(self.x or self.y)

colors = ['black','white']
sizes=['S','M','L']
for tshirt in ('%s %s' % (c, s) for c in colors for s in sizes):
    print(tshirt)

b, a = a, b
divmod(20, 8)
t = (20, 8)
divmod(*t)
import os
_, filename = os.path.split('/home/luciano/.ssh/idrsa.pub')
filename
a, b, *rest = range(5)
a, b, rest
a, *body, c, d = range(5)
a, body, c, d



from collections import namedtuple
City = namedtuple('City', 'name country population coordinates')
tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
tokyo
tokyo.population
tokyo.coordinates
tokyo[1]

City._fields
LatLong = namedtuple('LatLong', 'lat long')
delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.6, 77.2))
delhi = City._make(delhi_data)
delhi._asdict()
for key, value in delhi._asdict().items():
    print(key + ':', value)

my_list = [[]]*3
my_list
board = [['_'] * 3 for i in range(3)]
board[1][2]='X'
board
weird_board = [['_'] * 3] * 3
weird_board[1][2]='X'
weird_board

import dis
dis.dis('s[a] += b')
str.lower('ABc')

import bisect, sys
haystack=[1,4,5,6,8,12,15,20,21,23,23,26,29,30]
needles = [0,1,2,5,8,10,22,23,29,30,31]
row_fmt = '{0:2d} @ {1:2d}    {2}{0:<2d}'
def demo(bisect_fn):
    for needle in reversed(needles):
        position = bisect_fn(haystack, needle)
        offset = position * '  |'
        print(row_fmt.format(needle, position, offset))
if __name__ == '__main__':
    if sys.argv[-1] == 'left':
        bisect_fn = bisect.bisect_left
    else:
        bisect_fn = bisect.bisect
    print('DEMO: ', bisect_fn.__name__)
    print('haystack ->' , ' '.join('%2d' % n for n in haystack))
    demo(bisect_fn)

def grade(score, breakpoints=[60,70,80,90], grades='FDCBA'):
    i = bisect.bisect(breakpoints, score)
    return grades[i]
[grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]

import bisect, random
SIZE = 7
random.seed(1729)
my_list = []
for i in range(SIZE):
    new_item = random.randrange(SIZE*2)
    bisect.insort(my_list, new_item)
    print('%2d ->' % new_item, my_list)

from array import array
from random import random
floats = array('d', (random() for i in range(10**7)))
floats[-1]
fp = open('floats.bin','wb')
floats.tofile(fp)
fp.close()
floats2 = array('d')
fp = open('floats.bin', 'rb')
floats2.fromfile(fp, 10**7)
fp.close()
floats2[-1]
floats2==floats
# a = array('i', (1,5,4,2,3,6))
# a = array(a.typecode, sorted(a))
a

# pickle 모듈
# pickle.dump()

import array
numbers = array.array('h', [-2, -1, 0, 1, 2])
memv = memoryview(numbers)
len(memv)
memv[0]
memv_oct = memv.cast('B')
len(memv_oct)
memv_oct.tolist()
memv_oct[5]=4
numbers

import numpy
floats = numpy.loadtxt('floats-10M-lines.txt')
from time import perf_counter as pc
t0 = pc()

pc()-t0

from collections import deque
dq = deque(range(10), maxlen=10)
dq
dq.rotate(3)
dq
dq.rotate(-4)
dq
dq.appendleft(-1)
dq
dq.extend([11,22,33])
dq
dq.extendleft([10,20,30,40])
dq
# Queue, LifoQueue, PriorityQueue
# multiprocessing.JoinableQueue
l = [28,14,'28',5,'9','1',0,6,'23',19]
sorted(l, key=int)
sorted(l, key=str)

my_dict = {}
import collections
isinstance(my_dict, collections.abc.Mapping)
tt = (1, 2, (30, 40))
hash(tt)
tl = (1, 2, [30, 40])
hash(tl)
tf = (1, 2, frozenset([30,40]))
hash(tf)
zip([1,2],[10,20])
for x in zip([1,2],[10,20]):
    print(type(x), x)

import sys, re
WORD_RE = re.compile(r'\w+')
index = {}
with open(sys.argv[1], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start()+1
            location = (line_no, column_no)
            '''
            occurrences = index.get(word, [])
            occurrences.append(location)
            index[word]=occurrences
            '''
            index.setdefault(word, []).append(location)

for word in sorted(index, key=str.upper):
    print(word, index[word])

import sys, re, collections
WORD_RE = re.compile(r'\w+')
index = collections.defaultdict(list)
with open(sys.argv[1], incoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start+1
            location = (line_no, column_no)
            index[word].append(location)
for word in sorted(index, key=str.upper):
    print(word, index[word])

# StrKeyDict0 : __missing__(self, key)

# collections.OrderedDict.popitem() | popitem(last=True)
# collections.ChainMap
# collections.UserDict
# collections.Counter
ct = collections.Counter('abracadabra')
ct
ct.update('aaaaazzz')
ct
ct.most_common(2)
# MutableMapping, Mapping.get()

from types import MappingProxyType
d = {1: 'A'}
d_proxy=MappingProxyType(d)
d_proxy
d_proxy[1]
d_proxy[2] = 'x'
d[2] = 'B'
d_proxy
d_proxy[2]

s = {1}
type(s)
s.pop()
s
from dis import dis
dis('{1}')
dis('set([1])')
frozenset(range(10))
from unicodedata import name
{chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}

bytes.fromhex('31 4B CE A9')
import array
numbers = array.array('h', [-2, -1,0,1,2])
octets = bytes(numbers)
octets

import struct
fmt = '<3s3sHH'
with open('filter.gif', 'rb') as fp:
    img = memoryview(fp.read())
header = img[:10]
bytes(header)
struct.unpack(fmt, header)
del header
del img

# mmap
for codec in ['latin_1','utf_8','utf_16']:
    print(codec, '')

city = 'São Paulo'
city.encode('utf-8')
city.encode('utf-16')
city.encode('iso8859_1')
city.encode('cp437')
city.encode('cp437', errors='ignore')
city.encode('cp437', errors='replace')
city.encode('cp437', errors='xmlcharrefreplace')

octets = b'Mortr\xe9al'
octets.decode('cp1252')
octets.decode('iso8859_7')
octets.decode('koi8_r')
octets.decode('utf-8')
octets.decode('utf-8', errors='replace')

## Chardet
# chardetect 04-text-byte.asciidoc

u16 = 'El Niňo'.encode('utf-16')
u16
list(u16)

fp = open('cafe.txt','w',encoding='utf-8')
fp
fp.write('café')
fp.close()
import os
os.stat('cafe.txt').st_size
fp2 = open('cafe.txt')
fp2
fp2.encoding
fp2.read()
fp3 = open('cafe.txt',encoding='utf-8')
fp3.read()
fp4 = open('cafe.txt', 'rb')
fp4
fp4.read()

s1 = 'café'
s2 = 'cafe\u0301'
s1, s2
len(s1), len(s2)
s1 == s2

# unicodedata.normalize
# NFC, NFS, NFKC, NFKD
# C : 압축, D : 결합, K : 호환성(호환성 문자)
from unicodedata import normalize
s1 = 'café'
s2 = 'cafe\u0301'
len(s1), len(s2)
len(normalize('NFC', s1)), len(normalize('NFC', s2))
len(normalize('NFD', s1)), len(normalize('NFD', s2))
normalize('NFC', s1) == normalize('NFC', s2)
normalize('NFD', s1) == normalize('NFD', s2)

from unicodedata import normalize, name
ohm = '\u2126'
name(ohm)
ohm_c = normalize('NFC', ohm)
name(ohm_c)
ohm == ohm_c
normalize('NFC', ohm) == normalize('NFC', ohm_c)

half = '½'
normalize('NFKC', half)
four_squared = '42'
normalize('NFKC', four_squared)
micro = 'μ'
micro_kc = normalize('NFKC', micro)
micro, micro_kc
ord(micro), ord(micro_kc)
name(micro), name(micro_kc)

micro = 'μ'
name(micro)
micro_cf = micro.casefold()
name(micro_cf)
micro, micro_cf
eszett = 'ß'
name(eszett)
eszett_cf = eszett.casefold()
eszett, eszett_cf


# BEGIN ASCIIZE
single_map = str.maketrans("""‚ƒ„†ˆ‹‘’“”•–—˜›""",  # <1>
                           """'f"*^<''""---~>""")

multi_map = str.maketrans({  # <2>
    '€': '<euro>',
    '…': '...',
    'Œ': 'OE',
    '™': '(TM)',
    'œ': 'oe',
    '‰': '<per mille>',
    '‡': '**',
})

multi_map.update(single_map)  # <3>


def dewinize(txt):
    """Replace Win1252 symbols with ASCII chars or sequences"""
    return txt.translate(multi_map)  # <4>


def asciize(txt):
    no_marks = shave_marks_latin(dewinize(txt))     # <5>
    no_marks = no_marks.replace('ß', 'ss')          # <6>
    return unicodedata.normalize('NFKC', no_marks)  # <7>
# END ASCIIZE

import locale
locale.setlocale(locale.LC_COLLATE, 'pt_BR.UTF-8')
sorted_fruits = sorted(fruits, key=locale.strxfrm)

import pyuca
coll = pyuca.Collator()
fruits = ['caju', 'atemoia', 'cajá', 'açaí','acerola']
sorted_fruits = sorted(fruits, key=coll.sort_key)
sorted_fruits

str.isidentifier()
str.isprintable()
str.isdecimal()
str.isnumeric()

import regex
m = regex.search(r'(\w\w\K\w\w\w)', 'abcdef')
m

import re
re.ASCII

import os
os.listdir('.')
os.listdir(b'.')
os.fsencode
os.fsdecode
pi_name_bytes = os.listdir(b'.')[1]
pi_name_str = pi_name_bytes.decode('ascii', 'surrogateescape')
pi_name_str
pi_name_str.encode('ascii', 'surrogateescape')



##############################################
# 5장 일급 함수

def factorial(n):
    '''returns gorio!'''
    return 1 if n < 2 else n * factorial(n-1)
factorial(42)
factorial.__doc__
type(factorial)
fact = factorial
fact
fact(5)
map(factorial, range(11))
list(map(factorial, range(11)))
[factorial(n) for n in range(11)]
list(map(factorial, filter(lambda n: n % 2, range(11))))
[factorial(n) for n in range(11) if n % 2]

from functools import reduce
from operator import add
reduce(add, range(100))
sum(range(100))

fruits = ['strawberry','fig','apple','cherry','raspberry','banana']
sorted(fruits, key=lambda word: word[::-1])
sorted(reversed(fruits))
sorted(fruits, reverse=True)

abs, str, 13
[callable(obj) for obj in (abs, str, 13)]

try:
    raise IndexError
except:
    print('Imgoos')

class C: pass
obj = C()
def func(): pass
sorted(set(dir(func)) - set(dir(obj)))

def f(a, *, b):
    return a, b
f(1, b=2)


def clip(text, max_len=80):
    """Return text clipped at the last space before or after max_len
    """
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:  # no spaces were found
        end = len(text)
    return text[:end].rstrip()

clip.__defaults__
clip.__code__
clip.__code__.co_varnames
clip.__code__.co_argcount

from inspect import signature
sig = signature(clip)
sig
str(sig)
for name, param in sig.parameters.items():
    print(param.kind, ':', name, '=', param.default)

my_tag = {'text':'Sunset Gorio', 'max_len':20}
bound_args = sig.bind(**my_tag)
bound_args
for name, value in bound_args.arguments.items():
    print(name, '=', value)
del my_tag['text']
bound_args = sig.bind(**my_tag)
sig.return_annotation


from functools import reduce
def fact(n):
    return reduce(lambda a, b: a*b, range(1, n+1))
fact(5)
from operator import mul
def fact(n):
    return reduce(mul, range(1, n+1))
from operator import itemgetter
metro_data = [
    ('Gorio','Go',36.933, (35.12, 139.13))]
for city in sorted(metro_data, key=itemgetter(1)):
    print(city)
cc_name = itemgetter(1, 0)
for city in metro_data:
    print(cc_name(city))
from operator import attrgetter
name_lat = attrgetter('name', 'coord.lat')
from operator import methodcaller
s = 'The time has come'
upcase = methodcaller('upper')
upcase(s)
hiphenate = methodcaller('replace',' ' ,'-')
hiphenate(s)

from operator import mul
from functools import partial
triple = partial(mul, 3)
triple(7)
list(map(triple, range(1, 10)))
list(map(mul, range(1, 10)))
import unicodedata, functools
nfc = functools.partial(unicodedata.normalize, 'NFC')
s1 = 'café'
s2 = 'cafe\u0301'
s1, s2
s1 == s2
nfc(s1) == nfc(s2)

# lru_cache(), singledispatch(), wraps()

