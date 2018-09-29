import collections

# do not import dict
class DoppelDict2(collections.UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, [value] * 2)

dd = DoppelDict2(one=1)
dd

class A:
    def ping(self):
        print('A-ping:', self)

class B(A):
    def pong(self):
        print('B-pong:', self)

class C(A):
    def pong(self):
        print('C-PONG:', self)

class D(B, C):
   def ping(self):
       super().ping()
       print('D-post-ping:',self)
   def pingpong(self):
       self.ping()
       super().ping()
       self.pong()
       super().pong()
       C.pong(self)

d = D()
d.ping()
d.pong()
C.pong(d)

D.__mro__
d.pingpong()
bool.__mro__
def print_mro(cls):
    print(', '.join(c.__name__ for c in cls.__mro__))

print_mro(bool)

from FrenchDeck import FrenchDeck
print_mro(FrenchDeck)
import numbers
print_mro(numbers.Integral)
import io
print_mro(io.BytesIO)
print_mro(io.TextIOWrapper)
import tkinter
print_mro(tkinter.Text)

import decimal
ctx = decimal.getcontext()
ctx.prec = 40
one_third = decimal.Decimal('1') / decimal.Decimal('3')
one_third
one_third == +one_third
ctx.prec = 28
one_third
one_third == +one_third
+one_third
from collections import Counter
ct = Counter('abracadabra')
ct
ct['r']=-3
ct
+ct


import re
import reprlib
RE = re.compile('\w+')
class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = RE.findall(text)
    def __getitem__(self, item):
        return self.words[item]
    def __len__(self):
        return len(self.words)
    def __repr__(self):
        return '(Sentence(%s)' % reprlib.repr(self.text)


s = Sentence('"The time has come," the Walrus said,')
s
for word in s:
    print(word)
list(s)
class Foo:
    def __iter__(self):
        pass
from collections import abc
issubclass(Foo, abc.Iterable)
f = Foo()
isinstance(f, abc.Iterable)

class Sentence:
    def __init__(self, text):
        self.text = text
    def __repr__(self):
        return '(Sentence(%s)' % reprlib.repr(self.text)
    def __iter__(self):
        # for match in RE.finditer(self.text):
        #    yield match.group()
        return (match.group() for match in RE.finditer(self.text))

def gen_123():
    yield 1
    yield 2
    yield 3
gen_123
gen_123()
for i in gen_123():
    print(i)
g = gen_123()
next(g)

# 무한수열
def aritprog_gen(begin, step, end=None):
    result = type(begin + step)(begin)
    forever = end is None
    index = 0
    while forever or result < end:
        yield result
        index += 1
        result = begin + step*index

import itertools
gen = itertools.count(1, .5)
next(gen)
gen = itertools.takewhile(lambda n: n<3, itertools.count(1, .5))
list(gen)

def aritprog_gen(begin, step, end=None):
    first = type(begin + step)(begin)
    ap_gen = itertools.count(first, step)
    if end is not None:
        ap_gen = itertools.takewhile(lambda n: n<end, ap_gen)
    return ap_gen

# filtering
itertools.compress(it, selector_it)
itertools.dropwhile(predicate, it)
filter(predicate, it)
itertools.filterfalse(predicate, it)
itertools.islice(it.stop)
itertools.islice(it, start, stop, step=1)
itertools.takewhile(predicate, it)

# mapping
itertools.accumulate(it, [func])
enumerate(it, start=0)
map(func, it1, [it2, ..., itN])
itertools.starmap(func, it)

# merging
itertools.chain(it1, ..., itN)
itertools.chain.from_iterable(it)
itertools.product(it1, ..., itN, repeat=1)
zip(it1, ..., itN)
itertools.zip_longest(it1, ..., itN, fillvalue=None)

# expansion
itertools.combinations(it, out_len)
itertools.combinations_with_replacement(it, out_len)
itertools.count(start=0, step=1)
itertools.cycle(it)
itertools.permutations(it, out_len=None)
itertools.repeat(item, [times])

# rearrangement
itertools.groupby(it, key=None)
reversed(seq)
itertools.tee(it, n=2)

# reduce
all(it)
any(it)
max(it, [key=,] [default=])
min(it, [key=,] [default=])
import functools
functools.reduce(func, it, [initial])
sum(it, start=0)



def chain(*iterables):
    for i in iterables:
        yield from i
s = 'ABC'
t = tuple(range(3))
list(chain(s, t))

with open('mydata.txt') as fp:
    for line in iter(fp.readline, ''):
        pass


def fibonacci(end=None):
    a, b = 0, 1
    cnt = 0
    while cnt < end:
        yield a
        a, b = b, a+b
        cnt += 1
for f in fibonacci(100):
    print(f)



import contextlib
@contextlib.contextmanager























