# #############################################
# 7장 함수 데커레이터와 클로저

def deco(func):
    def inner():
        print('running inner()')

    return inner

@deco
def target():
    print('running target()')

target()
target

import dis

dis.dis('1+2')

class Averager():
    def __init__(self):
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total / len(self.series)

avg = Averager()
avg(10)
avg(11)
avg(12)

def make_averager():
    series = []

    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return total / len(series)

    return averager

avg = make_averager()
avg(10)
avg(11)
avg(12)
avg.__code__.co_varnames
avg.__code__.co_freevars
avg.__closure__
avg.__closure__[0].cell_contents

def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return averager

avg = make_averager()
avg(10)
avg(11)
avg(12)

def clock(func):
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result

    return clocked

@clock
def snooze(seconds):
    time.sleep(seconds)

@clock
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)

if __name__ == '__main__':
    print('*' * 40, 'Calling snooze(.123)')
    snooze(.123)
    print('*' * 40, 'Calling factorial(6)')
    print('6! =', factorial(6))

import time, functools

def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result

    return clocked

@functools.lru_cache(maxsize=128, typed=False)
@clock
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 2) + fibonacci(n - 1)

if __name__ == '__main__':
    print(fibonacci(6))

from functools import singledispatch
import html

@singledispatch
def htmlize(obj):
    content = html.escape(repr(obj))
    return '<>'

@htmlize.register(str)
def _(text):
    content = html.escape(text).replace('\n', '<br>\n')

class Gizmo:
    def __init__(self):
        print('Gizmo id: %d' % id(self))

x = Gizmo()
y = Gizmo() * 10
dir()

t1 = (1, 2, [30, 40])
t1[-1] += [50]
t1

import copy

lst1 = [1, (10, 20, 30), [5, 6, 7]]
lst2 = copy.copy(lst1)
lst3 = copy.deepcopy(lst1)
id(lst1), id(lst2), id(lst3)
del lst1[0]
lst1, lst2, lst3
del lst1[0]

import weakref

s1 = {1, 2, 3}
s2 = s1

def bye():
    print('Gone with the wind...')

ender = weakref.finalize(s1, bye)
ender.alive
del s1
ender.alive
s2 = 'spam'
ender.alive

a_set = {0, 1}
wref = weakref.ref(a_set)
wref
wref()
a_set = {2, 3, 4}
wref()
wref() is None
wref() is None

stock = weakref.WeakValueDictionary()
fs = frozenset({1, 2})
fs

# ###############################################
# 10

from array import array
import reprlib
import math
import numbers
import functools
import operator
import itertools

class Vector:
    _imgoos = 'Yamgoos'
    __IMGOOS = 'YAMGOOS'
    typecode = 'd'
    def __init__(self, components):
        self._components = array(self.typecode, components)
    def __iter__(self):
        return iter(self._components)
    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return 'Vector({})'.format(components)
    def __str__(self):
        return str(tuple(self))
    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +
                bytes(self._components))
    def __eq__(self, other):
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))
    def __hash__(self):
        hashes = map(hash, self._components)
        return functools.reduce(operator.xor, hashes, 0)
    def __abs__(self):
        return math.sqrt(sum(x*x for x in self))
    def __bool__(self):
        return bool(abs(self))
    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)
    def __len__(self):
        return len(self._components)
    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))
    shortcut_names = 'xyzt'
    def __getattr__(self, item):
        cls = type(self)
        if len(item) == 1:
            pos = cls.shortcut_names.find(item)
            if 0 <= pos < len(self._components):
                return self._components[pos]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, item))
    def __setattr__(self, key, value):
        cls = type(self)
        if len(key) == 1:
            if key in cls.shortcut_names:
                error = 'readonly attribute {attr_name!r}'
            elif key.islower():
                error = "can't set attributes 'a' to 'z' in {cls_name!r}"
            else:
                error = ''
            if error:
                msg = error.format(cls_name=cls.__name__, attr_name=key)
                raise AttributeError(msg)
        super().__setattr__(key, value)
    def angle(self, n):
        r = math.sqrt(sum(x*x for x in self[n:]))
        a = math.atan2(r, self[n-1])
        if (n == len(self)-1) and (self[-1] < 0):
            return math.pi*2-a
        else:
            return a
    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))
    def __format__(self, format_spec=''):
        if format_spec.endswith('h'):
            format_spec = format_spec[:-1]
            coords = itertools.chain([abs(self)], self.angles())
            outer_fmt = '<{}>'
        else:
            coords = self
            outer_fmt = '({})'
        components = (format(c, format_spec) for c in coords)
        return outer_fmt.format(', '.join(components))

v1 = Vector([3, 4, 5])
len(v1)
v1[0], v1[-1]
v7 = Vector(range(7))
v7[1:4]
help(slice.indices)

from itertools import zip_longest
list(zip_longest(range(3), 'ABC', [0.0, 1.1, 2.2, 3.3], fillvalue=-1))

v1._imgoos
v1._Vector__IMGOOS


# ###########################
# Chap 11


from random import shuffle
l = list(range(10))
shuffle(l)
l


from FrenchDeck import FrenchDeck

deck = FrenchDeck()

def set_card(deck, position, card):
    deck._cards[position] = card
FrenchDeck.__setitem__ = set_card
shuffle(deck)
deck[:5]

class Struggle:
    def __len__(self): return 23

from collections import abc
isinstance(Struggle(), abc.Sized)
abc.Sequence
abc.MutableSequence
# @abc.abstractmethod

















