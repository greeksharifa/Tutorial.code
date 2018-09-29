# Chap 6

from collections import namedtuple

#Chap 7

import unicodedata as ud
def unicode_test(value):
    name = ud.name(value)
    value2 = ud.lookup(name)
    print('value="%s", name="%s", value2 = "%s"' % (value, name, value2))

unicode_test('A')
unicode_test('$')
unicode_test('\u00a2')
unicode_test('\u20ac')
unicode_test('\u2603')
place = 'Café'
place
ud.name('\u00e9')
ud.lookup('LATIN SMALL LETTER E WITH ACUTE') # E WITH ACUTE, LATIN SMALL LETTER 
place = 'caf\u00e9'
place
place = 'caf\N{LATIN SMALL LETTER E WITH ACUTE}'
place
u_umlaut = '\N{LATIN SMALL LETTER U WITH DIAERESIS}'
u_umlaut
drink = 'Gew' + u_umlaut + 'rztraminer'
print('Now I can finally have my', drink, 'in a', place)
len('$')
len('\U0001f47b')
'\U0001f47b'
snowman = '\u2603'
len(snowman)
ds = snowman.encode('utf-8')
len(ds)
ds
snowman.encode('ascii', 'ignore')
snowman.encode('ascii', 'replace')
snowman.encode('ascii', 'backslashreplace')
snowman.encode('ascii', 'xmlcharrefreplace')
# ascii utf-8 latin-1 cp-1252 unicode-escape(\uxxxx or \Uxxxxxxxx)
place = 'caf\u00e9'
place
type(place)
place_bytes = place.encode('utf-8')
place_bytes
type(place_bytes)
place2 = place_bytes.decode('utf-8')
place2

import re
result = re.match('You', 'Young Frankenstein')
result
youpattern = re.compile('You')
result = youpattern.match('Young Frankenstein')
help(re)
# search() findall() split() sub()
source = 'Young Frankenstein'
m = re.match('You', source)
if m:
    print(m.group())
m = re.match('Frank', source)
if m:
    print(m.group())
bekgoos = '알게머람?'
m = re.search('Frank',source)
if m:
    print(m.group())
m = re.match('.*Frank',source)
if m:
    print(m.group())
m = re.findall('n', source)
m
print('Found', len(m), 'matches')
m = re.findall('n.', source)
m
m = re.findall('n.?', source)
m
m = re.split('n', source)
m
m = re.sub('n', '?', source)
m
'''
\d  숫자
\D  비숫자
\w  알파벳 문자, '_'
\W  비알파벳 문자
\s  공백 문자
\S  비공백 문자
\b  단어 경계(\w와 \W 또는 \W와 \w 사이의 경계)
\B  비단어 경계
'''
import string
printable = string.printable
len(printable)
printable[0:50]
printable[50:]
re.findall('\d', printable)
re.findall('\w', printable)
re.findall('\s', printable)
x = 'abc' + '-/*' + '\u00ea' + '\u0115'
re.findall('\w', x)

source = '''I wish I may, I wish I might
Have a dish of fish tonight.'''
re.findall('wish', source)
re.findall('wish|fish', source)
re.findall('^wish', source)
re.findall('^I wish', source)
re.findall('.*fish tonight.$', source)
re.findall('fish tonight\.$', source)
re.findall('[wf]ish', source)
re.findall('[wfs]+', source)
re.findall('ght\W', source)
re.findall('I (?=wish)', source)
re.findall('(?<=I) wish', source)
re.findall(r'\bfish', source)
m = re.search(r'(. dish\b).*(\bfish)', source)
m.group()
m.groups()
m = re.search(r'(?P<DISH>. dish\b).*(?P<FISH>\bfish)', source)
m.group()
m.groups()
m.group('DISH')
m.group('FISH')
