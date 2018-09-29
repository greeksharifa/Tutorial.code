poem = '''Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
It has survived not only five centuries, but also the leap into electronic typesetting, 
remaining essentially unchanged. 
It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including 
versions of Lorem Ipsum.'''
len(poem)
fout = open('Lorem ipsum.txt', 'wt')
fout.write(poem)
print(poem, file=fout)
fout.close()

fin = open('Lorem ipsum.txt','rt')
poem = fin.read()
fin.close()
print(len(poem))

poem = ''
fin = open('Lorem ipsum.txt','rt')
chunk = 100
while True:
    fragment = fin.read(chunk)
    if not fragment:
        break
    poem += fragment
fin.close()
len(poem)

poem = ''
fin = open('Lorem ipsum.txt','rt')
while True:
    line = fin.readline()
    if not line:
        break
    poem += line
fin.close()
len(poem)

poem = ''
fin = open('Lorem ipsum.txt','rt')
for line in fin:
    poem+=line
fin.close()
len(poem)

fin = open('Lorem ipsum.txt','rt')
lines = fin.readlines()
len(lines)
for line in lines:
    print(line, end='')
fin.close()
len(poem)

bdata = bytes(range(0,256))
len(bdata)
fout = open('bfile','wb')
fout.write(bdata)
fout.close()
with open('relativity', 'wt') as fout:
    fout.write(poem)
fin = open('bfile','rb')
fin.tell()
fin.seek(255)
bdata = fin.read()
len(bdata)
bdata[0]

import os
os.SEEK_SET
os.SEEK_CUR
os.SEEK_END
fin.seek(-1,2)
fin.tell()
bdata = fin.read()
len(bdata)
bdata[0]

import csv
villains = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big'],
    ['Auric', 'Goldfinger'],
    ['Ernst', 'Blofled'],
    ]
with open('villains', 'wt') as fout:
    csvout = csv.writer(fout)
    csvout.writerows(villains)

with open('villains', 'rt') as fin:
    cin = csv.reader(fin)
    villains = [row for row in cin]
print(villains)

with open('villains', 'rt') as fin:
    cin = csv.DictReader(fin, fieldnames=['first','last'])
    villains = [row for row in cin]
print(villains)

villains = [
    {'first': 'Doctor', 'last': 'No'},
    ]
with open('villains', 'wt') as fout:
    cout = csv.DictWriter(fout, ['first', 'last'])
    cout.writeheader()
    cout.writerows(villains)

import xml.etree.ElementTree as et
import json
import configparser
# MsgPack Protocol Buffers Avro Thrift 
import pickle

import sqlite3
conn = sqlite3.connect('enterprise.db')
curs = conn.cursor()
curs.execute('''Create Table zoo(
    (critter varchar(20) primary key, 
    count int,
    damages float)''')
ins = 'insert into zoo (critter,count,damages) values(?, ?, ?)'
curs.execute(ins, ('weasel', 1, 2000.0))
curs.execute('select * from zoo')
rows = curs.fetchall()
print(rows)

# NoSQL
import dbm
db = dbm.open('definitions', 'c')
db['mustard'] = 'yellow'
len(db)
db['mustard']
db.close()
db = dbm.open('definitions', 'r')
db['mustard']

import memcache
import redis
