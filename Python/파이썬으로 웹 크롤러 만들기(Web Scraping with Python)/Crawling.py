
from urllib.request import urlopen
from bs4 import BeautifulSoup

def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    try:
        bsObj = BeautifulSoup(html.read(), "html.parser")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title
    
url = "http://www.pythonscraping.com/pages/page1.html"
title = getTitle(url)
if title == None:
    print("Title could not be found")
else:
    print(title)
    

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, 'html.parser')
nameList = bsObj.findAll('span',{'class':'green'})
for name in nameList:
    print(name.get_text())

html = urlopen("http://www.pythonscraping.com/pages/page3.html")
bsObj = BeautifulSoup(html, 'html.parser')
for child in bsObj.find('table', {'id':'giftList'}).children:
    print(child)
for child in bsObj.find('table', {'id':'giftList'}).tr.next_siblings:
    print(child)
# next_siblings, previous_siblings, next_sibling, previous_sibling

print(bsObj.find('img',{'src':'../img/gifts/img1.jpg'}).parent.previous_sibling.get_text())

import re
images = bsObj.findAll('img',{'src':re.compile('\.\.\/img\/gifts/img.*\.jpg')})
for image in images:
    print(image['src'])

html = urlopen('http://en.wikipedia.org/wiki/Kevin_Bacon')
bsObj = BeautifulSoup(html, 'html.parser')
for link in bsObj.find('div',{'id':'bodyContent'}).findAll('a', href=re.compile('^(/wiki/)((?!:).)*$')):
    if 'href' in link.attrs:
        print(link.attrs['href'])
        
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
pages = set()
def getLinks(pageUrl):
    global pages
    html = urlopen('http://en.wikipedia.org' + pageUrl)
    bsObj = BeautifulSoup(html, 'html.parser')
    try:
        print(bsObj.h1.get_text())
        print(bsObj.find(id='mw-content-text').findAll('p')[0])
        print(bsObj.find(id='ca-edit').find('span').find('a').attrs['href'])
    except AttributeError:
        print('This page is missing something! No worries though!')
    for link in bsObj.findAll('a', href=re.compile('^(/wiki/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                newPage = link.attrs['href']
                print('------------' + newPage)
                pages.add(newPage)
                getLinks(newPage)
getLinks('')

token = '[your api key]'
webRequest = urllib.request.Request('http://myapi.com', headers={'token':token})
html = urlopen(webRequest)

from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os

downloadDirectory = 'downloaded'
baseUrl = 'http://pythonscraping.com'

def getAbsoluteURL(baseUrl, source):
    if source.startswith('http://www.'):
        url = 'http://' + source[11:]
    elif source.startswith('http://'):
        url = source
    elif source.startswith('www.'):
        url = source[4:]
        url = 'http://' + source
    else:
        url = baseUrl + '/' + source
    if baseUrl not in url:
        return None
    return url

def getDownloadPath(baseUrl, absoluteUrl, downloadDirectory):
    path = absoluteUrl.replace('www.','')
    path = path.replace(baseUrl, '')
    path = downloadDirectory + path
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return path

html = urlopen('http://www.pythonscraping.com')
bsObj = BeautifulSoup(html, 'html.parser')
downloadList = bsObj.findAll(src=True)

for download in downloadList:
    fileUrl = getAbsoluteURL(baseUrl, download['src'])
    if fileUrl is not None:
        print(fileUrl)

urlretrieve(fileUrl, getDownloadPath(baseUrl, fileUrl, downloadDirectory))

import csv

csvFile = open('files/test.csv', 'w+')
try:
    writer = csv.writer(csvFile)
    writer.writerow(('number', 'number plue 2', 'number times 2'))
    for i in range(10):
        writer.writerow( (i,i+2,i*2))
finally:
    csvFile.close()

import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://en.wikipedia.org/wiki/Steve_Jobs')
bsObj = BeautifulSoup(html, 'html.parser')
table = bsObj.findAll('table', {"class":'wikitable'})[0]
rows = table.findAll('tr')
csvFile = open('files/editors.csv','wt')
writer = csv.writer(csvFile)
try:
    for row in rows:
        csvRow = []
        for cell in row.findAll(['td', 'th']):
            csvRow.append(cell.get_text().encode('utf-8'))
        writer.writerow(csvRow)
finally:
    csvFile.close()

from urllib.request import urlopen
textPage = urlopen('http://www.pythonscraping.com/pages/warandpeace/chapter1-ru.txt')
print(str(textPage.read(), 'utf-8'))

html = urlopen('http://en.wikipedia.org/wiki/Steve_Jobs')
bsObj = BeautifulSoup(html, 'html.parser')
content = bsObj.find('div',{'id':'mw-content-text'}).get_text()
content = bytes(content, 'utf-8')
content = content.decode('utf-8')

from urllib.request import urlopen
from io import StringIO
import csv

data = urlopen('http://pythonscraping.com/files/MontyPythonAlbums.csv').read().decode(
    'ascii', 'ignore')
dataFile = StringIO(data)
csvReader = csv.reader(dataFile)

for row in csvReader:
    print(row)

dictReader = csv.DictReader(dataFile)
print(dictReader.fieldnames)
for row in dictReader:
    print(row)

from urllib.request import urlopen
from pdfminer.pdfinterp import PDFResourcemanager, process_pdf
from pdfminerl.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open

def readPDF(pdfFile):
    rsrcmgr = PDFResourcesManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()

    return content

pdfFile = urlopen('http://pythonscraping.com/pages/warandpeace/chapter1.pdf')
#pdfFile = open('../pages/warandpeace/chapter1.pdf', 'rb')
outputString = readPDF(pdfFile)
print(outputString)
pdfFile.close()

