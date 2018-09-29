
fout = open('oops.txt','wt')
print('Oops, I created Imgoos.', file=fout)
fout.close()

import os
os.path.exists('Oops.txt')
os.path.exists('./oops.txt')
os.path.exists('waffles')
os.path.exists('.')
os.path.exists('..')

name = 'oops.txt'
os.path.isfile(name)
os.path.isdir(name)
os.path.isdir('.')
os.path.isabs('name')
os.path.isabs('/big/fake/name')
os.path.isabs('big/fake/name/withoua/a/leading/slash')
import shutil
shutil.copy('oops.txt','ohno.txt')
shutil.move('ohno.txt','ohno2.txt')
os.rename('ohno2.txt', 'ohno.txt')
os.link('oops.txt','yikes.txt')
os.path.isfile('yikes.txt')
os.path.islink('yikes.txt')
os.symlink('oops.txt','jeepers.txt')
os.path.islink('jeepers.txt')
os.chmod('oops.txt',0o400)
import stat
os.chmoe('oops.txt', stat.S_IRUSR)
os.path.abspath('oops.txt')
os.remove('oops.txt')
os.path.exists('oops.txt')
os.mkdir('poems')
os.path.exists('poems')
os.rmdir('poems')
os.path.exists('poems')
os.mkdir('poems')
os.listdir('poems')
os.mkdir('poems/mcintyre')
os.listdir('poems')
fout = open('poems/mcintyre/the_good_man','wt')
fout.write('Imgoos is king')
fout.close()
os.listdir('poems/mcintyre')
os.chdir('poems')
os.listdir('.')
import glob
glob.glob('m*')
glob.glob('??')
glob.glob('m??????e')
glob.glob('[klm]*e')
os.getpid()
os.getcwd()

import subprocess
import multiprocessing
import calendar
calendar.isleap(1900)
from datetime import date
halloween = date(2015,10,31)
halloween
halloween.day
halloween.month
halloween.isoformat()
now = date.today()
from datetime import timedelta
one_day = timedelta(days=1)
tomorrow = now + one_day
tomorrow
now + 17 * one_day
now - one_day
from datetime import time
noon = time(12,0,0)
noon
noon.hour
noon.microsecond
from datetime import datetime
some_day = datetime(2015,1,2,3,4,5,6)
some_day
some_day.isoformat()
now = datetime.now()
now.minute
now.microsecond
noon = time(12)
this_day = date.today()
noon_today = datetime.combine(this_day, noon)
noon_today
noon_today.date()
noon_today.time()
import time
now = time.time()
now
time.ctime(now)
time.localtime(now)
time.gmtime(now)
tm = time.localtime(now)
time.mktime(tm)
time.ctime(time.time())
fmt = "It's %A, %B %d, %Y, local time %I:%M:%S%p"
t = time.localtime()
t
time.strftime(fmt, t)
fmt = "%Y-%m-%d"
time.strptime("2015-06-02", fmt)

import locale
from datetime import date
halloween = date(2015, 10, 31)
for lang_country in ['ko-kr', 'en-us']:
    locale.setlocale(locale.LC_TIME, lang_country)
    halloween.strftime('%A, %B %d')

names = locale.locale_alias.keys()
good_names = [name for name in names if len(name) == 5 and name[2] == '_']
good_names[:5]
de = [name for name in names if name.startswith('de')]
de
# arrow dateutil iso8601 fleming
