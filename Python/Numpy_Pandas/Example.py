# Spring Code Sheet
# Python for Data Analysis
# p51 - p58
# MovieLens 영화평점데이터
import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

users = pd.read_table('C:/Users/YY/Documents/Winter Data/ml-1m/users.dat', sep = '::', header = None, names = unames)
ratings = pd.read_table('C:/Users/YY/Documents/Winter Data/ml-1m/ratings.dat', sep = '::', header = None, names = rnames)
movies = pd.read_table('C:/Users/YY/Documents/Winter Data/ml-1m/movies.dat', sep = '::', header = None, names = mnames)

users[:5]
ratings[:5]
movies[:5]

type(users)    # pandas.core.frame.DataFrame

# merge로 DF를 합쳐준다.
data = pd.merge(pd.merge(ratings, users), movies)
type(data)     # pandas.core.frame.DataFrame
data[:5]
data.ix[0]

# 성별에 따른 각 영화의 평균 평점을 구함: pivot_table 메서드
mean_ratings = data.pivot_table('rating', index = 'title', columns = 'gender', aggfunc = 'mean')

# 영화제목으로 데이터를 그룹화하고 size 함수를 통해 제목별 평점건수(size)를 Series 객체로 얻어냄
# 250건 이상의 정보가 있는 영화만 추림 -> Index로 활용
ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
type(active_titles)           # pandas.core.indexex.base.Index
active_titles


mean_ratings = mean_ratings.ix[active_titles]
mean_ratings[:5]

# 여성에게 높은 평점을 받은 영화목록 확인
top_female_mean_ratings = mean_ratings.sort_values(by = 'F', ascending = False)
top_female_mean_ratings[0:10]

# 평점 성별 차이
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by = 'diff')
sorted_by_diff[:10]
sorted_by_diff[::-1][:10]

# 분산 순 정렬: 호불호 갈리는 영화 찾기
ratings_std_by_title = data.groupby('title')['rating'].std()

ratings_std_by_title = ratings_std_by_title.ix[active_titles]
ratings_std_by_title.sort_values(ascending = False)[:10]






#############
### numpy ###
############# p119

# ndarray의 모든 원소는 같은 자료형
# 모든 배열은 차원크기: shape, 자료형: dtype이라는 객체를 갖고 있음
import numpy as np

# ndarray생성
data1 = [[1,2,3,4], [3,4,5,6]]
arr1 = np.array(data1)
arr1
arr1.shape
arr1.dtype

np.zeros((4,4))
np.empty((4,4))
np.ones((4,4))

np.empty((2,3,2))  # 초기화 되지 않은 배열을 생성함
np.eye(4,4)        # 단위행렬
np.arange(15)
np.arange(15,20,1) # range 메서드의 numpy 버전, 리스트 대신 ndarray를 반환함

# dtype 변경
arr0 = np.array([1,2,3], dtype = np.float64)
arr0.dtype
arr2 = np.array([[1,2,3],[3,2,2]])
float_arr2 = arr2.astype(np.float64)
float_arr2

numeric_strings = np.array(['1.24', '-9.5'], dtype = np.string_)
numeric_strings.astype(np.float).dtype

# 벡터화 - 배열계산: 반복문을 쓰지 않을 수 있다.
arr = np.array([[1,2,3], [1,2,3]], dtype = 'float64')
arr * arr      # 스칼라 곱셈
arr ** arr     # 스칼라 지수곱셈
arr - arr

# 색인과 슬라이싱
# 1차원 배열은 리스트와 유사하다
arr = np.arange(10)
arr[5]
arr[5:8:1]
arr[5:8] = 12
arr            # broadcasting

# numpy의 배열의 일부분, 즉 배열조각은 원본의 일부이다.
# 배열 조각을 변경하면 원본도 바뀐다.  데이터가 무작정 복사되는 것이 아니다.
# 이를 원본의 view를 얻는다고 한다.
arr_slice = arr[5:8]
arr_slice[1:3] = 100
arr_slice
arr

# view 대신 복사를 하고 싶으면
arr = np.arange(10)
arr_copy = arr[5:8].copy()   # copy메서드는 리스트, 배열에 적용된다.
arr_copy[:] = 99
arr_copy
arr

# 다차원 배열
arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr2d
arr2d[0][1]   # arr2d[0, 1]와 같은 표현이다.
arr2d[0, 1]
# 배열의 부분집합은 모두 배열의 뷰를 반환한다.

# 슬라이스 색인
arr2d[:2][1: , :2]
arr2d[:, :1]

# numpy.random 서브 패키지
# seed: 의사 랜덤 상태 설정
# shuffle: 뒤섞기 / choice: 샘플링
x = np.arange(10)
np.random.shuffle(x)
x

np.random.choice(x, 5, replace = False)
np.random.choice(x, 3, replace = False)
np.random.choice(10, 10)                   # 디폴트가 True
np.random.choice(5, 10, p = [0, 0, 0.5, 0, 0.5])

# random_integers: uniform integer
# rand: uniform
# randn: 정규 분포
np.random.random_integers(-10, 10, 5)  # -10부터 10까지 5개
np.random.rand(10, 5)   # 0-1 균일분포를 10 X 5 배열로 생성하라
np.random.randn(10, 5)  # 표준정규분포를 10 X 5 배열로 생성하라

# 카운트 함수
np.unique([11,11,2,2,34,34])

a = np.array([1,1], [2,3])
np.unique(a)

# 부울 색인
names = np.array(['bob', 'joe', 'will', 'bob', 'will', 'joe', 'joe'])
data = np.random.randn(7, 4)   # 표준정규분푸 7 X 4 배열 생성
names                          # 7 X 1 열벡터임
data

names == 'bob'
names.dtype  # 불 타입이 되었으므로 색인으로 활용할 수 있다.

# data[index]꼴
data[names == 'bob', 2:]
data[names != 'bob', 2:]
data[~(names == 'bob'), 2:]

index = (names == 'bob') | (names == 'will')
index
data[index]
data[data < 0] = 0
data

# 팬시 색인: 정수 배열을 사용한 색인
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr

# 특정한 순서로 행을 선택하고 싶다?
arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
arr
# 비교해보자
arr[[1,5,7,2], [0,3,1,2]]         # 한방에 처리
arr[[1,5,7,2]][:, [0,3,1,2]]     # 단계별로 처리
arr[np.ix_([1,5,7,2], [0,3,1,2])]

# 배열 전치와 축 바꾸기
arr = np.arange(15).reshape((3, 5))
arr
arr.T  # transpose
np.dot(arr.T, arr)  # X^t %*% X
# transpose 메서드, swapaxes 메서드 추가적으로 공부할 것


# 유니버셜 함수: ufunc
# 단항 유니버셜
arr = np.arange(10)
np.exp(arr)

# 이항 유니버셜
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)


# 벡터화 연산: 배열을 사용한 데이터 처리 p141
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys
ys.shape
z = np.sqrt(xs ** 2 + ys ** 2)

import matplotlib.pyplot as plt
z
plt.imshow(z, cmap = plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()


# 조건절을 벡터화 연산으로 표현
import numpy as np
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# cond의 값이 True일 때 x값을 출력하고 싶다.
# zip: 동일한 크기의 자료형을 각각 묶어 줌
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result
result2 = np.where(cond, xarr, yarr)
result2

arr = np.random.randn(4, 4)    # 표준정규분포를 4 X 4로
np.where(arr > 0, 2, -2)       # 중요한 기능
np.where(arr > 0, 0, arr)

# 수학-통계 메서드
arr = np.arange(32).reshape((2,4,4))
arr
arr.mean()   # = np.mean(arr)
arr.sum()
arr.mean(0)
arr.sum(0)          # 차원이 큰 배열단계부터 0, 1, 2 이런 식으로 축 배정
arr.sum(axis = 0)   # 3차원 축
arr.sum(axis = 1)   # 열
arr.sum(axis = 2)   # 행
arr = np.array([[0,1,2], [3,4,5], [6,7,8]])
arr.cumsum(0)
arr.cumprod(0)

# 부울 배열을 위한 메서드
arr = np.random.randn(100)
(arr > 0).sum()
bools = np.array([False, False, True, False])
bools.all()
bools.any()

# 정렬
arr = np.random.randn(8)
arr
arr.sort()
arr
arr = np.random.randn(5, 3)
arr
arr.sort(axis = 1)      # 행별로 정렬될 것이다.
arr

# 배열의 분위수 구하기
large_arr = np.random.randn(100)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]   # 5% quantile
names = np.array(['bob', 'joe', 'bob', 'joe', 'will', 'will', 'joe'])
sorted(set(names))
np.unique(names)
# 집합함수: unique, intersect1d, union1d, in1d, setdiff1d, setxor1d

# 입출력
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')

# !cat array_ex.txt
arr = np.loadtxt('array_ex.txt', delimiter=',')

# 선형대수
x = np.array([[1,2,3], [4,5,6]])
np.dot(x, np.ones(3))

import numpy as np
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = np.dot(X.T, X)    # = X.T.dot(X)
inv(mat)
np.dot(mat, inv(mat))   # = mat.dot(inv(mat))
q, r = qr(mat)
r

# diag, dot, trace, linalg.det, linalg.eig, linalg.qr, linalg.solve,
# linalg.lstsq

# 난수 생성
samples = np.random.normal(size = (4, 4))
samples

from random import normalvariate
N = 1000000
# %timeit samples = [normalvariate(0, 1) for _ in range(N)]

# seed, permutation, shuffle, rand, randint, randn, binomial, normal, beta, chisquare, gamma, uniform

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size = (nwalks, nsteps))  # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
walks.max()
walks.min()
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()         # 30 혹은 -30에 도달한 시뮬레이션의 개수


##############
### pandas ###
##############
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# Series
obj = pd.Series([1,2,3,4])
obj2 = pd.Series([1,2,3,4], index = ['a','b','c','d'])
obj.values
obj.index
obj
obj2
obj2[['a', 'b']]
obj[obj > 3]

# 파이썬의 사전형과 비슷하게 작동한다.
'a' in obj2
data_beck = {'ohio':3000, 'max':200, 'james':250}
obj3 = pd.Series(data_beck)
obj3

# 키 값: 사전 객체
states = ['ohio', 'max', 'texas']
obj4 = pd.Series(data_beck, index = states)   # data자리에는 사전데이터도 되고, Series도 된다.
obj3
pd.isnull(obj4)
pd.notnull(obj4)
obj4[pd.isnull(obj4)]
obj4

# Series 객체와 색인은 모두 name 속성이 있다.
obj4.name = 'population'
obj4.index.name = 'states'
obj4
# 대입으로 색인 변경
obj4.index = ['bob', 'steve', 'imgoos']
obj4


# DataFrame
data_imgoos = {'state' : ['ohio', 'ohio', 'ohio', 'nevada', 'nevada'],
               'year' : [2001, 2002, 2003, 2004, 2005],
               'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}
type(data_imgoos) 
data_fr = pd.DataFrame(data_imgoos)
type(data_fr)       # pandas.core.frame.DataFrame
data_fr

data_fr2 = pd.DataFrame(data_imgoos,
                        columns = ['state', 'year', 'pop', 'what'],
                        index = ['one', 'two', 'three', 'four', 'five'])
# 컬럼 접근
data_fr2['state']
data_fr2.state
data_fr2['what'] = np.arange(5)
data_fr2

val = pd.Series([-1, -3, -4], index = ['one','two','five'])
data_fr2['what'] = val
data_fr2

data_fr2['new'] = 3
data_fr2
data_fr2.columns
del data_fr2['new']

# 로우 접근
data_fr2.ix['three']

# 사전-Series로 DF를 만들면 컬럼이 하나뿐이다.  여러개로 한번에 만들려면?
# 중첩된 사전을 이용하면 된다.  그럼 차례대로 칼럼(키), 로우, 값으로 편성된다.
pop = {'fly':{2001:30, 2002:35},
       'emirates':{2001:25, 2002:30}}
df = pd.DataFrame(data = pop)
df
df.T

df2 = pd.DataFrame(data = pop, index = [2001, 2002, 2003])
df2
df2.ix[2003] = 4
df2

df2.index.name = 'year'
df2.columns.name = 'state'
df2

# Series와 유사하게 values 속성은 DF에 저장된 데이터를 2차원 배열로 반환한다.
df2.values   # array


# 색인 객체: 색인 객체는 변경될 수 없다.
index = pd.Index(np.arange(3))
obj = pd.Series(data = [1.5, 2.3, 3.3], index = index)
obj.index is index
# 색인 메서드와 속성: apopend, diff, intersection, union, isin, delete
# drop, insert, is_monotonic, is_unique, unique

obj = Series([4.5,7.2,-5.3,3.6], index=['d','b','a','c'])
obj
obj2 = obj.reindex(list('abcde'))
obj2
obj.reindex(list('abcde'), fill_value=0)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0,2,4])
obj3.reindex(range(6), method='ffill')
# method: ffill, pad / bfill, backfill

# 핵심기능 p175
# 재색인: reindex - 로우, 칼럼 다 가능
# 중요: reindex는 '뷰'가 아니다.  새로운 객체를 생성한다.
import numpy as np
import pandas as pd
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index = ['d','b','c','a'])
obj2 = obj.reindex(['a','b','c','d','e'])     # 재배열에 NaN추가 기능까지 있음
obj2 = obj.reindex(['a','b','c','d','e'], fill_value = 'a')
obj2

# ex) 시계열: 누락된 값을 보간하거나 채워넣고 싶을 때
obj3 = pd.Series(['blue','purple','yellow'], index = [0,2,4])
obj3.reindex(range(6), method = 'ffill')    # 앞의 값으로 채워넣는다.
obj3.reindex(range(6), method = 'backfill')

frame = pd.DataFrame(data = np.arange(9).reshape((3,3)),
                     index = np.arange(3),
                     columns = ['ohio','texas','california'])
frame2 = frame.reindex(columns = ['baby', 'teeth', 'brush'])
frame2                         # 새로운 칼럼을 생성하면 값은 없어질 수 밖에

# 재색인은 ix를 이용해서 라벨로 색인하면 더 간결하다.
states = ['ohio','texas','maybe']
frame.ix[[1,2,3,4], states]
frame.ix[[1,2,3,4]]
frame

# 만약 그냥 삭제하고 싶으면?
del frame2['baby']
frame2
# 추가하고 싶으면?
val = pd.Series([1,2,3], index = ['a','c','d'])
frame2['new'] = val
frame2

# 하나의 로우/칼럼 삭제하기
obj = pd.Series(data = np.arange(5), index = ['a','b','c','d','e'])
obj.drop(['a','c'])
obj_reduced = obj.drop('c')
obj_reduced

obj = pd.Series(data = [2,3,2], index = ['a','b','c'])

df = pd.DataFrame(data = np.arange(16).reshape((4,4)),
                  index = ['ohio','colorado','maybe','may'],
                  columns = ['a','b','c','d'])
df2 = df.drop(['ohio','may'], axis = 0)    # 행(로우)이 axis = 0
df3 = df.drop(['a','b'], axis = 1)         # 컬럼이 axis = 1
df2
df3

# 색인하기, 선택하기, 거르기
import pandas as pd
import numpy as np
obj = pd.Series(data = np.arange(4.), index = ['a','b','c','d'])
obj[obj < 2]
obj['a':'c']           # index로 슬라이싱하면 시작, 끝점 모두 포함한다.(그럴수밖에)
obj['a':'c'] = 30
obj

# 데이터프레임은 기본적으로 칼럼 접근이다.  []안에 칼럼을 쓰는거다
# 로우를 선택하고 싶으면?
df = pd.DataFrame(data = np.arange(16).reshape((4,4)),
                  index = ['ohio','colorado','maybe','may'],
                  columns = ['one','two','three','four'])
df
# df[하나만] >> 칼럼 색인
# df.ix[하나만] >> 로우 색인
# df.ix[로우, 칼럼] >> 동시 색인

# 칼럼색인
df['two']
df.two

# 로우 색인
df.ix[['ohio', 'maybe']]
df.ix[2]

# 동시 색인
df.ix[:2, 'two']
df.ix[1:3, ['two', 'three']]
df.ix[df.three > 5, :3]


# 객체(Series, DF)간 산술연산 메서드
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index = ['a','c','d','e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index = ['a','c','e','f','g'])
s1 + s2

df1 = pd.DataFrame(np.arange(9.).reshape((3,3)), index = ['ohio','texas','colorado'],
                   columns = list('bcd'))
df2 = pd.DataFrame(np.arange(12.).reshape((4,3)), index = ['utah','ohio','texas','oregon'],
                   columns = list('bde'))
df1
df2
df1 + df2

# reindex는 빈값을 채워넣으면서 재색인할 수는 있지만 산술여산 메서드는 아니다.
df4 = df1.reindex(columns = df2.columns, fill_value = 0)

# 빈 값에 적용할 값을 입력하고 산술연산을 할 수 있다.
# 각각 매치가 되지 않는, 한쪽에는 존재하고 한쪽에는 값이 없는 곳에는 0이 있다고 가정한다.
# 양쪽에 다 없으면 NaN이 된다.
#종류: add, sub, mul, div
df3 = df1.add(df2, fill_value = 0)
df3

df4 = df1.mul(df2, fill_value = 1)
df4

# 배열 연산
arr = np.arange(12.).reshape((3,4))
arr
arr[0]
arr - arr[0]

# DF와 Series간의 연산
frame = pd.DataFrame(np.arange(12.).reshape((4,3)), index = ['utah','ohio','texas','oregon'],
                     columns = list('bde'))
series = frame.ix[0]
frame
series  # Series의 색인을 DF의 칼럼에 맞추고 아래 로우로 전파한다.
frame - series
# 만약 색인 값을 DF의 칼럼이나 Series의 색인에서 못찾으면 그 객체는 재색인됨
series2 = pd.Series(data = range(3), index = ['b','e','f'])
frame + series2
# DF, Series간 산술연산
series3 = frame['d']
series3
frame.sub(series3, axis = 0)

# 함수 적용과 매핑
frame = pd.DataFrame(np.random.randn(4,3), columns = list('bde'),
                     index = ['utah', 'ohio', 'texas', 'oregon'])
frame
frame.abs()      # = np.abs(frame)
# apply메서드
temp_f = lambda x: x.max() - x.min()
frame.apply(temp_f, axis = 0)
frame.apply(temp_f, axis = 1)
format = lambda x: '%.2f' % x
frame.applymap(format)



# 정렬과 순위: order, sort_values, sort_index
series = pd.Series(data = range(4), index = ['d','b','c','a'])
series.sort_index()    # 색인에 따른 정렬
series.sort_values()   # 값에 따른 정렬
frame = pd.DataFrame(np.arange(8).reshape((2,4)), columns = ['d','a','b','c'],
                     index = ['three', 'one'])
frame
frame.sort_index(axis = 0)   # 로우 기준이 디폴트
frame.sort_index(axis = 1)
frame.sort_index(axis = 1, ascending=False)

frame = pd.DataFrame({'b':[4,7,-3,2], 'a':[0,1,0,1]})
frame.sort_values(by = ['a','b'])


obj = pd.Series([7,7,4,2,0,1,10])
obj.rank()
obj.rank(ascending = False, method = 'max')   # average가 디폴트
# method 인자는 순위의 동률을 처리함: max, min, first, average
frame = pd.DataFrame({'b': [4.3,7,-3,2], 'a':[3,10,1,2]})
frame
frame.rank(axis = 1)


# 기술통계 계산과 요약
# 통계에 있어 numpy와 달리 pandas는 처음부터 누락된 데이터를 제외하도록 설계되었음
df = pd.DataFrame(data = [[1.4, np.nan], [7.1,-4.5], [np.nan,np.nan], [0.75,-1.3]],
                  columns = ['one', 'two'], index = list('abcd'))
df

### 특이하다 ###
# 정렬할 때 axis = 0가 디폴트인데 로우 기준이다.
# 기술통계량을 구할 때는 axis = 0이 디폴트이지만 각 칼럼의 통계량을 반환한다.
# axis=0: 인덱스를 훑는 것을 의미한다고 생각하자.
df.sum(axis = 0)
df.sum(axis = 1)
df.mean(axis = 0, skipna = False)
df.idxmax()
df.cumsum()
df.describe()        # Summary
# quantile, mad, var, std, skew, kurt, cumsum, cummin, cummax, cumprod, diff,
             # pct_change


##### 중요 #####
# 상관관계와 공분산 p199
# 주식 가격과 시가 총액을 담고 있는 데이터
import pandas as pd
import numpy as np
from pandas_datareader import data as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')

price = pd.DataFrame({tic: data['Adj Close']
                      for tic, data in all_data.items()})
volume = pd.DataFrame({tic: data['Volume']
                       for tic, data in all_data.items()})

# 각 주식의 퍼센트 변화율 계산: AAPL, IBM, MSFT, GOOG
returns = price.pct_change()
returns.tail()                                          
returns.MSFT.corr(returns.IBM)    # 각각의 상관계수
returns.MSFT.cov(returns.IBM)     # 각각의 공분산
returns.corr()                    # 상관계수 행렬
returns.cov()

# 다른 Series나 DF와의 상관관계를 계산하고 싶다.
returns.corrwith(returns.IBM)     # DF과 Series > 각 칼럼에 대해 계산한 r을 담은 Series 반환
returns.corrwith(volume)          # DF와 DF > 각 칼럼의 이름에 대한 r 계산

# 간단한 예시
import numpy as np
import pandas as pd
exam = pd.DataFrame(data = np.random.randint(0,10, size = 16).reshape((8,2)),
                    columns = ['grade','face'],
                    index = list('abcdefjh'))
exam.corr()
exam.grade.corr(exam.face)
exam.corrwith(exam.face)

# 유일 값, 값 세기, 멤버십
obj = pd.Series(['c','a','d','a','a','b','b','c','c'])
uniques = obj.unique()            # Series를 반환한다.
uniques.sort()                    # 정렬
uniques
# 최상위 메서드: value-counts - 배열과 Series에 대해 도수를 계산하여 반환함
obj.value_counts()
pd.value_counts(obj, sort = True)
pd.value_counts(obj, sort = False)
# Series, DF의 칼럼에서 어떤 값을 골라내고 싶을 때
mask = obj.isin(['b','c'])
mask
obj[mask]
# DF의 여러 로우에 대한 히스토그램을 동시에 그리고 싶다면?
data = pd.DataFrame({'Qu1':[1,3,4,3,4], 'Qu2':[2,3,1,2,3], 'Qu3':[1,5,2,4,4]})
data
result = data.apply(pd.value_counts, axis = 0)
result
result = data.apply(pd.value_counts, axis = 0).fillna(0)
result


# 누락된 데이터 처리하기
# None값도 NaN이다.  (np.nan) // isnull()로 확인
# NA처리 메서드
# dropna, fillna, isnull, notnull

# 누락된 데이터 골라내기
from numpy import nan as NA
# Series
data = pd.Series([1,NA,3,4,NA])
data
data.dropna()
data[data.notnull()]

# DF: 기본적으로 NA가 들어있는 로우나 칼럼을 전체 삭제한다.
data = pd.DataFrame(data = [[1,3,NA], [3,2,4], [NA,21,10], [NA,NA,NA]])
data
data.dropna()
data.dropna(how = 'all')
data.dropna(thresh=3)

# 누락된 값 채우기
data2 = data.fillna(0, inplace = False)    # 기본적으로 fillna는 새로운 값을 반환함
data2
data.fillna(0, inplace = True)             # inplace 인자를 통해 기존 객체를 변경할 수 있음
data
data.fillna(data.mean())
# fillna 인자: value, method, inplace, axis, limit
# 재색인에서 사용 가능한 보간메서드는 fillna메서드에서도 사용이 가능함


# 계층적 색인 Hierarchical Indexing
data = pd.Series(np.random.randn(10),
                 index = [['a','a','a','b','b','b','c','c','d','d'],
                          [1,2,3,1,2,3,1,2,2,3]])
data
data.index
data['b':'c']
data.ix[['b','d']]
data[['b','d']]
# data['b','d'] 안됨
data[:,2]                  # 하위 계층 객체 선택
data[:,3]
# 계층적인 색인은 데이터를 재형성하고 피컷 생성같은 그룹 기반 작업에 중요함
data.unstack()
data.unstack().stack()     # 계층화
# 데이터 프레임은 두 축 모두 계층적 색인을 가질 수 있다.
frame = pd.DataFrame(np.arange(12).reshape((4,3,)),
                     index = [['a','a','b','b'], [10,20,10,20]],
                     columns = [['ohio','ohio','colorado'],['green','red','green']])

MultiIndex.from_arrays(index = [['a','a','b','b'], [10,20,10,20]],
                       columns = [['ohio','ohio','colorado'],['green','red','green']])

frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame

# 칼럼 색인
frame['ohio']
# 다음은 모두 상위칼럼-하위칼럼 순임
frame['ohio','green']
frame['ohio'][0:1]
frame['ohio']['red']

# 로우 색인
frame.ix['a']
# 다음은 모두 상위로우-하위로우 순임
frame.ix['a',10]
frame.ix['a',1]
frame.ix['a'].ix[10]


# p212 계층 순서 바꾸고 정렬하기
import pandas as pd
import numpy as np
frame.swaplevel('key1', 'key2')
# swaplevel을 사용해서 계층을 바꿀때 sortlevel을 사용해서 결과도 사전식으로 정렬하는 것이 일반적
# frame.swaplevel(0,1).sortlevel(0)
frame.swaplevel(0,1).sort_index(level=0)

frame
# 단계별 요약통계
frame.sum(level = 'key2', axis = 0)     # 로우 기준
frame.sum(level = 'color', axis = 1)    # 칼럼 기준

# DF의 칼럼사용하기
frame = pd.DataFrame({'a':range(7), 'b':range(7,0,-1), 'c':['one','one','one','two','two','two','two'],
                      'd':[0,1,2,0,1,2,3]})
frame
# set_index: 하나 이상의 칼럼을 색인으로 하는 새로운 DF 생성, 반대는 reset_index
frame2 = frame.set_index(['c','d'], drop = True)
frame2
frame2.reset_index()

# Series(.icol, irow, iget_value)

##### Chapter6 ######
# 데이터 로딩,저장 #
#####################
df = pd.read_csv('examples/ex1.csv', header = None)
df2 = pd.read_table('examples/ex1.csv',
                   sep = ',', names = ['a','b','c','d','message'], index_col = 'message')

df = pd.read_csv('B:/pydata-book-2nd-edition/examples/ex1.csv')
df2 = pd.read_table('B:/pydata-book-2nd-edition/examples/ex1.csv', sep=',')
df
df2

names = ['a','b','c','d','message']
df3 = pd.read_csv('examples/ex1.csv',
                   names = names, index_col = 'message')

parsed = pd.read_csv('examples/ex2.csv', index_col = ['key1','key2'])
parsed

norule = pd.read_csv('examples/ex3.txt', sep = '\s+')
norule

pd.read_csv('examples/ex4.csv', skiprows = [0,2,3])

# na_values: 문자열이나 리스트를 받아서 누락된 값을 처리함
# []안에 있는 녀석을 NaN으로 처리하라는 말
pd.read_csv('examples/ex5.csv', na_values = ['NULL'])
pd.read_csv('examples/ex5.csv',
            na_values = {'message':['foo','NA'], 'something':['three']})
#read_csv, read_table 인자
# sep header index_col names skiprows na_values parse_dates keep_date_col
# converters dayfirst date_parser nrows iterator
# chunksize, skip_footer, verbose, endocing, squeeze, thousands

# 텍스트 파일 조금씩 읽어오기
pd.read_csv('examples/ex6.csv', nrows = 5)
chunker = pd.read_csv('examples/ex6.csv', chunksize = 1000)
chunker    # parser.TextFileReader
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value = 0)
tot = tot.sort_values(ascending = False)
tot[:10]

# 데이터를 텍스트 형식으로 기록하기
import sys
data = pd.read_csv('examples/ex5.csv')
data
data.to_csv('examples/ex5hout.csv', na_rep = 'NULL') # 누락된 값 채워넣기
data.to_csv(sys.stdout, index = False, header = False)  # 실제파일을 기록하지 않고 여기서 보기만 할 거
data.to_csv(sys.stdout, index = False, columns = ['a','b','c'], na_rep = "NULL")
# Series에도 똑같이 to_csv 메서드를 쓸 수 있음
dates = pd.date_range('1/1/2000', periods = 7)   # 시계열 날짜 함수
ts = pd.Series(np.arange(7), index = dates)
ts.to_csv(sys.stdout)
ts.to_csv('examples/tsout.csv')
# from_csv로 Series 객체를 얻을 수 있다.
series = pd.Series.from_csv('examples/tsout.csv', parse_dates = True)
type(series)

# p232 수동으로 구분형식 처리하기 ~ p251
import csv
f = open('examples/ex7.csv')
reader = csv.reader(f) # delimiter, lineterminator, quotechar, quoting, 
# skipinitialspace, doublequote, escapechar
for line in reader:
    print(line)

lines = list(csv.reader(open('examples/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


import requests
import json
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup as Soup
url = 'https://twitter.com/search?q=pythonpandas&src=typd'
response = requests.get(url)
contents = response.content
bsobj = Soup(contents, "html.parser")
data = json.loads(response.content)
r = requests.get(url)
json_normalize(json.loads(r.text), ['data','data'])
jsonString = json.dumps(customer, indent=4)


##### Chapter7 데이터 준비하기: 다듬기, 변형, 병합 #####
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# merge 총정리
# 인자: left,right,how / on,left_on,right_on / left_index,right_index /
# sort,suffixes,copy
df1 = DataFrame({'key':['b','b','a','c','a','a','b'], 'data':range(7)})
df2 = DataFrame({'key':['a','b','d'], 'data2':range(3)})
df3 = DataFrame({'lkey':['b','b','a','c','a','a','b'], 'data1':range(7)})
df4 = DataFrame({'rkey':['a','b','d'], 'data2':range(3)})                
df1
df2
df3
df4
pd.merge(df1, df2, on = 'key')
pd.merge(df3, df4, left_on = 'lkey', right_on = 'rkey')  # 디폴트는 Inner Join이다.
pd.merge(df1, df2, how = 'outer')                        # inner, outer, left, right
pd.merge(df1, df2, on = 'key', sort = True, copy = True)

# 여러 개의 키를 병합: 칼럼 이름이 들어간 리스트를 넘김
left = DataFrame({'key1':['foo','foo','bar'], 'key2':['one','two','three'], 'lval':[1,2,3]})
right = DataFrame({'key1':['foo','foo','bar','bar'], 'key2':['one','two','one','two'], 'rval':[4,5,6,7]})
left
right
pd.merge(left, right, on = ['key1','key2'], how = 'outer')
pd.merge(left, right, on = 'key1', suffixes = ['_왼쪽', '_오른쪽'])

# 색인 merge: 머지하려는 키가 index임
left1 = DataFrame({'key1': ['a','b','a','a','b','c'], 'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index = ['a','b'])
left1
right1
pd.merge(left1, right1, left_on = 'key1', right_index = True)      # 로우색인 녀석이 오른쪽 녀석이다.
pd.merge(left1, right1, left_on = 'key1', right_index = True, how = 'outer')
# 둘다 로우 색인이면 left_index, right_index 모두 True로 설정


# 축 따라 이어붙이기: concatenate
# Series 적용
arr = np.arange(12).reshape((3,4))
np.concatenate([arr, arr], axis = 0)    # 로우가 늘어남
np.concatenate([arr, arr], axis = 1)    # 칼럼이 늘어남
s1 = Series([0,1], index = ['a','b'])
s2 = Series([2,3,4], index = ['c','d','e'])
s3 = Series([5,6], index = ['f','g'])
s4 = pd.concat([s1 * 5, s3])
pd.concat([s1, s2, s3], axis = 0)
pd.concat([s1, s2, s3], axis = 1)
s1
s4
pd.concat([s1, s4], axis = 1, join = 'inner')   # on이 아니라 join
pd.concat([s1, s4], axis = 1, join_axes = [['a','c','b','e']])
# 그런데 이렇게 붙이면 결합 전 '출처'를 알 수 없다.
pd.concat([s1, s1, s3], keys = ['s1','s2','s3'], axis = 0)    # 출처 밝히기
pd.concat([s1, s1, s3], keys = ['one','two','three'], axis = 1)

# DF 적용
df1 = DataFrame(np.arange(6).reshape((3,2)), index = ['a','b','c'], columns = ['one','two'])
df2 = DataFrame(5 + np.arange(4).reshape((2,2)), index = ['a','c'], columns = ['three','four'])
df1
df2
pd.concat(objs = [df1, df2], axis = 1, keys = ['level1','level2'], join = 'outer',
          names = ['colcol','indind'], ignore_index = True, verify_integrity = False)
pd.concat({'level1':df1, 'level2':df2}, axis = 1)               # 사전으로도 데이터를 넘길 수 있음
# 함수인자는 264 페이지


# 겹치는 데이터 합치기 p264 - p266 다시보기

# 재형성(Reshaping)과 피벗 Pivot p266
# 계층적 색인으로 재형성하기
# stack: 데이터의 칼럼을 로우로 피벗 또는 회전시킴
# unstack: 로우를 칼럼으로 피벗시킴
data = DataFrame(np.arange(6).reshape((2,3)), index = pd.Index(['ohio','colorado'], name = 'state'),
                 columns = pd.Index(['one','two','three'], name = 'number'))
data
result = data.stack()        # Series 객체 반환
result
result.unstack('state')

# 피버팅으로 데이터 나열 방식 바꾸기
data = pd.read_csv('examples/macrodata.csv')
data.head()
# 기간 인덱스: 인자 - year,month,quarter,day,hour,minute,second,tz,dtype,
# copy,freq,start,end,periods
periods = pd.PeriodIndex(year = data.year, quarter = data.quarter, name = 'date')
periods
# 데이터 축소
data = DataFrame(data = data.to_records(), columns = pd.Index(['realgdp', 'infl', 'unemp'], name = 'item'),
                 index = periods.to_timestamp(freq = 'D', how = 'end'))
# 이 'D'는 무슨 의미지?
data.head()
stacked_data = data.stack()
stacked_data.head()
ldata = data.stack().reset_index().rename(columns = {0: 'value'})  # 0을 value로 바꾼다는 뜻
ldata[:10]
pivoted = ldata.pivot(index = 'date', columns = 'item', values = 'value')
pivoted.head()

# 한 번에 2개의 칼럼을 변형하고 싶다.
ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]
pivoted = ldata.pivot(index = 'date', columns = 'item')   # 특별히 values를 지정하지 않았음
pivoted[:5]


# 데이터 변형
# 중복 제거
data = DataFrame({'k1':['one'] * 3 + ['two'] * 4, 'k2': [1,1,2,3,3,4,4]})
data
data.duplicated()
data.drop_duplicates()
data.drop_duplicates(['k1'])
# 기본적으로 처음 발견된 값을 유지하므로 마지막 발견 값을 넣고 싶으면
data.drop_duplicates(keep = 'last')

# 함수나 매핑 이용해 데이터 변형하기: DF의 칼럼, series 안의 값을 기반으로 변형
# Series의 map: 사전 류의 객체나 함수를 인자로 받음
data = DataFrame({'food': ['Bacon','pulled pork','bacon','Pastrami','corned beef','bacon',
                           'pastrami','honey ham','nova lox'],
                  'ounces':[4,3,12,6,7.5,8,3,5,6]})
meat_to_animal = {'bacon':'pig','pulled pork':'pig','pastrami':'cow','corned beef':'cow',
                  'honey ham':'pig', 'nova lox':'salmon'}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data

# 값 치환하기
series = Series([1., -999., 2., -999., -1000., 3.])
series
series.replace([-999, -1000], [np.nan, 30000])
series.replace({-999:np.nan, -1000:300000})

import pandas as pd
import numpy as np
from pandas import DataFrame
# 축 색인 이름 바꾸기
data = DataFrame(np.arange(12).reshape((3,4)), columns = ['ohio','colorado','york','mesa'],
                 index = ['one','two','three'])
data.index = data.index.map(str.upper)                  # 원래 객체 변경
data.rename(index = str.title, columns = str.upper)     # 새로운 객체 생성
# rename 메서드: 사전 형식의 객체를 이용하여 축 이름 중 일부만 변경 가능
                                                   # 바로바로 변경하려면 inplace = True
data
_ = data.rename(index = {'ONE':'nothing'}, columns = str.capitalize, inplace = True)
data

# 개별화와 양자화
# cut 함수
ages = [20,22,25,27,21,23,37,61,45,41,32]
bins = [18,25,35,60,100]
# right: 포함하는 측면을 바꾸고 싶다?  [와 ) 위치 변경
# labels: 그룹의 이름 지정
group_names = ['youth', 'youngadult', 'middleaged', 'senior']
cats = pd.cut(ages, bins, right = True, labels = group_names)
cats
cats.codes
cats.categories
cats.value_counts().sort_values(ascending = False)

# 그룹의 개수를 인자로 넘기면 균등분할한다.
data = np.random.rand(20)
pd.cut(data, 4, precision = 2)

# qcut: 표변 변위치를 기반으로 데이터를 나눔
data = np.random.randn(1000)        # 정규분포
cats = pd.qcut(data, 4)
cats
pd.value_counts(cats).sort_values(ascending = False)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])

# 특이값 찾아내고 제외하기
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.describe()
col = data[3]
col[np.abs(col) > 3]    # outlier
# ourlier가 들어 있는 모든 로우를 선택하고 싶다
data[(np.abs(data) > 3).any(1)]
data[(np.abs(data) > 3)] = np.sign(data) * 3    # 부호에 따라 -1, 1이 담긴 배열 반환
data.describe()

# 치환과 임의 샘플링
df = DataFrame(np.arange(5 * 4).reshape((5,4)))
sampler = np.random.permutation(5)
sampler
df.take(sampler)

# 표시자, 더미 변수: get_dummies
df = DataFrame({'key':['b','b','a','c','a','b'], 'data1':range(6)})
df
pd.get_dummies(df)
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix = 'keyofkey')
dummies
df[['data1']]   # DF로 불러오기
df['data1']     # Series로 불러오기
df_with_dummy = df[['data1']].join(dummies)       # join 메서드: DF 병합
df_with_dummy


# 문자열 다루기 p289
val = ' a , b ,    guido'
val.split(',')
pieces = []
for x in val.split(','):
    pieces.append(x.strip())
    print(pieces)
pieces
# 한 번에
pieces = [x.strip() for x in val.split(',')]
'::'.join(pieces)

val.index(',')  # index는 찾지 못하면 error를
val.find(':')   # find는 찾지 못하면 -1d을 반환함
val.count(',')
val.replace(',', "#")

# 정규표현식
import re
text = "foo   bar\t bac  \tqux"
re.split('\s+', text)

# re.compile: 직접 정규표현식을 컴파일하고 정규표현식 객체를 생성한다.
regex = re.compile('\s+')
regex.split(text)
regex.findall(text)

text = "Dave dave@google.com, Steve steve@gmail.com, Rob rob@yahoo.com, Ryan ryan@naver.com"
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags = re.IGNORECASE)
regex
regex.findall(text)

# search: 텍스트에서 첫 번째 이메일 주소를 찾아줌
m = regex.search(text)
m
m.start()
m.end()
text[m.start():m.end()]
print(regex.match(text))
# sub: 찾은 패턴을 주어진 문자열로 치환하여 새로운 문자열 반환
print(regex.sub('REDACTED', text))

# 이메일 주소를 찾아서 동시에 각 이메일 주소를 사용자 이름, 도메인 이름, 도메인 접미사의
# 3가지의 컴포넌트로 나눠야 한다면 각 패턴을 괄호로 묶는다.
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags = re.IGNORECASE)
# match 객체로 만든 뒤 groups 메서드를 이용한다.
m = regex.match('wes@bright.net')
m.groups()
t_list = regex.findall(text)
type(t_list)
t_list

# 벡터화된 문자열 함수 p295


#______________________________#
# Chapter8 도식화와 시각화
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn

fig = plt.figure()

# fig.add_subplot은 AxesSubplot 객체를 반환함
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
# plt.plot(randn(50).cumsum(), 'k--') # k--: 검은 점선을 그리기 위한 스타일 옵션

# 각각의 인스턴스 메서드를 호출해서 다른 빈 서브플롯에 직접 그래프를 그릴 수 있음
ax1.hist(randn(100), bins = 20, color = 'red', alpha = 0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
ax3.plot(randn(50).cumsum(), color = 'k')
plt.show()

# subplots: 특정한 배치에 맞춰 여러 개의 서브플롯을 포함하는 figure 생성
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = False, sharey = False)
fig
axes
# axes 배열은 axes[0,1] 처럼 2차원 배열로 쉽게 색인할 수 있다.

# 서브플롯 간 간격 조절하기
# 서브플롯 간 간격이 없는 그래프를 생성하고 싶다.
fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace = 0, hspace = 0,
                    left = None, right = None, bottom = None, top = None)
plt.show(fig)

# 색상, 마커, 선스타일
plt.plot(randn(30).cumsum(), color = 'g', linestyle = 'dashed', marker = 'o', label = 'Default')
plt.show(fig)

plt.plot(randn(30).cumsum(), color = 'r', linestyle = 'dashed', marker = 'o', label = 'steps-post',
         drawstyle = 'steps-post')
plt.legend(loc = 'best')
plt.show(fig)

'''
font_options = {'family':'monospace',
                'weight':'bold',
                'size':'small'}
plt.rc('font', **font_options)
'''

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(randn(1000).cumsum(), color = 'yellow', label = 'one')
ax.plot(randn(1000).cumsum(), color = 'blue', label = 'two')

ax.set_xlim([-100, 1100])
ax.set_xlabel('prac')
ax.set_xticks([250,500,750])
ax.set_xticklabels(['down','middle','up'], rotation = 30, fontsize = 'large')
ax.set_title('TITLE')
ax.text(500, 1, family = 'monospace', fontsize = 10)
ax.legend(loc = 'best')
plt.show()
plt.savefig('good.png', dpi = 400, bbox_inches = 'tight')



# 눈금, 라벨, 범례
# 현재 x축 범위
plt.xlim()     # 지정하고 싶으면 plt.xlim( [0,10] )
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(), color = 'b')

ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# 다른 눈금 이름을 사용하고 싶다면?
labels = ax.set_xticklabels(['one','two','three','four','five'], rotation = 30, fontsize = 'small')
ax.set_title('Imgoos Plot')
ax.set_xlabel('Stages')
ax.set_ylabel('Scores')
plt.show(fig)

# 범례 추가하기: 각 그래프에 label 인자를 넘긴다.
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(), color = 'k', linestyle = 'dashed', label = 'korea')
ax.plot(randn(1000).cumsum(), color = 'g', label = 'japan')
ax.plot(randn(1000).cumsum(), color = 'b', linestyle = 'dotted', label = 'china')
ax.legend(loc = 'best')
plt.show(fig)

# 주석과 그림추가: text, arrow, annotate
from datetime import datetime
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

data = pd.read_csv('examples/spx.csv')
spx = data['SPX']
spx.plot(ax = ax, color = 'k', linestyle = '-')
# data.plot(ax = ax, 나머지~) / 그 전에는 ax.plot(data = data, 나머지~)
# ax.plot(spx, color = 'k', linestyle = '-') 이거랑 같음
crisis_data = [(datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')]

ax.annotate('peak of bull market', xy = (10 / 11 / 2007, 1000))

# 오류가 나는 이유는?
for date, label in crisis_data:
    ax.annotate(label, xy = (date, spx.asof(date) + 50),
                xytext = (date, spx.asof(date) + 200),
                arrowprops = dict(facecolor = 'black', shrink = 0.05),
                horizontalalignment = 'left', verticalalignment = 'top')

ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])
ax.set_title('Important dates in 2008-2009 financial crisis')

plt.show(fig)

# 도형 추가: ax.add_patch()
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt
from numpy.random import randn

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color = 'k', alpha = 0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color = 'b', alpha = 0.3)
pgon = plt.Polygon([[0.15,0.15], [0.35,0.4], [0.2,0.6]], color = 'g', alpha = 0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
plt.show(fig)
# 그래프를 파일로 저장
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(100).cumsum(), color = 'g')
plt.show()
plt.savefig('figpath.png',
            dpi = 100, bbox_inches = 'tight', facecolor = 'w', edgecolor = 'w')
# frame, dpi, facecolor, edgecolor, format, bbox_inches

plt.rc('figure', figsize=(10,10))
# figure, axes, xtick, ytick, grid, legend

s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
plt.show()


# pandas에서 그래프 그리기 p324에서 인자 공부할 것
# label, ax, style, alpha, kind, logy, use_index, rot, xticks, yticks, xlim, ylim, grid
# Only Dataframe: subplots, sharex, sharey, figsize, title, legend, sort_columns
df = DataFrame(randn(10,4).cumsum(0), columns = ['A','B','C','D'], index = np.arange(0,100,10))
df
df.plot()
plt.show()


# 막대 그래프: kind = 'bar', 'barh'
fig, axes = plt.subplots(2,1)
data = Series(np.random.rand(16), index = list('abcdefghijklmnop'))
data.plot(kind = 'bar', ax = axes[0], color = 'k', alpha = 0.7)
data.plot(kind = 'barh', ax = axes[1], color = 'k', alpha = 0.7)
plt.show()

# 각 로우별로 그래프를 생성한다.
df = DataFrame(np.random.rand(6,4), index = ['one','two','three','four','five','six'],
               columns = pd.Index(['A','B','C','D'], name = 'Genus'))
df
df.plot(kind = 'bar')
plt.show()

df.plot(kind = 'barh', stacked = True, alpha = 0.5)
plt.show()

series = Series(np.random.randint(1,5,10), index = list('abcdefghji'))
series
series.value_counts().plot(kind = 'barh')
plt.show()

# crosstab 메서드
# 요일별 파티 숫자를 뽑고 파티 숫자 대비 팁 비율을 보여주는 막대그래프
tips = pd.read_csv('examples/tips.csv')
tips = tips.drop(['smoker','total_bill','time'], axis = 1)
tips.head()
tips.tail()
party_counts = pd.crosstab(index = tips['day'], columns = tips['size'])
party_counts
party_counts = party_counts.ix[:, 2:5]
# 총합이 1이 되도록 정규화
party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis = 0)
party_pcts
party_pcts.plot(kind = 'bar', stacked = True)
plt.show()


# 히스토그램과 밀도그래프
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins = 50)
tips['tip_pct'].plot(kind = 'kde')
plt.show()

comp1 = np.random.normal(loc = 0, scale = 1, size = 200)    # N(0, 1)
comp2 = np.random.normal(loc = 10, scale = 2, size = 200)   # N(10, 4)
values = Series(np.concatenate([comp1, comp2], axis = 0))   # pd.concat
values.head()
values.hist(bins = 100, alpha= 0.3, color = 'k', normed = True)
values.plot(kind = 'kde', color = 'k', linestyle = 'dashed')
plt.show()


# 산포도
macro = pd.read_csv('examples/macrodata.csv')
macro.head()
data = macro[['cpi','m1','tbilrate','unemp']]
trans_data = np.log(data).diff().dropna()
trans_data.tail()

plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes')
plt.show()

pd.scatter_matrix(trans_data, diagonal = 'kde', color = 'b', alpha = 0.3)
plt.show()


# p335 아이티 지진 데이터 시각화




#______________________________________
###### Chapter9 데이터 수집과 그룹연산 #####
# 분리-적용-결합 : split-apply-combine

# groupby
df = DataFrame({'key1':['a','a','b','b','a'],
                'key2':['one','two','one','two','one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})
df

# 그룹으로 묶을 축: DF의 칼럼이름을 지칭하는 값
grouped = df['data1'].groupby(df['key1'])
grouped.mean()

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means       # 결과값은 계층적 색인을 가지는 Series
means.ix['a']['one']
means.unstack()

# 그룹으로 묶을 축: 같은 길이의 리스트나 배열
states = np.array(['ohio','cali','cali','ohio','ohio'])
years = np.array([2005,2005,2006,2005,2006])
df['data1'].groupby([states, years]).mean()

# 같은 Df내에선 더 간결하게
df.groupby(df['key1']).mean()   # df.groupby('key1').mean()

# size: 그룹의 크기를 담고 있는 Series 반환
df.groupby(['key1','key2']).size()

# 그룹 간 순회하기: 그룹 이름과 그에 따른 데이터 묶음을 튜플로 반환함
for name, group in df.groupby('key1'):
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1','key2']):
    print((k1, k2))
    print(group)
    
for names, group in df.groupby(['key1','key2']):
    print(names)
    print(group)

# 원하는 데이터만 골라내기
list(df.groupby('key1'))
dict(list(df.groupby('key1')))
pieces = dict(list(df.groupby('key1')))
pieces['b']
df[df.key1 == 'b']

dict(list(df.groupby(df.dtypes, axis = 1)))

# 칼럼 또는 칼럼의 일부 선택하기
df['data2'].groupby([df['key1'],df['key2']]).mean()

# 그룹으로 묶을 축: 그룹 이름에 대응하는 사전이나 Series 객체
people = DataFrame(np.random.randn(5,5),
                   columns = ['a','b','c','d','e'],
                   index = ['joe','steve','wes','jim','travis'])
people
people.ix[2:3, ['b','c']] = np.nan
people

mapping = {'a':'red', 'b':'red', 'c':'blue', 'd':'blue', 'e':'red', 'f':'orange'}
by_column = people.groupby(by = mapping, axis = 1)
by_column.sum()

# 그룹으로 묶을 축: 함수
people.groupby(len).sum()

# 섞어쓰기
key_list = ['one','one','one','two','two']
people.groupby([len, key_list]).sum()

# p355 색인 단계로 묶기: 추후 다시 공부



# 데이터수집
df
df['data1'].groupby(df['key1']).quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()
df['data1'].groupby(df['key1']).agg(peak_to_peak)
df['data1'].groupby(df['key1']).describe()

# 최적화된 groupby메서드: sum, count, mean, median, std, var, min, max, prod, fitst,
# last
tips = pd.read_csv('examples/tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

# 칼럼에 여러 함수 적용하기
cnt = 0
sex = []
while cnt < len(tips):
    for j in ['Female', 'Male']:
        sex.append(j)
        cnt += 1

tips['sex'] = sex
tips.head()

grouped = tips.groupby(['sex', 'smoker'])
grouped.mean()

# 하나만 계산하고 싶다.
grouped_pct = grouped['tip_pct']
grouped_pct.mean()
grouped_pct.agg('mean')     # 같음


# 함수별로 계산: agg, aggregate
grouped_pct.agg(['mean', 'std'])
functions = ['count','mean','std','max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result

tips.groupby(['sex','smoker'], as_index = True).mean()      # 디폴트: index가 계층적 색인
tips.groupby(['sex','smoker'], as_index = False).mean()


# 그룹별 연산과 변형: transform, apply
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from numpy.random import randn
df
k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means
pd.merge(df, k1_means, left_on = 'key1', right_index = True)

# 매끄럽게 해보자
people = DataFrame(np.random.randn(5,5),
                   columns = ['a','b','c','d','e'],
                   index = ['joe','steve','wes','jim','travis'])
people.ix[2:3, ['b','c']] = np.nan
people
key = ['one','two','one','two','one']
people.groupby(key).mean()
# DF의 형태를 유지하면서 각 그룹(칼럼)에 np.mean이란 함수를 적용한다: transform
people.groupby(key).transform(np.mean)

def demean(arr):
    return arr - arr.mean()

demeaned = people.groupby(key).transform(demean)
demeaned
demeaned.groupby(key).mean()

# apply: 분리-적용-병합; apply에 인자로 들어갈 함수는 lambda로 지정을 해야 한다.
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by = column)[-n:]
top(tips, n = 6)
# tips.sort_values(by = 'tip_pct', ascending = False).head(6)
tips.groupby('smoker', group_keys = True).apply(top)
tips.groupby('smoker', group_keys = False).apply(top)
tips.groupby('smoker', as_index = True).apply(top)
tips.groupby('smoker', as_index = False).apply(top)
tips.groupby('smoker').apply(top, n=1, column = 'total_bill')

tips.groupby('smoker')['tip_pct'].describe().unstack()

# p369 - p373 이후에 다시 보기

# p373 예제: 그룹 가중평균과 상관관계
df = DataFrame({'category':['a','a','a','a','b','b','b','b'],
                'data':np.random.randn(8), 'weights':np.random.randn(8)})
df
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights = g['weights'])
grouped.apply(get_wavg)

###########################################
# 주식 - S&P 500지수 (SPX) 상관관계 알아보기
close_px = pd.read_csv('examples/stock_px.csv',
                       parse_dates = True, index_col = 0)
close_px.info()
close_px[-4:]

# 퍼센트 변화율: 일일변화율
rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])   # 상관관계 Series 반환
rets.head()

# groupby에서 함수로 묶었다: 함수는 색인 값 하나하나마다 한번 씩 호출된다.
# 각 연도마다 lambda x를 적용할 채비를 마친 것이다.
by_year = rets.groupby(lambda x: x.year)
# 준비된 연도 행마다 spx_corr함수를 apply 한다.
by_year.apply(spx_corr)

# 애플과 MS 주가 수익의 연간 상관관계
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


###########################
# 피벗테이블과 교차일람표 #
tips.head()
tips.pivot_table(values = 'total_bill', index = 'smoker', columns = 'sex')
tips.pivot_table(index = ['sex', 'smoker'])

temp = tips.pivot_table(values = ['tip_pct', 'size'], margins = True,
                        index = ['sex', 'day'], columns = 'smoker')
temp
temp.ix['Female'].ix['Fri']['smoker' == 'All']

# 다른 집계함수를 사용하고 싶다면?  aggfunc = count, len
tips.pivot_table(values = 'tip_pct', index = ['sex', 'smoker'], columns = 'day',
                 aggfunc = len, margins = True, fill_value = 0)

# 교차 일람표: 그룹빈도를 계산하기 위한 피벗의 특수한 경우
# pandas의 crosstab을 쓰는 것이 편리하다.
data = DataFrame({'Sample':[1,2,3,4,5,6,7,8,9,10],
                 'Gender':['f','m','m','f','f','m','m','m','f','f'],
                 'Handed':['r','r','l','l','r','l','l','r','r','r']})
pd.crosstab(index = data.Gender, columns = data.Handed, margins = True, dropna = True)
pd.crosstab([tips.time, tips.day], tips.smoker, margins = True)


# p381 예제: 2012년 연방 선관위 데이터베이스
fec = pd.read_csv('C:/Users/YY/Documents/Winter Project/P00000001-ALL.csv', encoding = "ISO-8859-1",
                  low_memory = False)
fec.info()

fec.ix[123456]

# 모든 정당의 후보 목록을 얻는다.
unique_cands = fec.cand_nm.unique()
unique_cands

# 소속정당을 사전을 이용하여 표기
parties = {'Bachman, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
parties

fec['party'] = fec.cand_nm.map(parties)
fec['party'].value_counts()

# 분석을 단순화하기 위해 기부금이 양수인 데이터만 골라냄
fec = fec[fec.contb_receipt_amt > 0]

# isin 함수 복습: DF에 쓰는 거
df = DataFrame({'us':[1,2,3], 'ja':[3,10,2]})
df.us.isin([3])

# 양대 후보 오바마, 롬니의 기부금 정보만 따로 추려낸다.
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]


# 직장 및 피고용별 기부 통계
fec.contbr_occupation.value_counts()[:10]

# 다르게 써져 있는 직업 정보를 하나의 직업으로 매핑(통일)
occ_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
    'INFORMATION REQUESTED' : 'NOT PROVIDED',
    'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
    'C.E.O.' : 'CEO'}

# 고용주에도 같은 코드를 적용한다.
emp_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
    'INFORMATION REQUESTED' : 'NOT PROVIDED',
    'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
    'C.E.O.' : 'CEO'}

# 매핑 정보가 없는 직업은 키를 그대로 반환한다.
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)
k = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(k)

fec.contbr_occupation[:10]
fec.contbr_employer[:10]

# 정당과 직업별로 데이터를 집계한 다음, 최소 2백만 $ 이상 기부한 직업만 골라낸다.
by_occupation = fec.pivot_table(values = 'contb_receipt_amt',
                                index = 'contbr_occupation',
                                columns = 'party', aggfunc = 'sum')
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm
over_2mm.plot(kind = 'barh')
import matplotlib.pyplot as plt
plt.show()

# 오바마와 롬니 별로 가장 많은 금액을 기부한 직군은?
def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending = False)[-n:]

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n = 7)    
grouped.apply(get_top_amounts, 'contbr_employer', n = 7) 


# 기부금액
# 기부 규모 별로 버킷을 만들어 기부자 수 분할
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels

# 기부자이름과 버킷이름으로 그룹을 묶어 기부 규모의 금액에 따른 히스토그램 그리기
grouped = fec_mrbo.groupby(['cand_nm', labels])
gruoped.size()
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
bucket_sums
normed_sums = bucket_sums.div(bucket_sums.sum(axis = 1), axis = 0)
normed_sums
normed_sums[:-2].plot(kind = 'barh', stacked = True)
plt.show()



#####################
# Chapter 10 시계열 #
#####################

# p394 날짜, 시간 자료형, 도구
from datetime import datetime, timedelta
from dateutil.parser import parse
now = datetime.now()
now.year
now.day
now.minute

timediff = datetime(2018,2,15,8,10) - datetime(2018,1,14,21,9)
timediff

start = datetime(2010, 10, 1)
start + timedelta(10)

# datetime객체 / Timestemp객체 >> << 문자열: strftime(날짜 -> 문자) /// strptime(문자 -> 날짜)
stamp = datetime(2011,1,3)      # str(stamp)
stamp.strftime('%Y %m %d')

example = ['7/6/2011', '5/20/2013']
[datetime.strptime(x, '%m/%d/%Y') for x in example]

what = '2017/8/20'
datetime.strptime(what, '%Y/%m/%d')

stamp = datetime(2000,1,1)
stamp.strftime('%m %d')

# 흔히 쓰는 날짜 형식을 파싱하기 위해선
from dateutil.parser import parse
parse('2011-01-01')
parse('Jan 01 2019 10:10 PM', dayfirst = False)

datestr = ['2010/1/1', '2011/7/6', '2013/3/12']
pd.to_datetime(datestr)

# %Y %y %m %d %H %I %M %S %w %U %W %z %F %D

# 시계열 기초: TimeSeries 객체 = Series의 하위 클래스이다!
from datetime import datetime as dt
dates = [dt(2018,2,1), dt(2018,2,5), dt(2018,2,9)]
ts = pd.Series(np.random.randn(3), index = dates)
type(ts)
type(ts.index)
ts.index[2]

ts
ts['2018/2/5']

long_ts = pd.Series(np.random.randn(1000), index = pd.date_range(start = '2015/1/1', periods = 1000))
long_ts[:10]

long_ts['2016'][:10]
long_ts[datetime(2015,3,3):datetime(2015,3,5)]

dates = pd.date_range(start = '2015/1/1', periods = 100, freq = 'W-Wed')
dates[:3]
long_df = pd.DataFrame(np.random.randn(100,4), index = dates, columns = list('abcd'))
long_df.ix['2015-2']


# 날짜 범위, 빈도
pd.date_range(start = '2018/1/1', end = '2018/1/11')
pd.date_range(start = '2017/1/1', freq = 'M', periods = 10, normalize = False)  # normalize: 자정에 맞추기
pd.date_range(start = '2010/1/1', freq = '4h', periods = 10)
# 월별주차: 매월 셋째 주 금요일
pd.date_range(start = '1/1/2012', freq = 'WOM-3FRI', periods = 3)

# 데이터 시프트
ts = Series(data = np.random.randn(4), index = pd.date_range('2018/1/1', freq = '4D', periods = 4))
ts
ts.shift(2)
ts.shift(2, freq = 'D')

# 시간대 다루기(p412: 나중에)

# 기간: Period 클래스
p = pd.Period(2008, freq = 'A-DEC')
p
p + 5
pd.Period(2009, freq = 'A-DEC') - p

values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq = 'Q-DEC')
index

p = pd.Period('2007', freq = 'A-DEC')
p.asfreq('M', how = 'end')

p = pd.Period('2011', freq = 'A-Jun')
p.asfreq('M', how = 'start')
# A-는 Annual 연 빈도를 뜻함

# 기간객체인 Period나 PeriodIndex는 asfreq을 통해 빈도를 변환할 수 있음
ts
ts.asfreq('D', how = 'start')
ts.resample('D').mean()

# 분기 빈도: Q-
p = pd.Period('2012Q4', freq = 'Q-FEB')   # 회계연도 마감이 2월이다.
p.asfreq('D', how = 'start')
p.asfreq('M', how = 'end')

rng = pd.period_range('2011Q3', '2012Q4', freq = 'Q-JAN')
rng
ts = Series(np.arange(len(rng)), index=rng)
ts


# 타임스탬프와 기간 서로 변환하기
# 타임스탬프로 색인된 Series와 DataFrame 객체는 to_periods 메서드를 사용해서 Period로 변환 가능
rng = pd.date_range('2011/1/1', freq = 'D', periods = 5)
ts = Series(np.random.randn(5), index=rng)
pts = ts.to_period()                    # 기간으로 변환
type(pts)
pts
ts2 = pts.to_timestamp(how = 'end')     # 타임스탬프로 변환
type(ts2)
ts2


#p425 리샘플링과 빈도변환: pandas객체의 resample
rng = pd.date_range('1/1/2000', periods = 100, freq = 'D')
ts = Series(np.random.randn(len(rng)), index = rng)

ts.resample('D', axis = 0, closed = 'left', label = 'right', kind = 'period')[:10].ffill().mean()

# 메서드 인자 설명
# 'D', 'M', '5Min', Second(15): 원하는 리샘플링 빈도를 가리키는 문자열, 오프셋
# axis = 0: 이게 디폴트, 리샘플링을 수행할 축
# 보간:'ffill', 'bfill' - 뒤에 추가하는 것임!!
# closed = 'right': 다운샘플링 시 어느 쪽을 포함할지 결정
# label = 'right', 다운샘플링 시 집계된 결과의 라벨 결정
# kind = None, 'period', 'timestamp': 집계방법 구분 - 기간별 or 타임스탬프별
# convention = 'end', 'start': 기간을 리샘플링 할 때 하위빈도에서 상위빈도로 변환할 때의 방식
# 'mean', 'np.max', 'ohlc' 등 집계된 값을 생성하는 함수를 끝에 붙인다.


# 다운 샘플링
rng = pd.date_range('1/1/2000', periods = 12, freq = 'T')
ts = Series(np.arange(12), index=rng)
ts[:3]

ts.resample('5Min', closed = 'left', label = 'left').sum()     # 데이터를 5분 단위로 묶어서 각 그룹의 합을 집계함
ts.resample('5Min', closed = 'right', label = 'right').sum()


# OHLC 리샘플링
ts.resample('5min').ohlc()

# Groupby 활용
rng = pd.date_range('1/1/2000', periods = 365, freq = 'D')
ts = Series(np.arange(365), index=rng)
ts.groupby(lambda x: x.month).mean()


# 시계열 그래프
fout = open('examples/thisis.txt', 'wt')
fout.write('wow')
fout.close()

close_px_all = pd.read_csv('examples/stock_px.csv',
                           parse_dates = True, index_col = 0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px[:10]
close_px = close_px.resample('B', fill_method = 'ffill')
close_px = close_px.ix['2003':'2011']
close_px.info()

close_px.ix['2009'].plot()
plt.show()
close_px['AAPL'].ix['01-2011':'03-2011'].plot()
plt.show()
close_px['AAPL'].resample('Q-DEC').ffill().ix['2009':].plot()
plt.show()


# 이동창 기능
close_px['AAPL'].plot()
close_px.AAPL.rolling(window = 250, center = False).mean().plot()
plt.show()

close_px.AAPL.rolling(window = 250, min_periods = 10).std().plot()
plt.show()

# 지수 가중 함수
aapl_px = close_px.AAPL['2005':'2009']
aapl_px.plot()
aapl_px.ewm(span = 60, min_periods = 0, adjust = True, ignore_na = False).mean().plot(style = 'k--')
plt.show()


# 이진 이동창 함수
spx_px = close_px_all['SPX']
spx_rets = spx_px / spx_px.shift(1) - 1     # 시계열 내의 퍼센트 변화 계산
returns = close_px.pct_change()

corr = returns.rolling(window = 125, min_periods = 100).corr(other = spx_rets)
corr.plot()
plt.show()


#############################################
# Chapter 11 금융, 경제 데이터 애플리케이션 #
#############################################




