/*
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

template <typename T> T& larger(T& a, T& b) {
	return a > b ? a : b;
}

template <typename T> class Array {
private:
	T * elements;								//타입 T의 배열
	size_t count;								//배열 원소의 개수

public:
	explicit Array(size_t arraySize);			//생성자
	Array(const Array other);					//복제 생성자
	Array(Array&& other);						//이동 생성자
	virtual ~Array();							//소멸자
	T& operator[](size_t index);				//첨자 연산자
	const T& operator[](size_t index) const;	//첨자 연산자 - 상수 배열
	Array& operator=(const Array& rhs);			//할당 연산자
	Array& opeator = (Array&& rhs);				//이동 할당 연산자
	size_t size() { return count; }				//count 변수를 위한 접근자
};

template <typename T>
Array<T>::Array(size_t arraySize) try : elements{ new T[arraySize] }, count{ arraySize } {}
catch (const std::exception& e) {
	st::cerr << "Array 생성자에서 메모리 할당 실패." << std::endl;
	rethrow e;
}

template <typename T>
inline Array<T>::Array(const Array& other)
try :elements{ new T[array.count] }, count{ array.count }
{
	for (size_t i{}; i < count; ++i)
		elements[i] = array.elements[i];
}
catch (bad_alloc&) {
	cerr << "Array 객체 복제를 위한 메모리 할당에 실패했습니다." << endl;
}

class T {
public:
	T();
	T(const T& t);
	~T();
	T& operator=(const T& t);
};



int main() {
cout << larger(2.1, 3.5);
Array<int> data{ 40 };
template<typename T> using ptr = shared_ptr<T>;

using std::string;
ptr<string>;

}



///////////////////////////////////////////
// 반복자 사용하기
#include <numeric>
#include <iostream>
#include <iterator>
int main() {
	double data[]{ 2.5, 4.5, 6.5, 5.5, 8.5 };
	std::cout << "elements:\n";
	for (auto iter = std::begin(data); iter != std::end(data); ++iter)
		std::cout << *iter << " ";
	auto total = std::accumulate(std::begin(data), std::end(data), 0.0);
	std::cout << "\nsum : " << total << std::endl;
}


#include <numeric>
#include <iostream>
#include <iterator>
using namespace std;

int main() {
	cout << "\nThe sum of the values you entered is "
		<< accumulate(istream_iterator<double>(cin), istream_iterator<double>(), 0.0)
		<< endl;
}


#include <numeric>
#include <iostream>
#include <iterator>
#include <memory>
using namespace std;
int main() {
	int data[]{ 10,20,30,40,50,60 };
	auto iter = begin(data);
	advance(iter, 3);
	cout << "4th element : " << *iter << endl;
	cout << "the number of elements : " << distance(begin(data), end(data)) << endl;
	iter = begin(data);
	auto fourth = next(iter, 3);
	cout << "1st: " << *iter << ", and 4th: " << *fourth << endl;
	iter = end(data);
	cout << "4th: " << *prev(iter, 3) << endl;

	// unique_ptr
	unique_ptr<string> pname1{ new string{"Algernon"} };
	unique_ptr<string> pname2{ new string( "Algernon" ) };
	auto pname3 = make_unique<string>("Algernon");
	auto pstr = make_unique<string>(6, '*');
	// ? cout << *pname3 << endl;
	size_t len( 10 );
	unique_ptr<int[]> pnumbers{ new int[len] };
	for (size_t i{}; i < len; ++i)
		pnumbers[i] = i * i;
	for (size_t i{}; i < len; ++i)
		cout << pnumbers[i] << ' ';
	auto unique_p = make_unique<string>(6, '*');
	// string pstr2 { unique_p.get() };
	pname1.reset();
	pname1.reset(new string{ "Fred" });
	cout << endl << pname1.get() << endl;
	auto up_name = make_unique<string>("Algernon");
	unique_ptr<string> up_new_name{ up_name.release() };
	auto pn1 = make_unique<string>("Jack");
	auto pn2 = make_unique<string>("Jill");
	pn1.swap(pn2);
	if (up_new_name)
		cout << up_new_name.get() << endl;
	if (!up_name)
		cout << "null" << endl;

	//shared_ptr
	shared_ptr<double> pdata1{ new double{999.0} };
	*pdata1 = 8888.0;
	cout << *pdata1 << endl;
	*pdata1 = 9999.0;
	cout << *pdata1 << endl;
	auto pdata = make_shared<double>( 999.9 );
	shared_ptr<double> pdata2{ pdata };
	pdata2 = pdata;
	cout << *pdata << endl;
	auto pvalue = pdata.get();
	pname1 = nullptr;
	pname2.reset();
	pname3.reset(new string{ "Lane Austen" });
	cout << (pname1 == pname2 && pname1 != nullptr) << endl;
	cout << (pname1 && pname1 == pname2) << endl;
	auto pname4 = make_shared<string>("Charles Dickens");
	cout << "unique: " << pname4.unique() << endl;
	cout << "number: " << pname4.use_count() << endl;

	//weak_ptr
	auto pData = make_shared<int>(0);
	weak_ptr<int> pwData{ pData };
	weak_ptr<int>pwData2{ pwData };
	if (pwData.expired())
		cout << "Doesn't exist" << endl;
	shared_ptr<int> pNew{ pwData.lock() };
	cout << pNew << endl;
}


#include <iostream>
#include <functional>
using namespace std;

class Volume {
public:
	double operator()(double x, double y, double z) { return x * y*z; }
	double operator()(const Volume& volume) { return 0.0; }
};

template <typename ForwardIter, typename F>
void change(ForwardIter first, ForwardIter last, F fun) {
	for (auto iter = first; iter != last; ++iter)
		*iter = fun(*iter);
}

int main() {
	Volume volume;
	double room{ volume(1,2,3.4) };
	cout << room << endl;
	auto cube = [](double value) {return value * value*value; };
	double x{ 2.5 };
	cout << x << " cubed is: " << cube(x) << endl;

	int data[]{ 1,2,3,4 };
	change(begin(data), end(data), [](int value) {return value * value; });
	for (auto iter = begin(data); iter != end(data); ++iter)
		cout << *iter << ' ';
	cout << endl;
	function<double(double)>op{ [](double value) {return value * value*value; } };
	op = [](double value) {return value * value; };
}
*/

#include <iostream>
#include <algorithm>
#include <iterator>
#include <functional>
using namespace std;

class Root {
public:
	double operator()(double x) { return sqrt(x); };
};

int main() {
	double data[]{ 1.5,2.5,3.4,4.5,5.5 };
	Root root;
	cout << "Square roots are: " << endl;
	transform(begin(data), end(data), ostream_iterator<double>(cout, " "), root);
	cout << "\n\nCubes are: " << endl;
	transform(begin(data), end(data), ostream_iterator<double>(cout, " "), [](double x) {return x * x*x; });
	function<double(double)> op{ [](double x) {return x * x; } };
	cout << "\n\nSquares are: " << endl;
	transform(begin(data), end(data), ostream_iterator<double>(cout, " "), op);
	cout << "\n\n4th powers are: " << endl;
	transform(begin(data), end(data), ostream_iterator<double>(cout, " "), [&op](double x) {return op(x)*op(x); });
	cout << endl;
}