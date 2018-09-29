if __name__ == "__main__":
    lis = ['Imgoos', 'Yamgoos']
    lis.append('Yomgoos')
    print(lis)
    lis+='Doonggoos'
    del lis[2]
    lis.append('Doonggoos')
    print(lis)
    help(list.remove)
    lis.index('s')
    Imgoos = 'Imgoos', 'Yamgoos', 'Yomgoos'
    Imgoos
    a,b,c=Imgoos
    print(a, b, c)
    lot = [('a','bbd'),('c',1), ('e', {1:True, 'x':19.2})]
    dict(lot)
    help(dict)
    help(set)
    help(zip)
    number_thing = (number for number in range(1, 6))
    for number in number_thing:
        print(number)
    number_list = list(number_thing)
    try_again = list(number_thing)
    print(number_list)
    print(try_again)

    def echo(anything):
        'echo returns Imgoos'
        print(locals())
        return anything + ' is Imgoos'
    echo('1')
    help(echo)
    help(echo.__doc__)
    echo('Yamgoos')
    print(5)
    locals()
    globals()

def weather():
    """ Return Imgoos weather"""
    from random import choice
    Doong = ['rain','Imgoos','Yamgoos']
    return choice(Doong)
