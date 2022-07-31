# -- coding: utf-8 --
# from mlab.releases import latest_release as mlab
# #mlab.plot([1,2,3],'-o')
# #mlab.path(mlab.path(),'D:/mproject/nricp-master/icp')
# mlab.path(mlab.path(),'D:/mproject/nricp-master/icp')

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals
import numpy as np
from functools import wraps
from time import time
def mainFunction():
    a = int(1)
    b = int(2)
    #result = mlab.get_sum(a, b) # 调用matlab的函数
    #print (result)
def test_icp():
    # M = np.array([[1, 5, 0],
    # [1.6, 5.5, 0],
    # [2.3, 5.8, 0],
    # [3, 6, 0],
    # [3.5, 6.5, 0],
    # [4, 7.5, 0],
    # [4.2, 8, 0],
    # [4.5, 8.5, 0]
    # ])
    # D = np.array([[4, 1.5, 0],
    # [4.5, 2, 0],
    # [4.9, 2.3, 0],
    # [5.5, 2.5, 0],
    # [6, 3, 0],
    # [6.3, 3.6, 0],
    # [6.5, 4.2, 0],
    # [6.8, 4.5, 0],
    # [7.2, 5, 0],
    # [7.5, 5.2, 0],
    # [8, 5.4, 0],
    # [8.6, 5.9, 0],
    # [9.2, 6.2, 0],
    # [10, 6.5, 0]
    # ])
    M = [[1, 5, 0],
    [1.6, 5.5, 0],
    [2.3, 5.8, 0],
    [3, 6, 0],
    [3.5, 6.5, 0],
    [4, 7.5, 0],
    [4.2, 8, 0],
    [4.5, 8.5, 0]
    ]
    D = [[4, 1.5, 0],
    [4.5, 2, 0],
    [4.9, 2.3, 0],
    [5.5, 2.5, 0],
    [6, 3, 0],
    [6.3, 3.6, 0],
    [6.5, 4.2, 0],
    [6.8, 4.5, 0],
    [7.2, 5, 0],
    [7.5, 5.2, 0],
    [8, 5.4, 0],
    [8.6, 5.9, 0],
    [9.2, 6.2, 0],
    [10, 6.5, 0]
    ]

    # m,n= mlab.python_icp(M, D)
    # print m
    # U,S,V = mlab.svd([[1,2],[1,3]],nout =3)
    # print  U,S,V

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        t1 = time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

# =======================================
import matlab_wrapper
def test_matlab_wrapper():
    #matlab = matlab_wrapper.MatlabSession()
    matlab = matlab_wrapper.MatlabSession(
        options='-nojvm',
        buffer_size=1000,
    )

    #k =matlab.get('y')    # matlab.put('M', 2.)
    # matlab.put('D', 3.)
    # matlab.eval('get_sum')
    # matlab.put('p1', 21.)
    # matlab.put('p2', 21.)
    # matlab.eval('test_matlabwrapper1')
    # y1 = matlab.get('y1')
    #
    # print("And the winner is:", y1)
    matlab.eval('clc;clear;')
    matlab.put('x', 2.)
    matlab.eval('m_script9')
    y = matlab.get('p')
    print("And the winner is:", y)
    matlab.eval("c = {1, 2, 3; 'text', eye(2,3), {11; 22; 33}}")

    print(matlab.get('c'))


    matlab.put('a', np.array([[1,'asdf'], [3, np.array([1,2,3],dtype='O')]], dtype='O'))
    matlab.eval('a')
    print(matlab.get('a'))
    print(matlab.output_buffer)






    #print("And the winner is:", k)
@fn_timer
def test_matlab_wrapper2():
    #matlab = matlab_wrapper.MatlabSession()

    M = [[1, 5, 0],
         [1.6, 5.5, 0],
         [2.3, 5.8, 0],
         [3, 6, 0],
         [3.5, 6.5, 0],
         [4, 7.5, 0],
         [4.2, 8, 0],
         [4.5, 8.5, 0]
         ]
    D = [[4, 1.5, 0],
         [4.5, 2, 0],
         [4.9, 2.3, 0],
         [5.5, 2.5, 0],
         [6, 3, 0],
         [6.3, 3.6, 0],
         [6.5, 4.2, 0],
         [6.8, 4.5, 0],
         [7.2, 5, 0],
         [7.5, 5.2, 0],
         [8, 5.4, 0],
         [8.6, 5.9, 0],
         [9.2, 6.2, 0],
         [10, 6.5, 0]
         ]
    matlab.put('M',M)
    matlab.put('N',D)
    matlab.eval('python_icp7(M,D)')
    #print(matlab.get('Ricp'))





    #print("And the winner is:", k)

if __name__ == '__main__':
    matlab = matlab_wrapper.MatlabSession(
        options='-nojvm',
        buffer_size=1000,
    )
    M = [[1, 5, 0],
         [1.6, 5.5, 0],
         [2.3, 5.8, 0],
         [3, 6, 0],
         [3.5, 6.5, 0],
         [4, 7.5, 0],
         [4.2, 8, 0],
         [4.5, 8.5, 0]
         ]
    D = [[4, 1.5, 0],
         [4.5, 2, 0],
         [4.9, 2.3, 0],
         [5.5, 2.5, 0],
         [6, 3, 0],
         [6.3, 3.6, 0],
         [6.5, 4.2, 0],
         [6.8, 4.5, 0],
         [7.2, 5, 0],
         [7.5, 5.2, 0],
         [8, 5.4, 0],
         [8.6, 5.9, 0],
         [9.2, 6.2, 0],
         [10, 6.5, 0]
         ]
    matlab.put('M', M)
    matlab.put('D', D)
    #matlab.put('x', 3.)
    #matlab.eval('test1')
    matlab.eval('[Ricp Ticp ER t] = python_icp7(M,D);pause;')
    #print(matlab.output_buffer)
    #K =matlab.get('M')
    #print(type(K))
    #print(K.shape)
    #y = matlab.get('y')
    print( matlab.get('Ricp'))
    print(matlab.get('Ticp'))
    print(matlab.get('ER'))
    print(matlab.get('t'))

    #print("And the winner is:", y)

    # matlab = matlab_wrapper.MatlabSession(
    #     options='-nojvm',
    #     buffer_size=1000,
    # )
    #mainFunction()
    #test_icp()
    #test_matlab_wrapper()

