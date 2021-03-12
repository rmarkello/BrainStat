import numpy as np
import pytest
from scipy.io import loadmat
import brainstat.tests.surstat_wrap as sw
from brainstat.stats.SLM import SLM, f_test
from brainstat.stats._t_test import t_test
from brainstat.stats.terms import Term


sw.matlab_init_surfstat()


def dummy_test(A, B):

    try:
        # wrap matlab functions
        Wrapped_slm = sw.matlab_F(A, B)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # convert input dicts into python objects for python function
    SLM1 = SLM(Term(1), Term(1))
    SLM2 = SLM(Term(1), Term(2))

    for key in A.keys():
        setattr(SLM1, key, A[key])
        setattr(SLM2, key, B[key])

    # run python function
    Python_SLM = f_test(SLM1, SLM2)

    # compare matlab-python outputs
    testout_F = []
    for key in Wrapped_slm:
        testout_F.append(np.allclose(Wrapped_slm[key],
                                     getattr(Python_SLM, key),
                                     rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_F)


def test_01():
    # slm1['coef'] is 2D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D array of integers
    n = 5
    p = 6
    k = 2
    v = 1

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n, p))
    slm1['df'] = (n-1)
    slm1['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm1['coef'] = rng.integers(100, size=(p, v))

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = n
    slm2['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm2['coef'] = rng.integers(100, size=(p, v))

    dummy_test(slm1, slm2)


def test_02():
    # slm1['coef'] is 2D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D array of integers
    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = np.random.randint(3, 100)
    v = np.random.randint(3, 100)

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n, p))
    slm1['df'] = (n-1)
    slm1['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm1['coef'] = rng.integers(100, size=(p, v))

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = n
    slm2['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm2['coef'] = rng.integers(100, size=(p, v))

    dummy_test(slm1, slm2)


def test_03():
    # slm1['coef'] is 2D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random arrays

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = np.random.randint(3, 100)
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = np.random.rand(n, p)
    slm1['df'] = n
    slm1['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm1['coef'] = np.random.rand(p, v)

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm2['coef'] = np.random.rand(p, v)

    dummy_test(slm1, slm2)


def test_04():
    # k= 3
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers

    rng = np.random.default_rng()

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 3
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n, p))
    slm1['df'] = p
    slm1['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm1['coef'] = np.ones((p, v, k)) + 2

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p+1
    slm2['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm2['coef'] = np.ones((p, v, k))

    dummy_test(slm1, slm2)


def test_05():
    # k = 2
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers

    rng = np.random.default_rng()

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 2
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n, p))
    slm1['df'] = p+1
    slm1['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm1['coef'] = np.ones((p, v, k)) + 2

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm2['coef'] = np.ones((p, v, k))

    dummy_test(slm1, slm2)


def test_06():
    # k = 1
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers

    rng = np.random.default_rng()

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 2
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n, p))
    slm1['df'] = n
    slm1['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm1['coef'] = rng.integers(1, 100, size=(p, v, k))

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = rng.integers(1, 100, size=(int(k*(k+1)/2), v))
    slm2['coef'] = rng.integers(1, 100, size=(p, v, k))

    dummy_test(slm1, slm2)


def test_07():
    # k= 3
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 3
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = np.random.rand(n, p)
    slm1['df'] = p
    slm1['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm1['coef'] = np.random.rand(p, v, k)

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p+1
    slm2['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm2['coef'] = np.random.rand(p, v, k)

    dummy_test(slm1, slm2)


def test_08():
    # k = 2
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 2
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = np.random.rand(n, p)
    slm1['df'] = p+1
    slm1['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm1['coef'] = np.random.rand(p, v, k)

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm2['coef'] = np.random.rand(p, v, k)

    dummy_test(slm1, slm2)


def test_09():
    # k = 1
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array

    n = np.random.randint(3, 100)
    p = np.random.randint(3, 100)
    k = 1
    v = np.random.randint(3, 100)

    slm1 = {}
    slm1['X'] = np.random.rand(n, p)
    slm1['df'] = p+1
    slm1['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm1['coef'] = np.random.rand(p, v, k)

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = np.random.rand(int(k*(k+1)/2), v)
    slm2['coef'] = np.random.rand(p, v, k)

    dummy_test(slm1, slm2)


def test_10():
    fname = './data_OFF/thickness_slm.mat'
    f = loadmat(fname)
    slm1 = {}
    slm1['X'] = f['slm']['X'][0, 0]
    slm1['df'] = f['slm']['df'][0, 0][0, 0]
    slm1['coef'] = f['slm']['coef'][0, 0]
    slm1['SSE'] = f['slm']['SSE'][0, 0]
    slm1['tri'] = f['slm']['tri'][0, 0]
    slm1['resl'] = f['slm']['resl'][0, 0]
    AGE = f['slm']['AGE'][0, 0]

    # run python t-test function
    Python_slm = SLM(Term(1), Term(1))
    for key in slm1.keys():
        setattr(Python_slm, key, slm1[key])
    setattr(Python_slm, 'AGE', AGE)
    t_test(Python_slm)

    slm2 = slm1.copy()
    slm2['t'] = getattr(Python_slm, 't') + np.random.rand()

    dummy_test(slm1, slm2)


def test_11():
    fname = './data_OFF/thickness_slm.mat'
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0, 0]
    slm['df'] = f['slm']['df'][0, 0][0, 0]
    slm['coef'] = f['slm']['coef'][0, 0]
    slm['SSE'] = f['slm']['SSE'][0, 0]
    slm['tri'] = f['slm']['tri'][0, 0]
    slm['resl'] = f['slm']['resl'][0, 0]
    AGE = f['slm']['AGE'][0, 0]

    # run python t-test function
    Python_slm = SLM(Term(1), Term(1))
    for key in slm.keys():
        setattr(Python_slm, key, slm[key])
    setattr(Python_slm, 'AGE', AGE)
    t_test(Python_slm)

    slm1 = slm.copy()
    slm1['X'] = getattr(Python_slm, 'X') + 2
    slm1['t'] = getattr(Python_slm, 't') + 0.2
    slm1['df'] = getattr(Python_slm, 'df') + 2
    slm1['coef'] = getattr(Python_slm, 'coef') + 0.2

    dummy_test(slm, slm1)


def test_12():
    fname = './data_OFF/sofopofo1.mat'
    f = loadmat(fname)

    Y = f['sofie']['T'][0, 0]

    params = f['sofie']['model'][0, 0]
    colnames = ['1', 'ak', 'female', 'male', 'Affect', 'Control1',
                'Perspective', 'Presence', 'ink']
    M = Term(params, colnames)

    SW = {}
    SW['tri'] = f['sofie']['SW'][0, 0]['tri'][0, 0]


    ### slm = SurfStatLinMod(Y, M, SW) ###

    Py_slm = SLM(M, Term(1))
    Py_slm.surf = {"tri": SW["tri"]}
    Py_slm.linear_model(Y)


    ### slm = SurfStatT(slm, contrast) ###


    contrast = np.array([[37], [41], [24], [37], [26], [28], [44], [26], [22],
                         [32], [34], [33], [35], [25], [22], [27], [22], [29],
                         [29], [24]])

    setattr(Py_slm, 'contrast', contrast)

    t_test(Py_slm)

    slm1 = {}
    slm1['X'] = getattr(Py_slm, 'X')
    slm1['df'] = getattr(Py_slm, 'df')
    slm1['coef'] = getattr(Py_slm, 'coef')
    slm1['SSE'] = getattr(Py_slm, 'SSE')
    slm1['tri'] = getattr(Py_slm, 'tri')
    slm1['resl'] = getattr(Py_slm, 'resl')
    slm1['t'] = getattr(Py_slm, 't')

    slm2 = slm1.copy()
    slm2['t'] = slm1['t'] + np.random.rand()

    dummy_test(slm1, slm2)
