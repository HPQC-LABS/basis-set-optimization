#!/usr/bin/env python

import pyscf, re, os, numpy as np
from pyscfad import gto, scf

from scipy import optimize

SOLVER = 'SLSQP'
VERBOSITY = 9

def parse_basis_str(slug):
    numbers_and_letters = re.findall(r'[A-Za-z]+|\d+', slug)
    numbers_with_letters = [
        [int(numbers_and_letters[i]), numbers_and_letters[i+1].capitalize()]
        for i in range(0, len(numbers_and_letters), 2)
    ]
    return numbers_with_letters

def decaying_nums(n):
    return np.array([0.5 * (n - i) for i in range(n)])

def get_basis_substring(exponent, orbital):
    substring = f'''
    Ar  {orbital}
        {exponent}              1.0'''
    return substring

def get_basis_string(basis_str, exponents, exp_array=None):
    basis_set = parse_basis_str(basis_str)
    basis_nums = [num for [num, _] in basis_set]
    basis_cum_nums = np.cumsum(basis_nums)

    if exp_array is None:
        exp_array = np.zeros((exponents.size, 2))

    exp_array[np.where(exp_array[:, 1] == 0), 0] = exponents[np.where(exp_array[:, 1] == 0)]

    basis_string = ''.join([''.join([
        get_basis_substring(exp_array[i, 0], orbital) if j == 0 else
        get_basis_substring(exp_array[i + basis_cum_nums[j-1], 0], orbital)
        for i in range(num)
    ]) for j, [num, orbital] in enumerate(basis_set)])

    return basis_string

def atomic_energy(exponents, basis_str, exp_array=None):
    mol = gto.Mole()
    mol.atom = 'Ar 0 0 0'  # in Angstrom

    basis_string = get_basis_string(basis_str, exponents, exp_array)
    mol.basis = {'Ar': pyscf.gto.basis.parse(basis_string)}

    mol.verbose = VERBOSITY
    mol.build()

    mf = scf.RHF(mol)
    e = mf.kernel()

    print(f"exp = {exponents}")
    print(f"E = {e}")

    return e

def grad_atomic_energy(exponents, basis_str, exp_array=None):
    mol = gto.Mole()
    mol.atom = 'Ar 0 0 0'  # in Angstrom

    basis_string = get_basis_string(basis_str, exponents, exp_array)
    mol.basis = {'Ar': pyscf.gto.basis.parse(basis_string)}
    
    mol.verbose = VERBOSITY
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    jac = mf.energy_grad()

    print(f"exp = {exponents}")
    print(f"grad_E = {jac.exp}")

    grad_E = np.array(jac.exp)

    return grad_E

def minimize_energy(basis_str, exp_array=None):
    x0 = exp_array[:, 0]
    bnds = ((1e-9, None) for _ in range(exp_array.shape[0]))

    res = optimize.minimize(
        atomic_energy,
        x0,
        args=(basis_str, exp_array),
        method=SOLVER,
        jac=grad_atomic_energy,
        hess=None,
        hessp=None,
        bounds=bnds,
        tol=1e-9,
        callback=None,
        options={"maxfev": 10000, "ftol": 1e-9},
    )
    
    print(res)
    print(f"E = {atomic_energy(res.x, basis_str)}")
    print(f"exp = [{','.join(['{:.16e}'.format(x) for x in res.x])}]")
    
number_of_s_exps = 7
number_of_p_exps = 10

exps = np.zeros((number_of_s_exps+number_of_p_exps,2))
#exps[:, 0] = decaying_nums(5)

exps=np.matrix('\
2.6182507849929175e+04  0 ;\
7.6182507849929175e+03  0 ;\
3.6182507849929175e+03  0 ;\
1.6182507849929175e+03  0 ;\
2.3978398735110531e+02  0 ;\
5.1897914206496637e+01  0 ;\
4.3870852942693928e+00  0 ;\
1.3627930032386791e+03  0 ;\
6.6518072575151780e+02  0 ;\
3.0332064186335555e+02  0 ;\
3.1614645606305231e+02  0 ;\
4.9300144018458525e+01  0 ;\
4.7537785732569198e+00  0 ;\
1.4747293500057861e+01  0 ;\
4.0328123273769173e-01  0 ;\
1.2333383692864934e+00  0 ;\
3.8716558480433622e-02  0  \
')
exps=np.array(exps)

basis =  str(number_of_s_exps)+"s"+str(number_of_p_exps)+"p"

minimize_energy(basis, exps)

if np.max(exps[:,1]) ==1:
    name = 'out_'+SOLVER+'-f_'+basis+'.job'
else:    
    name = 'out_'+SOLVER+'_'+basis+'.job'
os.rename("out_and_err."+os.environ["SLURM_JOB_ID"],name+'.'+os.environ["SLURM_JOB_ID"])
