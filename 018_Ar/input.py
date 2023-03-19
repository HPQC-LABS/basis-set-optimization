#!/usr/bin/env python

import pyscf, re, os, numpy as np
from pyscfad import gto, scf

from scipy import optimize

SOLVER = 'Nelder-Mead'
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
    bnds = ((1e-9, None) for _ in range(exp_array.shape[0])) # Lower and upper bounds
    res = optimize.minimize(
        atomic_energy,
        x0,
        args=(basis_str, exp_array),
        method=SOLVER,
        jac=grad_atomic_energy,
        hess=None,
        hessp=None,
        bounds=bnds,
        tol=1e-13,
        callback=None,
        options={"maxfev": 10000, "ftol": 1e-10},
    )
    print(res)
    print(f"Final energy = {atomic_energy(res.x, basis_str)}")
    print(f"exp = [{','.join(['{:.16e}'.format(x) for x in res.x])}]")
    
number_of_s_exps = 15
number_of_p_exps = 10

exps = np.zeros((number_of_s_exps+number_of_p_exps,2))
#exps[:, 0] = decaying_nums(5)

exps=np.matrix('\
553443.07887896 0 ;\
88097.43848988  0 ;\
21527.90787686  0 ;\
905.67284079    0 ;\
6652.52600651   0 ;\
2345.47638154   0 ;\
367.18677893    0 ;\
151.2057879     0 ;\
63.4933354      0 ;\
27.14624137     0 ;\
8.89046117      0 ;\
3.90630183      0 ;\
1.74380611      0 ;\
0.60349363      0 ;\
0.21667994      0 ;\
1362.10974035   0 ;\
324.06753199    0 ;\
40.00847179     0 ;\
105.3809149     0 ;\
16.59082098     0 ;\
3.12845673      0 ;\
7.15220658      0 ;\
0.46206191      0 ;\
1.20115777      0 ;\
0.15913117      0  \
')
exps=np.array(exps)

basis =  str(number_of_s_exps)+"s"+str(number_of_p_exps)+"p"
minimize_energy(basis, exps)
if np.max(exps[:,1]) ==1:
    name = 'out_'+SOLVER+'-f_'+basis+'.job'
else:    
    name = 'out_'+SOLVER+'_'+basis+'.job'
os.rename("out_and_err."+os.environ["SLURM_JOB_ID"],name+'.'+os.environ["SLURM_JOB_ID"])
