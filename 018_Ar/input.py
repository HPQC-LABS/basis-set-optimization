#!/usr/bin/env python

import pyscf
from pyscfad import gto, scf
import numpy as np
import re

from scipy import optimize

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
        method="BFGS",
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
    
exps = np.zeros((11, 2))
#exps[:, 0] = decaying_nums(5)
exps_old = np.array([8.0e+05,1.8354961655666553e+04,2.2321569373256953e+03,4.5777843088607506e+02,1.2189136988757589e+02,3.7618460714610308e+01,4.7411562274897623e+00,3.9118886400018660e-01,8.5986702324693809e+00,4.9085158086350911e-01])
exps[1:, 0] = exps_old[:]
exps[0, 0] = np.max(exps_old) * 4.0

basis = "9s2p"

minimize_energy(basis, exps)
