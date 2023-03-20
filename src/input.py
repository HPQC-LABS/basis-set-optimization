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

    exp_array[np.where(exp_array[:, 1] == 0), 0] = exponents[:]

    print(exp_array[:, 0])
    basis_string = ''.join([''.join([
        get_basis_substring(exp_array[i, 0], orbital) if j == 0 else
        get_basis_substring(exp_array[i + basis_cum_nums[j-1], 0], orbital)
        for i in range(num)
    ]) for j, [num, orbital] in enumerate(basis_set)])

    return basis_string

def atomic_energy(
    exponents,
    mol, basis_str, exp_array=None, return_J=False,
):
    basis_string = get_basis_string(basis_str, exponents, exp_array)
    mol.basis = {'Ar': pyscf.gto.basis.parse(basis_string)}

    mol.verbose = VERBOSITY
    mol.build()

    mf = scf.RHF(mol)
    e = mf.kernel()

    print(f"exp = {exponents}")
    print(f"E = {e}")

    if return_J is False:
        return e
    
    if return_J is True:
        jac = mf.energy_grad()
        grad_E = np.array(jac.exp)
        grad_E = grad_E[np.where(exp_array[:, 1] == 0)]
        return e, grad_E

def minimize_energy(mol, basis_str, exp_array=None, use_Jacobian=False):
    x0 = exp_array[np.where(exp_array[:, 1] == 0), 0]
    bnds = ((1e-9, None) for _ in range(x0.size))

    res = optimize.minimize(
        atomic_energy,
        x0,
        args=(mol, basis_str, exp_array, use_Jacobian),
        method="Nelder-Mead",
        jac=use_Jacobian,
        hess=None,
        hessp=None,
        bounds=bnds,
        tol=1e-10,
        callback=None,
        options={"maxfev": 100000, "ftol": 1e-12, "maxiter": 100000},
    )
    
    final_E = atomic_energy(res.x, mol, basis_str)
    print(res)
    print(f"E = {final_E}")
    
    nums_orbs = parse_basis_str(basis_str)
    max_n = max([n for [n, _] in nums_orbs])
    orbs = [orb for [_, orb] in nums_orbs]
    basis_nums = [num for [num, _] in nums_orbs]
    basis_cum_nums = np.cumsum(basis_nums)
    basis_cum_nums = np.concatenate(([0], basis_cum_nums))


    results = res.x

    print(''.join([f"{orb:<26}" for orb in orbs]))

    for i in range(max_n):
        print('    '.join([f"{results[i+basis_cum_nums[j]]:.16e}" if i < basis_nums[j] else '' for j in range(len(nums_orbs))]))

    print(f"E = {final_E}")
    print(f"exp = [{','.join(['{:.16e}'.format(x) for x in res.x])}]")

if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = 'Ar 0 0 0'

    basis = "3s2p"
    N = 5
    
    exps = np.zeros((N, 2))
    exps[:, 0] = np.array([0.5 * (N - i) for i in range(N)])

    # freeze exponent 1
    exps[0, 1] = 1

    minimize_energy(mol, basis, exps)
