import pyscf
from pyscfad import gto
from pyscf import scf
import re
from scipy import optimize
import numpy as np

VERBOSITY = 9

def parse_basis_str(slug):
    numbers_and_letters = re.findall(r'[A-Za-z]+|\d+', slug)
    numbers_with_letters = [
        [int(numbers_and_letters[i]), numbers_and_letters[i + 1].capitalize()]
        for i in range(0, len(numbers_and_letters), 2)
    ]
    return numbers_with_letters


def decaying_nums(n):
    return np.array([0.5 * (n - i) for i in range(n)])


def get_basis_substring(exponent, orbital):
    substring = f'''
    Li  {orbital}
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
        get_basis_substring(exp_array[i + basis_cum_nums[j - 1], 0], orbital)
        for i in range(num)
    ]) for j, [num, orbital] in enumerate(basis_set)])

    return basis_string


def atomic_energy(exponents, basis_str, exp_array=None):
    mol = gto.Mole()
    mol.atom = 'Li 0 0 0'  # in Angstrom
    mol.spin = 1

    basis_string = get_basis_string(basis_str, exponents, exp_array)
    mol.basis = {'Li': pyscf.gto.basis.parse(basis_string)}

    mol.verbose = VERBOSITY
    mol.build()

    mf = scf.ROHF(mol)
    e = mf.kernel()

    print(f"exp = {exponents}")
    print(f"E = {e}")

    return e


def grad_atomic_energy(exponents, basis_str, exp_array=None):
    mol = gto.Mole()
    mol.atom = 'Li 0 0 0'  # in Angstrom
    mol.spin = 1

    basis_string = get_basis_string(basis_str, exponents, exp_array)
    mol.basis = {'Li': pyscf.gto.basis.parse(basis_string)}

    mol.verbose = VERBOSITY
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()
    jac = mf.energy_grad()

    print(f"exp = {exponents}")
    print(f"grad_E = {jac.exp}")

    grad_E = np.array(jac.exp)

    return grad_E


def minimize_energy(basis_str, exp_array=None, optimizer='Nelder-Mead'):
    x0 = exp_array[:, 0]
    bnds = ((1e-9, None) for _ in range(exp_array.shape[0]))

    res = optimize.minimize(
        atomic_energy,
        x0,
        args=(basis_str, exp_array),
        method=optimizer,
        jac=None,
        hess=None,
        hessp=None,
        bounds=bnds,
        tol=1e-9,
        callback=None,
        options={"maxfev": 10000, "ftol": 1e-9},
    )

    final_E = atomic_energy(res.x, basis_str)
    # print(res)
    # print(f"E = {final_E}")

    nums_orbs = parse_basis_str(basis_str)
    max_n = max([n for [n, _] in nums_orbs])
    orbs = [orb for [_, orb] in nums_orbs]
    basis_nums = [num for [num, _] in nums_orbs]
    basis_cum_nums = np.cumsum(basis_nums)
    basis_cum_nums = np.concatenate(([0], basis_cum_nums))

    results = res.x

    print(''.join([f"{orb:<26}" for orb in orbs]))

    for i in range(max_n):
        print('    '.join(
            [f"{results[i + basis_cum_nums[j]]:.16e}" if i < basis_nums[j] else '' for j in range(len(nums_orbs))]))

    exponents =[x for x in res.x]

    print(f"E = {final_E}")
    print(f"exp = [{','.join(['{:.16e}'.format(x) for x in res.x])}]")

    return final_E, exponents

def prepare_inputs(orbital_number, exponents, factor):
    shape = 2
    exps = np.zeros((orbital_number, shape))

    prev_iter = np.array(exponents)
    initial_guess =  factor * max(prev_iter)

    print(f'initial guess of max exponent is: {initial_guess}')

    exps[:-1, 0] = prev_iter[:]
    exps[-1, 0] = initial_guess

    basis = f"{orbital_number}s"

    return basis, exps, initial_guess