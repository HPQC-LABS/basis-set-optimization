from energy_minimization import minimize_energy, prepare_inputs
import pandas as pd

orbital_number = 13
factor = 0.05
optimizer = 'Nelder-Mead'
csv_file_path = 'energy_Li.csv'


if __name__ == '__main__':
        df_energy = pd.read_csv(csv_file_path, converters={'exponents': pd.eval})
        exponents = df_energy[df_energy['orbital_num'] == (orbital_number-1) ]['exponents'].to_numpy()[0]

        basis, exps, initial_guess = prepare_inputs(orbital_number, exponents, factor)
        final_E, exponents = minimize_energy(basis, exps, optimizer=optimizer)

        data = {'orbital_num': [orbital_number],
                'energy': [final_E],
                'exponents': [exponents],
                'initial_guess': [initial_guess],
                'factor':[factor],
                'optimizer':[optimizer]
                }

        df_output = pd.DataFrame(data, columns = list(data.keys()))
        df_output.to_csv(csv_file_path, mode='a', index=False, header=False)

