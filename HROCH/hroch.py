
import os
import platform
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PHCRegressor(BaseEstimator, RegressorMixin):
    '''Parallel Hill Climbing symbolic regressor'''

    def __init__(self,
                 num_threads: int = 8,
                 time_limit: float = 5.0,
                 iter_limit: int = 0,
                 stopping_criteria: float = 0.0,
                 precision: str = 'f32',
                 problem: any = 'math',
                 feature_probs: any = None,
                 save_model: bool = False,
                 random_state: int = 0,
                 verbose: bool = False,
                 pop_size: int = 64,
                 pop_sel: int = 4,
                 const_size: int = 8,
                 code_size: int = 32
                 ):

        self.num_threads = num_threads
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.stopping_criteria = stopping_criteria
        self.verbose = verbose
        self.precision = precision
        self.problem = problem
        self.feature_probs = feature_probs
        self.save_model = save_model
        self.random_state = random_state
        self.pop_size = pop_size
        self.pop_sel = pop_sel
        self.const_size = const_size
        self.code_size = code_size

        self.dir = TemporaryDirectory()

    def fit(self, X: np.ndarray, y: np.ndarray):

        X, y = check_X_y(X, y, accept_sparse=False)

        fname_x = self.dir.name + '/x.csv'
        fname_y = self.dir.name + '/y.csv'
        fname_model = self.dir.name + '/model.hrx'
        fname_program = self.dir.name + '/program.hrx'

        if os.path.exists(fname_x):
            os.remove(fname_x)
        if os.path.exists(fname_y):
            os.remove(fname_y)

        np.savetxt(fname_x, X, delimiter=' ')
        np.savetxt(fname_y, y, delimiter=' ')

        path = os.path.dirname(os.path.realpath(__file__))

        cli = path + '/hroch.bin'
        if platform.system() == 'Windows':
            cli = path + '/hroch.exe'

        process = subprocess.Popen([cli,
                                    '--problem', self.__problem_to_string(
                                        self.problem),
                                    '--task', 'fit',
                                    '--x', fname_x,
                                    '--y', fname_y,
                                    '--precision', f'{self.precision}',
                                    '--modelFile' if self.save_model else '--programFile', fname_model if self.save_model else fname_program,
                                    '--timeLimit', f'{int(round(self.time_limit*1000.0))}',
                                    '--iterLimit', f'{self.iter_limit}',
                                    '--numThreads', f'{self.num_threads}',
                                    '--stoppingCriteria', f'{self.stopping_criteria}',
                                    '--randomState', f'{self.random_state}',
                                    '--featProbs', self.__feature_probs_to_string(
                                        self.feature_probs),
                                    '--popSize', f'{self.pop_size}',
                                    '--popSel', f'{self.pop_sel}',
                                    '--constSize', f'{self.const_size}',
                                    '--codeSize', f'{self.code_size}'
                                    ],
                                   cwd=self.dir.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        eq_prefix = 'eq= '
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8')
            if line.startswith(eq_prefix):
                self.sexpr = line[len(eq_prefix):]
            if self.verbose:
                print(line, end='')

        process.stdout.close()
        process.wait()

        self.is_fitted_ = True

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        fname_x = self.dir.name + '/x.csv'
        fname_y = self.dir.name + '/y.csv'
        fname_model = self.dir.name + '/model.hrx'
        fname_program = self.dir.name + '/program.hrx'

        if os.path.exists(fname_x):
            os.remove(fname_x)
        if os.path.exists(fname_y):
            os.remove(fname_y)

        np.savetxt(fname_x, X, delimiter=' ')

        path = os.path.dirname(os.path.realpath(__file__))

        cli = path + '/hroch.bin'
        if platform.system() == 'Windows':
            cli = path + '/hroch.exe'

        process = subprocess.Popen([cli,
                                    '--task', 'predict',
                                    '--x', fname_x,
                                    '--y', fname_y,
                                    '--modelFile' if self.save_model else '--programFile', fname_model if self.save_model else fname_program,
                                    ],
                                   cwd=self.dir.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8')
            if self.verbose:
                print(line, end='')

        process.stdout.close()
        process.wait()
        y = np.genfromtxt(fname_y, delimiter=' ')

        return y

    def __problem_to_string(self, problem: any):
        if isinstance(problem, str):
            if problem == 'simple' or problem == 'math' or problem == 'fuzzy':
                return problem
            raise ValueError('Invalid problem type')
        if not isinstance(problem, dict):
            raise TypeError('Invalid problem type')
        result = ""
        for instr, prob in problem.items():
            if not isinstance(instr, str) or not isinstance(prob, float):
                raise TypeError('Invalid instruction type')
            if len(instr) == 0 or prob < 0.0:
                raise ValueError('Invalid instruction value')
            result = result + f'{instr} {prob};'
        return result

    def __feature_probs_to_string(self, feat: any):
        result = ""
        if feat is None:
            return result
        for prob in feat:
            result = result + f'{prob};'
        return result
