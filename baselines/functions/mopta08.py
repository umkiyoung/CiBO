import os
import stat
import subprocess
import sys
import tempfile
import urllib
from logging import info, warning
from platform import machine
from typing import Optional, Union, List, Type
import numpy as np
import torch
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from functools import partial
class ArgumentError(Exception):
    """
    An exception for an illegal input argmument.
    """
    pass

class EffectiveDimTooLargeException(Exception):
    """
    When the effective dimensionality is too large (for example when larger than the input dimensionality).
    """
    pass

class OutOfBoundsException(Exception):
    """
    When a point falls outside the search space.
    """
    pass

class BoundsMismatchException(Exception):
    """
    When the search space bounds don't have the same length.
    """
    pass

class UnknownBehaviorError(Exception):
    pass

class Benchmark(ABC):
    """
    Abstract benchmark function.

    Args:
        dim: dimensionality of the objective function
        noise_std: the standard deviation of the noise (None means no noise)
        ub: the upper bound, the object will have the attribute ub_vec which is an np array of length dim filled with ub
        lb: the lower bound, the object will have the attribute lb_vec which is an np array of length dim filled with lb
        benchmark_func: the benchmark function, should inherit from SyntheticTestFunction
    """

    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, noise_std: float):

        lb = np.array(lb)
        ub = np.array(ub)
        if (
                not lb.shape == ub.shape
                or not lb.ndim == 1
                or not ub.ndim == 1
                or not dim == len(lb) == len(ub)
        ):
            raise BoundsMismatchException()
        if not np.all(lb < ub):
            raise OutOfBoundsException()
        self.noise_std = noise_std
        self._dim = dim
        self._lb_vec = lb.astype(np.float32)
        self._ub_vec = ub.astype(np.float32)

    @property
    def dim(self) -> int:
        """
        The benchmark dimensionality

        Returns: the benchmark dimensionality

        """
        return self._dim

    @property
    def lb_vec(self) -> np.ndarray:
        """
        The lower bound of the search space of this benchmark (length = benchmark dim)

        Returns: The lower bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._lb_vec

    @property
    def ub_vec(self) -> np.ndarray:
        """
        The upper bound of the search space of this benchmark (length = benchmark dim)

        Returns: The upper bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._ub_vec

    @property
    def fun_name(self) -> str:
        """
        The name of the benchmark function

        Returns: The name of the benchmark function

        """
        return self.__class__.__name__

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        raise NotImplementedError()

class SyntheticBenchmark(Benchmark):
    """
    Abstract class for synthetic benchmarks

    Args:
        dim: the benchmark dimensionality
        ub: np.ndarray: the upper bound of the search space of this benchmark (length = benchmark dim)
        lb: np.ndarray: the lower bound of the search space of this benchmark (length = benchmark dim)
    """

    @abstractmethod
    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, noise_std: float):
        super().__init__(dim, ub, lb, noise_std=noise_std)

    @abstractmethod
    def __call__(
            self, x: Union[np.ndarray, List[float], List[List[float]]]
    ) -> np.ndarray:
        """
        Call the benchmark function for one or multiple points.

        Args:
            x: Union[np.ndarray, List[float], List[List[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional

        Returns:
            np.ndarray: The function values.


        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        # for y in x:
        #    if not np.sum(y < self._lb_vec) == 0:
        #        raise OutOfBoundsException()
        #    if not np.sum(y > self._ub_vec) == 0:
        #        raise OutOfBoundsException

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """

        Returns:
            Optional[Union[float, np.ndarray]]: the optimal value if known

        """
        return None


class MoptaSoftConstraints(SyntheticBenchmark):
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(
            self,
            dim,
            dtype,
            device,
            temp_dir: Optional[str] = None,
            binary_path: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        
        self.dtype = dtype
        self.device = device
        self.noise_std = noise_std
        
        super().__init__(dim, np.ones(dim), np.zeros(dim), noise_std=noise_std)
       
        bounds_np = np.stack([self._lb_vec, self._ub_vec], axis=0)
        self.bounds = torch.from_numpy(bounds_np).to(dtype=self.dtype)

        
        if self.device != 'cpu' and torch.cuda.is_available():
            try:
                self.bounds = self.bounds.to(device=self.device)
            except RuntimeError as e:
                warning(f"Could not move bounds to {self.device}: {e}. Keeping on CPU.")
                # fallback: keep self.device='cpu'
                self.device = 'cpu'

        if binary_path is None:
            self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
            self.machine = machine().lower()
            if self.machine == "armv7l":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_armhf.bin"
            elif self.machine == "x86_64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_elf64.bin"
            elif self.machine == "i386":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_elf32.bin"
            elif self.machine == "amd64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_amd64.exe"
            else:
                raise RuntimeError("Machine with this architecture is not supported")
            self._mopta_exectutable = os.path.join(
                os.getcwd(), self._mopta_exectutable
            )

            if not os.path.exists(self._mopta_exectutable):
                basename = os.path.basename(self._mopta_exectutable)
                info(f"Mopta08 executable for this architecture not locally available. Downloading '{basename}'...")
                urllib.request.urlretrieve(
                    f"https://mopta.papenmeier.io/{os.path.basename(self._mopta_exectutable)}",
                    self._mopta_exectutable)
                os.chmod(self._mopta_exectutable, stat.S_IXUSR)

        else:
            self._mopta_exectutable = binary_path
        if temp_dir is None:
            self.directory_file_descriptor = tempfile.TemporaryDirectory()
            self.directory_name = self.directory_file_descriptor.name
        else:
            if not os.path.exists(temp_dir):
                warning(f"Given directory '{temp_dir}' does not exist. Creating...")
                os.mkdir(temp_dir)
            self.directory_name = temp_dir

    def __call__(self, x):
        super(MoptaSoftConstraints, self).__call__(x)
        x = np.array(x)   
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        # create tmp dir for mopta binary

        # vals = np.array([self._call(y) for y in x]).squeeze()

        # return vals + np.random.normal(
        #     np.zeros_like(vals), np.ones_like(vals) * self.noise_std, vals.shape
        # )
        vals_and_constraints = [self._call(y) for y in x]
        vals, constraints = zip(*vals_and_constraints)
        vals = np.array(vals).squeeze()
        constraints = np.array(constraints)
        
        return vals, constraints

    def _call(self, x: np.ndarray):
        
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.txt")
            
            with open(input_path, "w") as f:
                for val in x:
                    f.write(f"{val}\n")

            popen = subprocess.Popen(
                self._mopta_exectutable,
                stdout=subprocess.PIPE,
                cwd=tmp_dir,
            )
            popen.wait()

            output_path = os.path.join(tmp_dir, "output.txt")
            with open(output_path, "r") as f:
                output = [float(line.strip()) for line in f if line.strip()]
            value, constraints = output[0], output[1:]
            return value, constraints
            # return value + 10 * np.sum(np.clip(constraints, 0, None))

    @property
    def optimal_value(self) -> Optional[np.ndarray]:

        return np.array(-200.0)
    

def evaluate_single(x_single, benchmark: MoptaSoftConstraints):
    return benchmark._call(x_single)

def evaluate_batch_parallel(x, benchmark: MoptaSoftConstraints, dtype,device, processes=None):
    x = x.detach().cpu().numpy()
    # if processes is None:
    #     processes = cpu_count()//2

    # with Pool(processes) as pool:
    #     results = pool.map(partial(evaluate_single, benchmark=benchmark), x)
    
    values, constraints = benchmark(x)
    # values, constraints = zip(*results)
    values = torch.as_tensor(values, dtype=dtype, device=device) * -1
    constraints = torch.as_tensor(constraints, dtype=dtype, device=device)
    return values, constraints
