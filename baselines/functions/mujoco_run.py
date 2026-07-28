import os
from typing import Tuple, Optional, ClassVar, Dict
from typing import NamedTuple, Optional, Union, Any, Type, Dict, Generic, TypeVar, Tuple
import gym
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent
import os
from typing import Tuple, Optional, ClassVar, Dict
from typing import NamedTuple, Optional, Union, Any, Type, Dict, Generic, TypeVar, Tuple
import gym
import numpy as np
import torch



class RunningStat(object):
    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape,  dtype=np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}".format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class MujucoPolicyFunc():
    ANT_ENV: ClassVar[Tuple[str, float, float, int]] = ('Ant-v4', -1.0, 1.0, 10)
    SWIMMER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Swimmer-v4', -1.0, 1.0, 10)
    HALF_CHEETAH_ENV: ClassVar[Tuple[str, float, float, int]] = ('HalfCheetah-v4', -1.0, 1.0, 10)
    HOPPER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Hopper-v4', -1.4, 1.4, 10)
    WALKER_2D_ENV: ClassVar[Tuple[str, float, float, int]] = ('Walker2d-v4', -1.8, 0.9, 10)
    HUMANOID_ENV: ClassVar[Tuple[str, float, float, int]] = ('Humanoid-v4', -1.0, 1.0, 10)

    ENV_CP = {
        ANT_ENV[0]: 10.0,
        SWIMMER_ENV[0]: 30.0,
        HALF_CHEETAH_ENV[0]: 10.0,
        HOPPER_ENV[0]: 100.0,
        WALKER_2D_ENV[0]: 50.0,
        HUMANOID_ENV[0]: 20.0
    }

    def __init__(self, policy_file: str, env: str, lb: float, ub: float, num_rollouts, dtype, device, seed, negate=True):
        self._env_name = env
        self._env = gym.make(env)
        self._env.reset(seed=2025)
        state_dims = self._env.observation_space.shape[0]
        action_dims = self._env.action_space.shape[0]
        self._dims = state_dims * action_dims
        self.dim = self._dims
        self._lb = np.full(self._dims, lb)
        self._ub = np.full(self._dims, ub)
        self._policy_shape = (action_dims, state_dims)
        self._num_rollouts = num_rollouts
        self._render = False
        self.dtype = dtype
        self.device = device
        self.bounds = torch.zeros((2, self.dim), dtype=dtype, device=device)
        self.negate = negate
        self.seed = seed
        self._rs = RunningStat(state_dims)

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def is_minimizing(self) -> bool:
        return False

    # 단일 정책 벡터를 평가하는 함수
    def _evaluate_single_vector(self, vector_idx, vector):
        # 정책 행렬로 벡터 재구성
        policy_matrix = vector.reshape(self._policy_shape)
        
        # 개별 환경 생성
        env = gym.make(self._env_name)
        rs = RunningStat(self._env.observation_space.shape[0])
        
        rewards = []
        for t in range(self._num_rollouts):
            obs, _ = env.reset(seed=self.seed + vector_idx + t)
            done = False
            total_reward = 0.
            while True:
                rs.push(obs)
                norm_obs = (obs - rs.mean) / (rs.std + 1e-6)
                action = np.dot(policy_matrix, norm_obs)
                obs, r, done, truncated, _ = env.step(action)
                total_reward += r
                if done or truncated:
                    break
            rewards.append(total_reward)
        
        env.close()
        if self._env_name == 'HalfCheetah-v4':
            return np.mean(rewards), -500 - np.array(rewards)
        elif self._env_name == 'Ant-v4':
            return np.mean(rewards), -1000 - np.array(rewards)
        else:
            return np.mean(rewards), 100 - np.array(rewards)
    
    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 텐서를 NumPy 배열로 변환
        x_batch_np = x_batch.cpu().detach().numpy()
        batch_size = x_batch_np.shape[0]
        
        # 결과를 저장할 배열 초기화
        fx = np.zeros(batch_size)
        cs = np.zeros((batch_size, self._num_rollouts))
        
        # ThreadPoolExecutor를 사용하여 벡터 평가 병렬화
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, batch_size)) as executor:
            # 모든 벡터에 대한 평가 작업 제출
            future_to_idx = {
                executor.submit(self._evaluate_single_vector, i, vector): i 
                for i, vector in enumerate(x_batch_np)
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    mean_reward, constraints = future.result()
                    fx[idx] = mean_reward
                    cs[idx] = constraints
                except Exception as e:
                    print(f"벡터 {idx} 평가 중 오류 발생: {e}")
                    fx[idx] = -1000.0  # 오류 발생 시 페널티 값
                    cs[idx] = 1000.0   # 오류 발생 시 제약조건 위반
        
        # 결과를 텐서로 변환
        fx_tensor = torch.tensor(fx, dtype=self.dtype, device=self.device).unsqueeze(-1)
        # 제약조건 텐서 생성
        c1 = torch.tensor(cs, dtype=self.dtype, device=self.device)
        
        return fx_tensor.squeeze(dim=-1), c1
    
    
T = TypeVar('T')

class ObjectFactory(Generic[T]):
    def __init__(self, clz: Type[T], args: Optional[Tuple] = None, kwargs: Optional[Dict] = None):
        self._clz = clz
        self._args = args
        self._kwargs = kwargs

    def make_object(self) -> T:
        if self._args is not None and self._kwargs is not None:
            return self._clz(*self._args, **self._kwargs)
        elif self._args is not None:
            return self._clz(*self._args)
        elif self._kwargs is not None:
            return self._clz(**self._kwargs)
        else:
            return self._clz()

    @property
    def clz(self) -> Type:
        return self._clz

    @property
    def args(self) -> Optional[Tuple]:
        return self._args

    @property
    def kwargs(self) -> Optional[Dict]:
        return self._kwargs


func_dir = os.path.dirname(os.path.abspath(__file__))
func_factories = {
    "ant": ObjectFactory(MujucoPolicyFunc,
                         (f"{func_dir}/trained_policies/Ant-v1/lin_policy_plus.npz", *MujucoPolicyFunc.ANT_ENV)),
    "half_cheetah": ObjectFactory(MujucoPolicyFunc,
                                  (f"{func_dir}/trained_policies/HalfCheetah-v1/lin_policy_plus.npz",
                                   *MujucoPolicyFunc.HALF_CHEETAH_ENV)),
    "hopper": ObjectFactory(MujucoPolicyFunc,
                            (f"{func_dir}/trained_policies/Hopper-v1/lin_policy_plus.npz",
                             *MujucoPolicyFunc.HOPPER_ENV)),
    "humanoid": ObjectFactory(MujucoPolicyFunc,
                              (f"{func_dir}/trained_policies/Humanoid-v1/lin_policy_plus.npz",
                               *MujucoPolicyFunc.HUMANOID_ENV)),
    "swimmer": ObjectFactory(MujucoPolicyFunc,
                             (f"{func_dir}/trained_policies/Swimmer-v1/lin_policy_plus.npz",
                              *MujucoPolicyFunc.SWIMMER_ENV)),
    "walker_2d": ObjectFactory(MujucoPolicyFunc,
                               (f"{func_dir}/trained_policies/Walker2d-v1/lin_policy_plus.npz",
                                *MujucoPolicyFunc.WALKER_2D_ENV)),
}