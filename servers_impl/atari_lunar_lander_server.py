import logging
import gym
import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from zmq_client.zmq_client import ZmqClient
from servers_impl.server_base import ServerBase
from messages.messages import (InfoMessage, MakeMessage, ResetMessage, StepMessage)
from servers_impl.utils.envs import TransposeImage, VecRewardInfo, VecFrameStack


class AtariLunarLander(ServerBase):
    def __init__(self, configuration, zmq_client: ZmqClient) -> None:
        super(AtariLunarLander, self).__init__(configuration=configuration,
                                               zmq_client=zmq_client)
        logging.info("AtariLunarLander server initialized...")

    def _serve(self) -> None:

        while True:

            # get the requets
            request = self.zmq_client.receive()
            method = request['method']
            param = request['param']

            if method == 'info':

                # Request is info about the world
                (action_space_type,
                 action_space_size,
                 observation_space_type,
                 observation_space_size, has_errors, error_msg) = self._info()

                self.zmq_client.send(InfoMessage(action_space_type, action_space_size,
                                                 observation_space_type, observation_space_size,
                                                 has_error=has_errors, error_msg=error_msg))
            elif method == 'make':

                try:
                    self._make()
                    self.zmq_client.send(MakeMessage(result="OK"))
                except Exception as e:
                    self.zmq_client.send(MakeMessage(result="Error " + str(e)))

            elif method == 'reset':
                try:
                    observation = self._reset()
                    logging.info(" Observation " + str(observation))
                    self.zmq_client.send(ResetMessage(observation=observation, result="OK"))
                except Exception as e:
                    self.zmq_client.send(ResetMessage(observation=np.ndarray([]), result="Error " + str(e)))

            elif method == 'step':
                result = self._step(action=param["action"], render=param["render"])
                self.zmq_client.send(StepMessage(result[0],
                                                 result[1],
                                                 result[2],
                                                 result[3]['reward']))

    def _info(self):

        """
        Return info about the currently loaded environment
        """

        logging.info("Sending info for env %s", self.env_name)

        if self.env is None:
            action_space_type = self.env.action_space.__class__.__name__
            action_space_size = self.env.action_space.n
            observation_space_type = self.env.observation_space.__class__.__name__
            observation_space_size = self.env.observation_space.n
            has_errors = True
            error_msg = "Environment not initialized"
        else:
            has_errors = False
            error_msg = ""
            action_space_type = self.env.action_space.__class__.__name__
            action_space_size = self.env.action_space.n
            observation_space_type = self.env.observation_space.__class__.__name__
            observation_space_size = self.env.observation_space.n

        logging.info("Action space type " + action_space_type)
        logging.info("Action space size " + str(action_space_size))
        logging.info("Observation space type " + observation_space_type)
        logging.info("Observation space size " + str(observation_space_size))

        return (action_space_type, action_space_size,
                observation_space_type, observation_space_size,
                has_errors, error_msg)

    def _reset(self):
        logging.info("Resetting env %s ", self.env_name)
        return self.env.reset()

    def _step(self, action, render):
        logging.info("Stepping in env %s action %s", self.env_name, action)

        observation = self.env.step(action)
        if render:
            self._env.render()
        return observation

    def _make(self):

        # if the environment is not
        # None close it down
        if self.env:
            self.env.close()
            self.env = None

        logging.info("Making %d %ss", self.n_environments, self.env_name)

        if self.n_environments == 1:
            self.env = gym.make(self.env_name)
        else:
            self.env = self.__make_atari_vec_envs(0, self.n_environments)

    def __make_env(self, seed: int, idx: int):

        def _thunk():
            env = make_atari(self.env_name)
            env.seed(seed + idx)

            if len(env.observation_space.shape) == 3:
                    env = wrap_deepmind(env)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env)

            return env

        return _thunk

    def __make_atari_vec_envs(self, seed: int, num_processes: int, num_frame_stack=None):

        envs = [self.__make_env(seed=seed, idx=i) for i in range(num_processes)]

        if len(envs) > 1:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)

        envs = VecRewardInfo(envs)

        if num_frame_stack is not None:
            envs = VecFrameStack(envs, num_frame_stack)
        elif len(envs.observation_space.shape) == 3:
            envs = VecFrameStack(envs, 4)

        return envs
