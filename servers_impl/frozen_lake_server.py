import logging
import numpy as np
from typing import Tuple
import gym


from zmq_client.zmq_client import ZmqClient
from servers_impl.server_base import ServerBase
from messages.messages import (InfoMessage, DynamicsMessage, MakeMessage, ResetMessage, StepMessage)


class FrozenLakeServer(ServerBase):
    """
    Server for FrozenLake environment
    """

    def __init__(self, configuration: dict, zmq_client: ZmqClient) -> None:
        super(FrozenLakeServer, self).__init__(configuration=configuration, zmq_client=zmq_client)
        logging.info("Gym server initialized")

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

                self.zmq_client.send(InfoMessage(action_space_type,
                                                 action_space_size,
                                                 observation_space_type,
                                                 observation_space_size,
                                                 has_error=has_errors,
                                                 error_msg=error_msg))
            elif method == 'dynamics':
                state = param["state"]
                action = param["action"]

                prob, next_state, reward, done = self._dynamics(state=state, action=action)

                self.zmq_client.send(DynamicsMessage(prob=prob, next_state=next_state,
                                                     reward=reward, done=done))

            elif method == 'make':
                self._make()
                self.zmq_client.send(MakeMessage())

            elif method == 'reset':
                observation = self._reset()
                logging.info(" Observation " + str(observation))
                self.zmq_client.send(ResetMessage(observation))

            elif method == 'step':
                result = self._step(action=param["action"], render=param["render"])
                self.zmq_client.send(StepMessage(result[0],
                                                 result[1],
                                                 result[2],
                                                 result[3]['reward']))

    def _info(self):

        logging.info("Sending info for env %s", self.env_name)

        """
        Return info about the currently loaded environment
        """

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

    def _dynamics(self, state, action) -> Tuple[list, list, list, list]:

        data = self.env.P[state][action]
        prob = []
        next_state = []
        reward = []
        done = []

        for item in data:
            prob.append(item[0])
            next_state.append(item[1])
            reward.append(item[2])
            done.append(item[3])
        return prob, next_state, reward, done

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

        logging.info("Making %d %ss", 1, self.env_name)
        self.env = gym.make(self.env_name)
