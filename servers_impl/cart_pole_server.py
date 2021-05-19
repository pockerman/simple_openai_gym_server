import logging
import numpy as np
import gym

from zmq_client.zmq_client import ZmqClient
from servers_impl.server_base import ServerBase
from messages.messages import (InfoMessage, DynamicsMessage, MakeMessage, ResetMessage, StepMessage)

class CartPoleServer(ServerBase):

    def __init__(self, zmq_client: ZmqClient) -> None:
        super(CartPoleServer, self).__init__(zmq_client=zmq_client)

    def _serve(self) -> None:
        while True:

            # get the requets
            request = self.zmq_client.receive()
            method = request['method']
            param = request['param']

            if method == 'info':

                # Request is info about the world
                (action_space_type,
                 action_space_shape,
                 observation_space_type,
                 observation_space_shape,
                 observation_space_size) = self._info()

                logging.info("Action space type " + action_space_type)
                logging.info("Action space shape " + str(action_space_shape))
                logging.info("Observation space type " + observation_space_type)
                logging.info("Observation space shape " + str(observation_space_shape))

                self.zmq_client.send(InfoMessage(action_space_type,
                                                 action_space_shape,
                                                 observation_space_type,
                                                 observation_space_shape,
                                                 observation_space_size))
            elif method == 'dynamics':
                state = param["state"]
                action = param["action"]
                data = self._env.P[state][action]
                prob = []
                next_state = []
                reward = []
                done = []

                for item in data:
                    prob.append(item[0])
                    next_state.append(item[1])
                    reward.append(item[2])
                    done.append(item[3])

                self.zmq_client.send(DynamicsMessage(prob=prob, next_state=next_state,
                                                     reward=reward, done=done))

            elif method == 'make':
                self.make(param['env_name'], param['num_envs'])
                self.zmq_client.send(MakeMessage())

            elif method == 'reset':
                observation = self.__reset()
                logging.info(" Observation " + str(observation))
                self.zmq_client.send(ResetMessage(observation))

            elif method == 'step':
                if 'render' in param:
                    result = self.__step(
                        np.array(param['actions']), param['render'])
                else:
                    result = self.__step(np.array(param['actions']))
                self.zmq_client.send(StepMessage(result[0],
                                                 result[1],
                                                 result[2],
                                                 result[3]['reward']))


    def _info(self):
            pass

    def _make(self):

        # if the environment is not
        # None close it down
        if self.env:
            self.env.close()
            self.env = None

        logging.info("Making %d %ss", 1, self.env_name)
        self.env = gym.make(env_name)
