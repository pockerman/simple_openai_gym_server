from zmq_client.zmq_client import ZmqClient
from abc import abstractmethod, ABC
import logging
import gym


class ServerBase(ABC):
    """
    Base class for deriving server implementations
    """
    def __init__(self, configuration: dict, zmq_client: ZmqClient) -> None:
        self._configuration = configuration
        self._zmq_client: ZmqClient = zmq_client
        self._env: gym.Env = None
        self._shut_down_flag = False

    @property
    def configuration(self) -> dict:
        return self._configuration

    @property
    def zmq_client(self)-> ZmqClient:
        return self._zmq_client

    @property
    def env(self) -> gym.Env:
        return self._env

    @env.setter
    def env(self, value: gym.Env) -> None:
        self._env = value

    @property
    def env_name(self) -> str:
        return self._configuration["env"]["name"]

    @property
    def n_environments(self) -> int:
        return self.configuration["env"]["copies"]

    def serve(self):
        """
        Run the server.
        """
        logging.info("Serving...")
        try:
            self._serve()
        except KeyboardInterrupt:
            self._shut_down_flag = True

    @abstractmethod
    def _serve(self) -> None:
        pass