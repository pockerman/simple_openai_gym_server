"""
Simple factory class for server instances creation
"""

from zmq_client.zmq_client import ZmqClient
from servers_impl.server_base import ServerBase
from servers_impl.frozen_lake_server import FrozenLakeServer
from servers_impl.cart_pole_server import CartPoleServer


class ServerFactory(object):

    @staticmethod
    def build(configuration: dict, zmq_client: ZmqClient) -> ServerBase:

        if "server_type" not in configuration:
            raise ValueError("Server type not specified in configuration")

        if configuration["server_type"] == "FrozenLake":
            return FrozenLakeServer(configuration=configuration, zmq_client=zmq_client)
        elif configuration["server_type"] == "CartPole":
            return CartPoleServer(configuration=configuration, zmq_client=zmq_client)

        raise ValueError("Unknown server type: {0}".format(configuration["server_type"]))