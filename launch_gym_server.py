#!/usr/bin/env python
"""
Pytorch-cpp-rl OpenAI gym server main script.
"""
import argparse
import logging
import json

from zmq_client.zmq_client import ZmqClient
from servers_impl.server_factory import ServerFactory


def main(configuration):
    """
    Host the server.
    """

    # If anything is logged during imports, it messes up our logging so we
    # reset the logging module here
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format=('%(asctime)s %(funcName)s '
                                '[%(levelname)s]: %(message)s'),
                        datefmt='%Y%m%d %H:%M:%S')

    logging.info("Initializing gym server...")
    zmq_client = ZmqClient(10201)
    logging.info("Connecting to client")
    zmq_client.send("Connection established")
    logging.info("Connected")

    # build the server
    server = ServerFactory.build(configuration=configuration, zmq_client=zmq_client)

    logging.info(f"Created gym server {server.env_name}")
    try:
        server.serve()
    #except Exception as e:  # pylint: disable=bare-except
    #    print("An exception was raised " + str(e))
    #    raise e
    except:
        import pdb
        pdb.post_mortem()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help="You must specify a json "
                             "formatted configuration file")

    args = parser.parse_args()

    with open(args.config) as json_file:
        json_input = json.load(json_file)

    main(configuration=json_input)
