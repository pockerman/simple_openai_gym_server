# Simple OpenAI Gym Server

A simple server implementation for serving OpenAI Gym environments. Each ```gy.Env```
is represented with a server (see the ```servers_impl``` directory). Applications
can communicate with the ```gym.Env``` instance via the usual request/response pattern.
Much of the implementation is take from https://github.com/Omegastick/pytorch-cpp-rl (Thanks a lot)


## Dependencies

- OpenAI Gym
- msgpack
- zmq

## How to use

Assuming that all the dependencies are install

```
python launch_gym_server.py --config "path/to/configuration/file
```

## Adding a new server
