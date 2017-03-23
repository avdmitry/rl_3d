### Reinforcement learning in 3D

Implemented DQN [3] and A3C [4] algorithm for ViZDoom [1] and DeepMind Lab [2] environments.


Small network on small screen resolution trained relatively fast on simple maps:
* DQN on 1 GPU: ~ 5 minutes on ViZDoom map *simpler_basic*.
* DQN on 1 GPU: ~ 5 hours on DeepMind Lab map *seekavoid_arena_01*.
* A3C on 1 CPU, 3 threads: ~13 minutes on ViZDoom map *simpler_basic*.
* A3C on 1 GPU, 3 workers: ~8 minutes on ViZDoom map *simpler_basic*.


_**DQN, ViZDoom map simpler_basic**_

[![ViZDoom map simpler_basic](http://i.imgur.com/zInpPnW.png)](https://youtu.be/mgY-G8rl9O4)

_**DQN, DeepMind Lab map seekavoid_arena_01**_

[![ViZDoom map simpler_basic](http://i.imgur.com/nDLoaNW.png)](https://youtu.be/G41s4FQPIX4)


### Dependencies

* numpy
* [opencv](https://github.com/opencv/opencv)
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [ViZDoom](https://github.com/mwydmuch/ViZDoom)
* [DeepMind Lab](https://github.com/deepmind/lab)

### How to run
_**ViZDoom**_
* Install [ViZDoom](https://github.com/mwydmuch/ViZDoom) and other dependencies
* Set path to it in variable *vizdoom_path*
* Set variable *lab* to *False*
* Set path to rl_3d in *path_work_dir*
* Run:
  * DQN: *./agent_dqn.py --gpu 0*
  * A3C: *./agent_a3c.py*

_**DeepMind Lab**_
* Install [DeepMind Lab](https://github.com/deepmind/lab) and other dependencies
* Set variable *lab* to *True*
* Set path to rl_3d in *path_work_dir*
* For now I used DeepMind Lab build and run system through bazel, so add build rule to *lab_path*/BUILD (change *path_work_dir* to your rl_3d path):
```
py_binary(
    name = "agent_dqn",
    srcs = ["*path_work_dir*/agent_dqn.py"],
    data = [":deepmind_lab.so"],
    main = "*path_work_dir*/agent_dqn.py",
)

py_binary(
    name = "agent_a3c",
    srcs = ["*path_work_dir*/agent_a3c.py"],
    data = [":deepmind_lab.so"],
    main = "*path_work_dir*/agent_a3c.py",
)
```
* From *lab_path* run:
  * DQN: *bazel run :agent_dqn -- --gpu 0*
  * A3C: *bazel run :agent_a3c*

### Thanks
A3C is a little bit tricky algorithm and there are a lot of it's implementations already. So as reference I used implementation by [Arthur Juliani](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb).

### References
[1] Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, Wojciech Jaśkowski. ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning. arXiv:[1605.02097](https://arxiv.org/abs/1605.02097), 2016.

[2] Charles Beattie, Joel Z. Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, Julian Schrittwieser, Keith Anderson, Sarah York, Max Cant, Adam Cain, Adrian Bolton, Stephen Gaffney, Helen King, Demis Hassabis, Shane Legg, Stig Petersen. DeepMind Lab. arXiv:[1612.03801](https://arxiv.org/abs/1612.03801), 2016.

[3] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. Playing Atari with Deep Reinforcement Learning. arXiv:[1312.5602](https://arxiv.org/abs/1312.5602), 2013.

[4] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. arXiv:[1602.01783](https://arxiv.org/abs/1602.01783), 2016.

