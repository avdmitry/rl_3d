### Reinforcement learning in 3D.
Implemented base DQN [1] algorithm for ViZDoom [2] and DeepMind Lab [3] environments.

Small network on small screen resolution trained relatively fast on simple maps (on 1 GPU):
* ~ 5 minutes on ViZDoom map *simpler_basic*.
* ~ 5 hours on DeepMind Lab map *seekavoid_arena_01*.

**ViZDoom map simpler_basic.**

[![ViZDoom map simpler_basic](http://i.imgur.com/zInpPnW.png)](https://youtu.be/mgY-G8rl9O4)

**DeepMind Lab map seekavoid_arena_01.**

*Video coming soon.*


### Main Dependencies.

* numpy
* [opencv](https://github.com/opencv/opencv)
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [ViZDoom](https://github.com/mwydmuch/ViZDoom)
* [DeepMind Lab](https://github.com/deepmind/lab)


### References.
[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. Playing Atari with Deep Reinforcement Learning, 2013.

[2] Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, Wojciech Jaśkowski. ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, 2016.

[3] Charles Beattie, Joel Z. Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, Julian Schrittwieser, Keith Anderson, Sarah York, Max Cant, Adam Cain, Adrian Bolton, Stephen Gaffney, Helen King, Demis Hassabis, Shane Legg, Stig Petersen. DeepMind Lab, 2016.

