from __future__ import division
from __future__ import print_function

import itertools as it

from vizdoom import *

class EnvVizDoom(object):
    def __init__(self, config_file_path):
        print("Initializing doom.")
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.RGB24) # ScreenFormat.GRAY8
        self.game.set_screen_resolution(ScreenResolution.RES_640X480) # ScreenResolution.RES_640X480
        self.game.init()
        print("Doom initialized.")

        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.num_actions = len(self.actions)
        print(self.num_actions)

    def NumActions(self):
        return self.num_actions

    def Reset(self):
        self.game.new_episode()

    def Act(self, action, frame_repeat):
        return self.game.make_action(self.actions[action], frame_repeat)

    def IsRunning(self):
        return (not self.game.is_episode_finished())

    def Observation(self):
        return self.game.get_state().screen_buffer

    def MapActions(self, action_raw):
        return action_raw

