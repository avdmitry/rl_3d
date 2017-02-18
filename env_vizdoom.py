from __future__ import division
from __future__ import print_function

import itertools as it

from vizdoom import *

class EnvVizDoom(object):
    def __init__(self, scenario_path):
        print("Initializing doom.")
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario_path)
        self.game.set_doom_map("map01")
        #self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_format(ScreenFormat.RGB24)
        #self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_render_hud(True) # False
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        #self.game.add_available_game_variable(GameVariable.AMMO2)
        #self.game.add_available_game_variable(GameVariable.POSITION_X)
        #self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.set_episode_timeout(300)
        self.game.set_episode_start_time(14) # 10 20
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
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
        action = self.MapActions(action)
        return self.game.make_action(self.actions[action], frame_repeat)

    def IsRunning(self):
        return (not self.game.is_episode_finished())

    def Observation(self):
        return self.game.get_state().screen_buffer

    def MapActions(self, action_raw):
        return action_raw
