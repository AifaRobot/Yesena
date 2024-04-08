import vizdoom as vzd
import numpy as np
import os

class Vizdoomenvmywayhome:
    def __init__(self, render = True):
        self.game = vzd.DoomGame()

        self.game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "my_way_home.wad"))
        self.game.set_doom_map("map01")
        #self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        #self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)
        self.game.set_render_messages(False)
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True)

        self.game.set_available_buttons(
            [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT]
        )

        self.game.set_available_game_variables([vzd.GameVariable.AMMO2])

        self.game.set_episode_timeout(600)
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(render)
        #self.game.set_living_reward(-1)
        self.game.set_living_reward(0)
        self.game.set_mode(vzd.Mode.PLAYER)

        self.game.init()

        self.actions = [[True, False, False], [False, True, False], [False, False, True]]

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = np.zeros([120, 160, 3])
        
        if(reward == -5):
            reward = -1
        elif(reward > 0):
            reward = 1

        if done:
            state = np.zeros([120, 160, 3], dtype=np.uint8)
        else:
            state = self.game.get_state()
            state = state.screen_buffer

        return state, reward, done, []

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        state = state.screen_buffer
        return state

    def close(self):
        self.game.close()

