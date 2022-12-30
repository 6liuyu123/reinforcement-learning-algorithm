'''
The environment most comes from MorvanZhou.
'''
import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40
MAZE_H = 4
MAZE_W = 4

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_states = MAZE_H * MAZE_W
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg = 'white', height = MAZE_H * UNIT, width = MAZE_W * UNIT)
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for c in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, c, MAZE_W * UNIT, c
            self.canvas.create_line(x0, y0, x1, y1)
        origin = np.array([20, 20])
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0] - 15, hell1_center[1] - 15, hell1_center[0] + 15, hell1_center[1] + 15, fill = 'black')
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0] - 15, hell2_center[1] - 15, hell2_center[0] + 15, hell2_center[1] + 15, fill = 'black')
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(oval_center[0] - 15, oval_center[1] - 15, oval_center[0] + 15, oval_center[1] + 15, fill = 'yellow')
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill = 'red')
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(0.2)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill = 'red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        observation = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if observation[1] > UNIT:
                base_action[1] -= UNIT
        if action == 1:
            if observation[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        if action == 2:
            if observation[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        if action == 3:
            if observation[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])
        next_observation = self.canvas.coords(self.rect)
        if next_observation == self.canvas.coords(self.oval):
            reward = 1
            done = True
            next_state = 'terminal'
        elif next_observation in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward =  -1
            done = False
            next_state = 'terminal'
        else:
            reward = 0
            done = False
            next_state = next_observation
        return next_state, reward, done
    
    def render(self):
        time.sleep(0.02)
        self.update()
