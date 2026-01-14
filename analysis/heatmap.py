#!/usr/bin/env python3.8

import numpy as np
import seaborn as sns

class Heatmap:
    def __init__(self, height=240, width=480):
        self.height = height
        self.width = width

        self.data = np.array([np.zeros(self.width) for _ in range(self.height)])

    def add_point(self, px, py):
        y = int(px*self.width)
        x = int(py*self.height)
        self.data[x][y] += 1

    def get_percent(self):
        s = 0
        for vect in self.data:
            s += np.sum(vect)

        data_percent = np.array([np.zeros(self.width) for _ in range(self.height)])
        for x in range(self.height):
            for y in range(self.width):
                data_percent[x][y] = 100 * float(self.data[x][y] / s) if s > 0 else 0

        return data_percent

def map_viz(data, ax=None, min=0, max=0.02):
    return sns.heatmap(
        data,
        linewidth=0,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cmap="coolwarm",
        cbar=False,
        vmin=min,vmax=max,
        ax=ax)

def map_mean(maps, height=240, width=480):
    data = np.array([np.zeros(width) for _ in range(height)])
    n = 0
    for map in maps:
        n+=1
        for y in range(width):
            for x in range(height):
                data[x][y] += map[x][y]
    for y in range(width):
        for x in range(height):
            data[x][y] = float(data[x][y]/n)

    return data
