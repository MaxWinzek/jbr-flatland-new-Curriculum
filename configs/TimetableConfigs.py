from configs.Config import Config

import torch
from agent.judge.Judge import Judge, DeterministicLaunchCap

class TimeTableConfig(Config):
    def __init__(self):
        pass

    def create_timetable(self):
        pass

class JudgeConfig(TimeTableConfig):
    def __init__(self, window_size_generator, lr, batch_size, optimization_epochs):
        self.window_size_generator = window_size_generator
        self.lr = lr
        self.batch_size = batch_size
        self.optimization_epochs = optimization_epochs

    def create_timetable(self):
        return Judge(self.window_size_generator, self.lr, self.batch_size, self.optimization_epochs, torch.device("cpu"))


class LaunchCapConfig(TimeTableConfig):
    def __init__(
            self,
            initial_fraction=0.4,
            min_active=4,
            cap_increment=1,
            ramp_interval=40,
            min_agents_to_limit=12,
            shuffle=True,
        ):
        self.initial_fraction = initial_fraction
        self.min_active = min_active
        self.cap_increment = cap_increment
        self.ramp_interval = ramp_interval
        self.min_agents_to_limit = min_agents_to_limit
        self.shuffle = shuffle

    def create_timetable(self):
        return DeterministicLaunchCap(
            initial_fraction=self.initial_fraction,
            min_active=self.min_active,
            cap_increment=self.cap_increment,
            ramp_interval=self.ramp_interval,
            min_agents_to_limit=self.min_agents_to_limit,
            shuffle=self.shuffle,
        )
