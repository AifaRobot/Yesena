import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class Logger():
    def __init__(self, method_name, env_name):
        self.method_name = method_name
        self.env_name = env_name
        self.path_name = 'plots'

        self.history_loss_curiosity = []
        self.history_loss_actor = []
        self.history_loss_critic = []
        self.history_loss_actor_critic = []
        self.history_extrinsic_rewards = []
        self.history_intrinsic_rewards = []

    def update_metrics(self, dict_metrics):
        if('intrinsic_rewards' in dict_metrics):
            self.history_intrinsic_rewards.append(np.array(dict_metrics['intrinsic_rewards']).mean())
        if('extrinsic_rewards' in dict_metrics):
            self.history_extrinsic_rewards.append(np.array(dict_metrics['extrinsic_rewards']).sum())
        if('curiosity_losses' in dict_metrics):
            self.history_loss_curiosity.append(np.array(dict_metrics['curiosity_losses']).mean())
        if('actor_losses' in dict_metrics):
            self.history_loss_actor.append(np.array(dict_metrics['actor_losses']).mean())
        if('critic_losses' in dict_metrics):
            self.history_loss_critic.append(np.array(dict_metrics['critic_losses']).mean())
        if('actor_critic_losses' in dict_metrics):
            self.history_loss_actor_critic.append(np.array(dict_metrics['actor_critic_losses']).mean())

    def draw_plot(self, title, xlabel, ylabel, title_file, history_score, color = "red"):
        new_array = []
      
        for i in range(len(history_score)):
            new_array.append(np.array(history_score[max(0, i-100):(i+1)]).mean())

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(new_array, color)
        plt.title(title)
        plt.savefig(title_file)
        plt.close()

    def draw_plots(self):
        if(os.path.exists(self.path_name + '/' + self.method_name + '-' + self.env_name) == False):
            os.mkdir(self.path_name + '/' + self.method_name + '-' + self.env_name) 

        now = datetime.now()
        path = self.path_name + '/' + self.method_name + '-' + self.env_name + '/' + now.strftime("%d%m%Y%H%M%S")

        os.mkdir(path)

        if(len(self.history_intrinsic_rewards) > 0):
            self.draw_plot("Recompensa Intrinseca Historial", "Ciclos", "Recompensa", path + "/recompensa_intrinseca_historial.png", self.history_intrinsic_rewards)
        if(len(self.history_extrinsic_rewards) > 0):
            self.draw_plot("Recompensas Extrinseca Historial", "Ciclos", "Recompensa", path + "/recompensa_extrinseca_historial.png", self.history_extrinsic_rewards)
        if(len(self.history_loss_curiosity) > 0):
            self.draw_plot("Perdidas Curiosity", "Ciclos", "Perdida", path + "/perdida_curiosity_historial.png", self.history_loss_curiosity)
        if(len(self.history_loss_actor) > 0):
            self.draw_plot("Perdidas Actor", "Ciclos", "Perdida", path + "/perdida_actor_historial.png", self.history_loss_actor)
        if(len(self.history_loss_critic) > 0):
            self.draw_plot("Perdidas Critic", "Ciclos", "Perdida", path + "/perdida_critic_historial.png", self.history_loss_critic)
        if(len(self.history_loss_actor_critic) > 0):
            self.draw_plot("Perdidas Actor Critic", "Ciclos", "Perdida", path + "/perdida_actor_critic_historial.png", self.history_loss_actor_critic)
