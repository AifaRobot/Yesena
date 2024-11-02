import torch
from rewards import Reward
from rewards.utils import discount

class Advantage(Reward):
    def __init__(self, gamma, tau, normalize = False):
        self.gamma = gamma
        self.tau = tau
        self.normalize = normalize

    '''
        La Estimación de Ventaja Generalizada (GAE) es un algoritmo utilizado en el aprendizaje por refuerzo para lograr con mayor precisión 
        estimar rendimientos descontados y valores estatales en un proceso de decisión de Markov (MDP).

        La ventaja en el aprendizaje por refuerzo se refiere a cuánto mejor es una acción específica en comparación con otras 
        posibles acciones en un estado dado. La idea básica detrás de GAE es combinar el uso de múltiples pasos de tiempo. 
        para estimar las ventajas con un factor de descuento, similar a cómo se calculan las recompensas con descuento. Esto permite 
        una estimación más estable y eficiente de la ventaja que tener en cuenta sólo un paso en el tiempo.

        La recompensa de un determinado paso de tiempo no sólo se ve afectada por la acción actual, sino también por todas las acciones futuras realizadas. 
        Esto se debe a que es posible que incluso si se realiza una acción con una buena recompensa en un determinado paso de tiempo, en el futuro 
        Es posible que las acciones no devuelvan tan buenas recompensas, lo que a largo plazo llevaría a un peor desempeño del agente. 
        Por el contrario, puede suceder que realizar una acción en un determinado momento devuelva una mala recompensa, pero esas acciones 
        futuras obtienen buenas recompensas, lo que a la larga conduciría a un mejor desempeño del agente.

        A^GAE_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}

        * A^GAE_t es la ventaja estimada en el paso de tiempo t.

        * γ es el factor de descuento.

        * λ es un parámetro que controla el equilibrio entre sesgo y varianza.

        * δ_{t+l} es el error temporal o error TD en el paso de tiempo t+l, calculado como
        δ_{t+l} = r_{t+l} + γ * V(s_{t+l+1}) − V(s_{t+l}), donde r_{t+l} es la recompensa en el paso de tiempo t+l y
        V es la función de valor de estado.
    '''

    def calculate_generalized_advantage_estimate(self, rewards, values, dones):
        delta_t = rewards + self.gamma * values[1:] * dones - values[:-1] # 1

        advantages = discount(delta_t, dones, self.gamma*self.tau)

        if self.normalize:
            advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

        return advantages

'''
    1. El error temporal se utiliza para estimar en qué medida difiere la recompensa real obtenida en un paso de tiempo específico. 
    de la recompensa esperada según la estimación actual del agente. Es parte fundamental del refuerzo. 
    algoritmos de aprendizaje como TD-learning o Q-learning.

    La fórmula general para el error temporal en un paso de tiempo t es:
    
    δ_{t} = r_{t} + γ * V(s_{t+1}) − V(s_{t} )

    Donde:

    * δ_{t} es el error temporal en el paso de tiempo t.

    * r_{t} es la recompensa obtenida en el paso de tiempo t.

    * γ es el factor de descuento.

    * V(s_{t+1}) es la estimación de la función de valor del siguiente estado s_{t+1}.

    * V(s_{t}) es la estimación de la función de valor del estado actual s_{t}.
''' 