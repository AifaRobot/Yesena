import torch

'''
    Las Recompensas Descontadas son una técnica común en el aprendizaje por refuerzo que ayuda a los agentes a aprender de manera más efectiva en 
    entornos donde las recompensas pueden ocurrir en el futuro y pueden estar correlacionadas.

    G_{t} = R_{t+1} + γ * R_{t+2} +γʌ2 * R_{t+3} + … 

    * G_{t} es la recompensa que se descuenta con el paso del tiempo.

    * R_{t+1} es la recompensa en el siguiente paso.

    * 𝛾 es el factor de descuento. Controla la importancia relativa de las recompensas futuras en comparación con las recompensas inmediatas. A
    𝛾 más cercano a 1 indica que las recompensas futuras son más importantes, mientras que un 𝛾 más cercano a 0 indica que las recompensas futuras 
    son menos importantes en relación con las recompensas inmediatas.

    * La suma se extiende hasta el infinito, pero en la práctica suele limitarse a un número finito de pasos futuros.

    La lógica detrás de las recompensas con descuento en el aprendizaje por refuerzo es capturar el concepto de "valor" a largo plazo. 
    En entornos donde las acciones tienen consecuencias a largo plazo y donde las recompensas pueden retrasarse en el tiempo, es importante 
    que un agente considere no sólo las recompensas inmediatas, sino también las recompensas futuras.

    El descuento de recompensas sirve para modelar la preferencia por recibir una recompensa ahora en lugar de en el futuro, debido a factores 
    como la incertidumbre, la posibilidad de que el agente deje de existir o que el entorno cambie. La idea 
    es que una recompensa futura se “descuente” en relación con su retraso temporal y la incertidumbre asociada.

    Por ejemplo, en un entorno de juego, un jugador podría preferir una recompensa de 10 puntos ahora a una recompensa de 15 puntos en el futuro,
    debido a la incertidumbre de si él o ella realmente obtendrá la recompensa futura y el beneficio de tener la recompensa 
    inmediatamente para usarlo en el juego.

    Las recompensas con descuento permiten a un agente considerar las consecuencias a largo plazo de sus acciones y tomar decisiones 
    que maximizan la recompensa total esperada a lo largo del tiempo, teniendo en cuenta la incertidumbre y el retraso en las recompensas futuras.
'''

def discount(rewards, dones, gamma):
    value_targets = []
    old_value_target = 0

    for t in reversed(range(len(rewards))):
        old_value_target = rewards[t] + gamma*old_value_target*dones[t]
        value_targets.append(old_value_target)

    value_targets.reverse()

    return  torch.tensor(value_targets)
