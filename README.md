# Yesena

Yesena es un marco de trabajo para principiantes que te permitirá aprender los principales métodos del Aprendizaje Reforzado.

## Tabla de Contenidos
- [Descripción](#descripción)
- [Instalación](#instalación)
- [Ejemplos](#ejemplos)

## Descripción

Este proyecto tiene documentado todos sus archivos con descripciones claras y precisas de sus componentes, para que puedas aprender cual es el sentido de cada una de las partes necesarias para resolver un problema de aprendizaje reforzado.

También posee muchas pruebas para que te familiarices con su funcionamiento.

## Instalación

```bash
git clone https://github.com/AifaRobot/Yesena.git
cd Yesena
pip install -r requirements.txt
```
## Ejemplo

```python
import argparse
from methods.ppo import PPO
from models import ActorCriticFactory3
from workers import WorkerFactoryGymLunarLander
from tests import GymTestLunarLander

    
args = argparse.ArgumentParser()

env_name = 'LunarLander-v3'

args.add_argument('-lr', type=float, default=0.0001)
args.add_argument('-gamma', type=float, default=0.99)
args.add_argument('-value_coeficient', type=float, default=0.5)
args.add_argument('-entropy_coeficient', type=float, default=0.01)
args.add_argument('-clip', type=float, default=0.1)
args.add_argument('-minibatch_size', type=int, default=32)
args.add_argument('-batch_size', type=int, default=128)
args.add_argument('-epochs', type=int, default=4)
args.add_argument('-train', choices=('True','False'), default='True')
args.add_argument('-episodes', type=int, default=2000)
args.add_argument('-lam', type=float, default=1.0)
args.add_argument('-in_channels', type=int, default=8)
args.add_argument('-n_actions', type=int, default=4)
args.add_argument('-num_processes', type=int, default=5)
args.add_argument('-alpha', type=float, default=0.1)
args.add_argument('-beta', type=float, default=0.2)

arguments = args.parse_args()

method = PPO(
    env_name = env_name,
    main_model_factory = ActorCriticFactory3(arguments.in_channels, arguments.n_actions),
    worker_factory = WorkerFactoryGymLunarLander(env_name, arguments.in_channels, arguments.batch_size),
    test_factory = GymTestLunarLander(env_name, arguments.in_channels, arguments.batch_size, True),
    arguments = arguments,
)
    
if(arguments.train == 'True'):
    print('Modo: Entrenamiento')
    method.train()
else:
    print('Modo: Testeo')
    method.test()
```