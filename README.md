# gym_example

Example implementation of an [OpenAI Gym](http://gym.openai.com/) environment,
to illustrate problem representation for [RLlib](https://rllib.io/) use cases.

## Usage

Clone the repo and connect into its top level directory.

To initialize and run the `gym` example:

```
pip install -r requirements.txt
pip install -e gym-example

python sample.py
```

To run Ray RLlib to train a policy based on this environment:

```
python train.py
```


## Kudos

h/t:

  - <https://github.com/DerwenAI/gym_trivial>
  - <https://github.com/DerwenAI/gym_projectile>
  - <https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/>
  - <https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa>
  - <https://github.com/openai/gym/blob/master/docs/creating-environments.md>
  - <https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai>
