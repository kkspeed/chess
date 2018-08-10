Chinese chess bot training.
---

The bot is supposed to be trained with reinforcement learning.

## Structural overview

```
model.py: defines the NN model. Currently it's Conv2D + Dense with max pooling and dropouts
agent.py: defines the self-play agent.
train.py: trains the model.
          python train.py    # trains from gen 1
          python train.py <your generation> # trains from the generation. Plz ensure model_<your generation>.h5 is present.
debug_test.py: tests the bot. 
          python debug_test.py <model1.h5> <model2.h5> # compares model1 (red) vs model2 (black)
visualize.py: visualize a game for debug purpose.
          python visualize.py 1/0_1.h5   # visualize moves in gen 1, game 0, player red (1). Note that 0_2.h5 is the black player but the
                                         # board still shows red. It's recorded from red side perspective.
                                         # In the visualization tool, use LEFT ARROW and RIGHT ARROW to navigate.
chess_types.py: defines basic pieces functionalities and rules.
chess_types_unittest.py: tests the functionalities
```

The bot depends on Keras (preferrably with tensorflow backend), h5py and pygame (for visualization).

## Tuning Params:
```
Learning rate, batch size, epochs:  train_batch function in agent.py
Randomness: clip_probs function in agent.py
Number of game plays for each training: __main__ in train.py
Rewards: self_play function in train.py. Currently it's rather naive: +1 for each step that leads to winning. -1 for each step that leads to failing. -2 for the last step for losing to prevent bot from committing suicide.
```
