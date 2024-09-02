Note:

The data inside of this folder ("ARAR") are organized as follows: 

Folders called "sample_1" through sample "sample_10" correspond to the 
50 Epoch (FCM on, broad-entropy training set) LifeGPT model, set to 
temperature=0. These contain data for 249 iterations (250 game states including ICs).

The sample numbers correspond to testing set samples (in order from 1 to 10):
"Order Param: 0", "Order Param: 0.25", "Order Param: 0.5", "Order Param: 0.75", "Order Param: 1",
"Glider", "Cloverleaf", "Blinkers", "Hammerhead Spaceship", "r-Pentomino".

The "10_iterations" folder contains two subfolders, "Epoch 50" and "Epoch 16". 

These correspond to Epoch 50 and 16 versions of the
same LifeGPT model as mentioned above.

However, these folders containing data for only 9 iterations
 (10 game states including ICs).

This is because each folder contains data for 5 different temperatures,
across 10 samples in the testing set. 

These data are referene by the files
 ARAR_9_iterations.ipynb and ARAR_249_iterations.ipynb,
 that are used for creating GIFs showing GT vs. predicted game
states, and the corresponding error grid.

