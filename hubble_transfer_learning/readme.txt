The code in this directory will compute the predictions of cutout images, calculate the tSNE data reduction and
then display/interact with the data.

The code requires the installation of some software from pip, those requirements are in the requirements.txt file.

# Code

## Main Code

1. run_transfer_learning.py
  Calculate the predictions for each image and store in a database.

  If one wants to change parameters in order to calculate predictions then this is the place to
  make changes. Probably best to change the name before storage in the database so things are kept
  different.

2. run_tSNE.py
    Calculate the tSNE value for each NN pre-trained model based on predictions from previous code.

    This just reads the predictions and calculates the tSNE values.

3. show_transfer_learning.py
  Display the tSNE and allow interaction between it and the images (very basic first version).


## Helper Code

show_blitzdb.py
  Helper script to make sure the database is created correctly.