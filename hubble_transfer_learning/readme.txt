The code in this directory will compute the predictions of cutout images, calculate the tSNE data reduction and
then display/interact with the data.

# Code

## Main Code

run_transfer_learning.py
  Calculate the predictions for each image and store in a database

run_tSNE.py
  Calculate the tSNE value for each NN pre-trained model based on predictions from previous code.

show_transfer_learning.py
  Display the tSNE and allow interaction between it and the images (very basic first version).


## Helper Code

show_blitzdb.py
  Helper script to make sure the database is created correctly.