# 2023--MLP-NN

# Code from the article, "Characterization and optimization of 5’ untranslated region containing poly-adenine tracts in *Kluyveromyces marxianus* using machine-learning model"

## introduction
You can find the code used in our article about construction of Multi-layer perceptron neural network (MLP-NN) model using a mini-library from *Kluyveromyces marxianus* and  how to perform prediction of the model and SHapley Additive exPlanation (SHAP) sensitivity analysis of features.

## Note

Firstly, each Python file, along with its corresponding calculated data file(s), should be placed in separate folders. The packages tensorflow v 2.12.0 and shap v 0.41.0 should be installed correctly.
  
To construct the MLP-NN model, you will need the following files: "construction and train for MLP-NN model.py" and "all.txt". The "all.txt" file contains a table that includes the feature values and relative GFP abundance of all samples, excluding the header (the feature names). You can change the time of repeating training by changing the value of the parameter *time_of_repeating_training*. It will require you to enter "0" to only perform a training-test split for further optimization of hyper-parameters through cross-validation using the train set as discribed below, or to enter "1" to proceed with the training process after you changing the hyper-paramters to the most optimal after that.

To perform n-fold cross-validation, the "train.txt" obtained from running "Construction and train for MLP-NN.py" needs to be placed in the same folder as "cross-validation.py", which then is being run directly. After running, the "result of cross-validation.txt" will contain the R2 for each validation set prediction, which is used to evaluate the performance of the hyperparameters. Once the optimal parameters are obtained, you can modify the setup of the model in "Construction and train for MLP-NN.py" and then input "1" to begin training the model.
  
For prediction, you will need the following files: "prediction through model.py", "model.h5" (the selected file that contains the saved weights), and "input.txt". The "input.txt" file contains a table of new 5' UTR sequences' feature values, where the feature rank remains consistent with the previous data.
  
For sensitivity analysis, the following files are required: "SHAP_sensitivity_analysis.py", "model.h5", and "input.txt". The "input.txt" file contains a feature value table with a header (where the feature names are listed as the first row) from the training set. The header will facilitate the plot of SHAP sensitivity analysis.

## Code
[MLP-NN model](https://github.com/CODdown/2023--MLP-NN/tree/main/Code/construction%20and%20train%20for%20MLP-NN.py): Python code used to construct and train the MLP-NN model  
[cross-validation](https://github.com/CODdown/2023--MLP-NN/tree/main/Code/cross-validation.py): Python code used to perform n-fold cross-validation  
[prediciton](https://github.com/CODdown/2023--MLP-NN/tree/main/Code/prediction%20through%20model.py): Python code used to predict the GFP abundacne of new sequence with new calculated feature values using selected and saved model  
[sensitivity analysis](https://github.com/CODdown/2023--MLP-NN/tree/main/Code/SHAP%20sensitivity%20analysis.py): Python code used to perform the SHAP sensitivity analysis of features of train set  
