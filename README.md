# 2023--FCNN

# Code from the article, "Characterization and optimization of 5’ untranslated region containing poly-adenine tracts in *Kluyveromyces marxianus* using machine-learning model"

## introduction
You can find the code used in our article about construction of Full Connected Neural Network (FCNN) model using a mini-library from *Kluyveromyces marxianus* and  how to perform prediction of the model and SHapley Additive exPlanation (SHAP) sensitivity analysis of features.

First of all, each python file with its required calculated data file(s) should be placed in one folder separately. The construction of FCNN model needs "construction and train for FCNN.py" and "all.txt" containing the sheet comprising features values and relative GFP abundance of all samples without

## Note



## Code
[FCNN model](https://github.com/CODdown/2023--FCNN/tree/main/Code/construction%20and%20train%20for%20FCNN.py): Python code used to construct and train the FCNN model  
[prediciton](https://github.com/CODdown/2023--FCNN/tree/main/Code/prediction%20through%20model.py): Python code used to predict the GFP abundacne of new sequence with new calculated feature values using selected and saved model  
[sensitivity analysis](https://github.com/CODdown/2023--FCNN/tree/main/Code/SHAP%20sensitivity%20analysis.py): Python code used to perform the SHAP sensitivity analysis of features of train set  
