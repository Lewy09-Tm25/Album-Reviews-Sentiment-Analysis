# Pitchfork Album Reviews Sentiment Analysis

The exploration of art and the proliferation of technology has allowed for widespread discussions about the evaluation of art through ML-assisted methods. With the expansion of Natural Language Processing, the evaluation of art has inevitably been numericalized, and this has its own accompaniment of danger. Reducing art to a number must be done with great care as this leaves a lasting impression to the audience and artist.
This project focuses on written music reviews, sourced from Pitchfork.com, a popular online music magazine. Pitchfork music critics publish a written review of an album, accompanied by a numerical rating. In an ideal landscape, the qualitative written review is consistent with its quantitative counterpart (the numerical rating). The goal of this project is to better understand the correspondence between written and numerical music criticism, through the use of sentiment analysis and machine learning methods.

To use any of these files, please execute the command
```
pip install -r requirements.txt
```
The directory is as follows:
- **/code** contains the source code for the project
- **/dataset** contains .parquet files required for the entire project pipeline
- **/images** contains the graphs and figures giving insights on the analysis of the dataset, and models
- **/metrics** contains the results of the model training in terms of the evaluation metrics of MSE, MAE, and R2
- **/models** contains machine learning models as pickled files for training and evaluation
- **/requirements.txt** text file containing Python packages required to run the project
