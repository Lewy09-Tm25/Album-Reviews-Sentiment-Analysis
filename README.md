# Pitchfork Album Reviews Sentiment Analysis

Art criticism, pivotal in shaping cultural perceptions, merges qualitative evaluation with numerical ratings, like those on Pitchfork.com. As technology and Natural Language Processing expand, caution is vital when quantifying art, as it profoundly impacts artists and audiences. Critics' influence on mainstream culture underscores the importance of a balanced correlation between qualitative and quantitative assessments in artistic evaluation. This project delves into music reviews to analyze the alignment between textual sentiments and numerical ratings. 

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