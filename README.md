# JigSolver - A Transformer based Image Jigsaw Solver
![sample](jigsaw_sample.png "Sample of Model Predictions")


## About
Archaeologist usually find broken peices of rare historic items after that they need to arrange the pieces in a paricular order that makes some sense but human mind is somewhat baised about the things that we see usually in our life thus in this project we aim to make a data-driven machine learning model that can solve 3\*3 jigsaw puzzles. For this, we use a standard CNN model for extracting features from different regions of the image. After that, the Transformer model is used to relate distinct image regions with each other. Finally, the output of the Transformer is passed to an MLP layer that predicts the correct position for the image regions. 

### Training 
Get the data from [Kaggle](https://www.kaggle.com/c/imet-2020-fgvc7). Put this zip file in data folder, then run the following to setup the data and start training. 
```
chmod +x ./data_setup
./data_setup
python train.py
```

### Demo
If you want to play with the model, you can look at this [notebook](https://github.com/i-m-vivek/JigSolver/blob/master/Pretrained%20Model%20Demo.ipynb). Make sure you download the pretrained model from my [GDrive](https://drive.google.com/file/d/1WUTiIvY0B3CH9GBXIociUa53DdIUiyo9/view?usp=sharing) & follow the next two steps to setup data.
```
chmod +x ./data_setup
./data_setup
```
#### Contribution 
If you would like to add some more features and do some more experiments, I can help you in getting started just open a new issue and share your ideas :)
