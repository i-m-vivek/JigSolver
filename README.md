# JigSolver - A Transformer based Image Jigsaw Solver
![sample](jigsaw_sample.png "Sample of Model Predictions")


## About
Archaeologist usually find broken peices of rare historic items after that they need to arrange the pieces in a paricular order that makes some sense but human mind is somewhat baised about the things that we see usually in our life thus in this project we aim to make a data-driven machine learning model that can solve 3\*3 jigsaw puzzles. For this, we use a standard CNN model for extracting features from different regions of the image. After that, the Transformer model is used to relate distinct image regions with each other. Finally, the output of the Transformer is passed to an MLP layer that predicts the correct position for the image regions. 

 
#### To Do 
- [x] Upload pretrained models
- [ ] Add data generation script
- [ ] Make a GUI for creating and automatic solving the  jigsaw. 
