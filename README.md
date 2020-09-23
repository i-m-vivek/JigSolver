# JigSolver
Transformer based Image Jigsaw Solver

## About
This project aims to make a data-driven machine learning that can solve 3\*3 jigsaw puzzles. For this, we used a standard CNN model for extracting features from different regions of the image. After that, the Transformer model is used to relate distinct image regions with each other. Finally, the output of the Transformer is passed to an MLP layer that predicts the correct position for the image regions. 
