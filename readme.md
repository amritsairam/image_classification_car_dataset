Car Model Image Classification

Overview

This project involves analyzing, preprocessing, and preparing car model images for a machine learning classification task. The workflow includes exploratory data analysis (EDA) to understand the dataset better, followed by preprocessing steps to organize and resize the images, wrapped up finally by using the cleaned dataset for training imagenet model to try getting the highest accuracy possible.

Project Structure

Datasets:

1.datasets/Car names and make.csv: Contains information about different car models and manufacturers.

2.datasets/train_8143_images.csv: A dataset with training images related to car models.

3.datasets/train/: Directory containing the raw training images, organized by car model.

4.datasets/val/: Directory containing validation images.

5.datasets/val_resized_224/: Directory where the resized validation images are saved after preprocessing.

Notebooks:

EDA.ipynb: Contains the exploratory data analysis of the car dataset.

data_preprocessing.ipynb: Preprocessing steps for counting images per car model in the training dataset.

data_preprocessing_part2.ipynb: Resizes validation images to a resolution of 224x224 while preserving their aspect ratio.
Installation


Exploratory Data Analysis (EDA)

Loading the Data:

The car names and make dataset is loaded using Pandas.
The training dataset containing image information is also loaded for analysis.


Visualizing Data Distribution:
The distribution of car model images is visualized to understand how well-represented each model is in the dataset.

Data Preprocessing

Part 1: Image Count and Organization

Folder Setup:
The script processes the train directory, where images are organized into subfolders based on car models.

Counting Images:

A function count_images_in_folder() is used to count the number of images for each car model and store the result in a dataframe.

Saving the Count Data:
The dataframe is saved as image_counts_by_folder.csv, which contains the car model names and the number of corresponding images.This data is used for finding out the imbalanced car images folders.

Part 2: Resizing Images

Resizing with Aspect Ratio:
The resize_with_aspect_ratio() function resizes images in the train and val  directory to a resolution of 224x224 pixels while preserving their aspect ratio.
Resized images are saved in their specific  directory with a white background to fill any empty space.

Maintaining Folder Structure:
The car model subfolders are preserved during the resizing process.



Transfer Learning Experiments: Without Fine-Tuning
Overview
This part of the project explores transfer learning using CNN models like EfficientNetB0 and InceptionV3. The models' pre-trained layers remain frozen, and only the top layers are retrained on the new car model dataset.


Models Used
VGG16
Resnet50
EfficientNetB0
InceptionV3 (GoogleNet)
Training Parameters
Epochs: 30
Batch Size: Defined in the notebook.
Optimizer: Adam (learning rate: 0.0001).
Loss Function: Categorical Crossentropy.
Results (Without Fine-Tuning)
The models initially were highly overfitting , just by chopping of the last layer. the training accuracies were around 99.5% and the test was around 27-30%

Adding a dense and a dropout layer to avoid overfitting resulted in a highly underfitting model , denoting the model was not learning anything significant even after going through several epoch. 

The above two meant that we had to fine tune the model and train even the last few layers of the model which will help the model to learn something significant.


Transfer Learning Experiments: With Fine-Tuning
Overview
This section extends the transfer learning experiments by fine-tuning the pre-trained models. After training the top layers, the last layers of the pre-trained network are unfrozen for further optimization.


Unfreezing the Last 10 Layers:

Initially, the last 10 layers of the pre-trained models were unfrozen for further optimization.

Unfreezing the Last 30 Layers:
After testing, unfreezing the last 30 layers of EfficientNetB0 yielded better results.

Training Parameters

Epochs: 30

Batch Size: Defined in the notebook.

Optimizer: Adam (learning rate: 0.0001).

Loss Function: Categorical Crossentropy.

Callbacks Used

ModelCheckpoint: Saves the model with the best validation loss.

EarlyStopping: Stops training when validation loss stops 
improving.

ReduceLROnPlateau: Reduces learning rate when validation loss stagnates.

Results (With Fine-Tuning)

EfficientNetB0:

Test accuracy: 77.02%

Train accuracy: 97.52%

InceptionV3:

Test accuracy: 68.24%

Train accuracy: 92.34%

Conclusion:

This project successfully demonstrates the effectiveness of transfer learning after experimenting with several differnet methods and epochs we can conclude fine-tuning combined with transfer learning helped in improving the classification accuracy of car model images. Fine-tuning significantly improves the performance of EfficientNetB0, making it the best-performing model. with a test accuracy of around 80%, it proves to be a very robust model even amongst a highly challenging dataset.


