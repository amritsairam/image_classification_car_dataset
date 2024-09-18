Car Model Image Classification

Overview

This project involves analyzing, preprocessing, and preparing car model images for a machine learning classification task. The workflow includes exploratory data analysis (EDA) to understand the dataset better, followed by preprocessing steps to organize and resize the images.

Project Structure

Datasets: datasets/Car names and make.csv: Contains information about different car models and manufacturers. datasets/train_8143_images.csv: A dataset with training images related to car models. datasets/train/: Directory containing the raw training images, organized by car model. datasets/val/: Directory containing validation images. datasets/val_resized_224/: Directory where the resized validation images are saved after preprocessing. Notebooks: EDA.ipynb: Contains the exploratory data analysis of the car dataset. data_preprocessing.ipynb: Preprocessing steps for counting images per car model in the training dataset. data_preprocessing_part2.ipynb: Resizes validation images to a resolution of 224x224 while preserving their aspect ratio. Installation

To replicate this project, install the necessary dependencies:

bash Copy code pip install pandas matplotlib seaborn numpy Pillow Exploratory Data Analysis (EDA)

Loading the Data: The car names and make dataset is loaded using Pandas. The training dataset containing image information is also loaded for analysis. Cleaning the Data: Resetting indices for the training dataset and preparing it for further analysis. Visualizing Data Distribution: The distribution of car model images is visualized to understand how well-represented each model is in the dataset. Data Preprocessing

Part 1: Image Count and Organization Folder Setup: The script processes the train directory, where images are organized into subfolders based on car models. Counting Images: A function count_images_in_folder() is used to count the number of images for each car model and store the result in a dataframe. Saving the Count Data: The dataframe is saved as image_counts_by_folder.csv, which contains the car model names and the number of corresponding images. Part 2: Resizing Images Resizing with Aspect Ratio: The resize_with_aspect_ratio() function resizes images in the val directory to a resolution of 224x224 pixels while preserving their aspect ratio. Resized images are saved in the val_resized_224 directory with a white background to fill any empty space. Maintaining Folder Structure: The car model subfolders are preserved during the resizing process.

Car Model Image Classification Overview This project focuses on classifying car models using image data. It involves:

Exploratory Data Analysis (EDA) and Data Preprocessing: Understanding the dataset and preparing the images for model training. Transfer Learning and Fine-Tuning: Experimenting with different CNN models to compare classification performance with and without fine-tuning. Project Structure Datasets datasets/Car names and make.csv: Contains car models and manufacturers. datasets/train_8143_images.csv: Contains information about training images. datasets/train/: Directory containing raw training images organized by car model. datasets/val/: Directory containing raw validation images. datasets/val_resized_224/: Directory with resized validation images (224x224 pixels). Notebooks EDA and Preprocessing:

EDA.ipynb: Contains the exploratory data analysis of the car dataset. data_preprocessing.ipynb: Preprocesses and organizes the images per car model for training. data_preprocessing_part2.ipynb: Resizes validation images to 224x224 while maintaining aspect ratio. Model Training (Transfer Learning and Fine-Tuning):

transfer_learning_without_finetuning.ipynb: Implements transfer learning without fine-tuning. transfer_learning_with_finetuning.ipynb: Implements transfer learning with fine-tuning. Installation To set up the project, install the required dependencies:

bash Copy code pip install pandas matplotlib seaborn numpy Pillow tensorflow keras Exploratory Data Analysis (EDA) Loading the Data: The car names and makes dataset is loaded using Pandas. The training dataset containing image metadata is loaded for analysis. Data Cleaning: The training dataset indices are reset for better analysis. Visualizing Data Distribution: Car model image distribution is visualized to understand class imbalance. Data Preprocessing Part 1: Image Count and Organization Folder Setup:

Images in the train/ directory are organized by car model into subfolders. Counting Images:

A custom function, count_images_in_folder(), counts the images in each folder and stores the result in a DataFrame. Saving Image Counts:

The DataFrame is saved as image_counts_by_folder.csv, showing car model names and image counts. Part 2: Resizing Images Resizing with Aspect Ratio:

The resize_with_aspect_ratio() function resizes images in the val/ directory to 224x224 pixels, maintaining aspect ratio and filling the background with white. Maintaining Folder Structure:

Resized images are saved in the val_resized_224/ directory, with the folder structure maintained. Transfer Learning Experiments: Without Fine-Tuning Overview This part of the project explores transfer learning using CNN models like EfficientNetB0 and InceptionV3. The models' pre-trained layers remain frozen, and only the top layers are retrained on the new car model dataset.

Setup Instructions Clone the repository:

bash Copy code git clone https://github.com/yourusername/your-repo.git Run the transfer_learning_without_finetuning.ipynb notebook:

bash Copy code jupyter notebook transfer_learning_without_finetuning.ipynb Models Used EfficientNetB0 InceptionV3 (GoogleNet) Training Parameters Epochs: 30 Batch Size: Defined in the notebook. Optimizer: Adam (learning rate: 0.0001). Loss Function: Categorical Crossentropy. Results (Without Fine-Tuning) EfficientNetB0 and InceptionV3 results to be filled after model training. Transfer Learning Experiments: With Fine-Tuning Overview This section extends the transfer learning experiments by fine-tuning the pre-trained models. After training the top layers, the last layers of the pre-trained network are unfrozen for further optimization.

Setup Instructions Run the transfer_learning_with_finetuning.ipynb notebook:

bash Copy code jupyter notebook transfer_learning_with_finetuning.ipynb Fine-Tuning Strategy Unfreezing the Last 10 Layers: Initially, the last 10 layers of the pre-trained models were unfrozen for further optimization. Unfreezing the Last 30 Layers: After testing, unfreezing the last 30 layers of EfficientNetB0 yielded better results. Training Parameters Epochs: 30 Batch Size: Defined in the notebook. Optimizer: Adam (learning rate: 0.0001). Loss Function: Categorical Crossentropy. Callbacks Used ModelCheckpoint: Saves the model with the best validation loss. EarlyStopping: Stops training when validation loss stops improving. ReduceLROnPlateau: Reduces learning rate when validation loss stagnates. Results (With Fine-Tuning) EfficientNetB0: Test accuracy: 77.02% Train accuracy: 97.52% InceptionV3: Test accuracy: 68.24% Train accuracy: 92.34% Conclusion This project successfully demonstrates the effectiveness of transfer learning with fine-tuning in improving the classification accuracy of car model images. Fine-tuning significantly improves the performance of EfficientNetB0, making it the best-performing model.