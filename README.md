# Probabilistic and Reconstruction-based Competency Estimation (PaRCE)

This is the codebase for the paper titled "PaRCE: Probabilisitc and Reconstruction-based Competency Estimation for CNN-based Image Classification," which was submitted to the 2025 IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR). This README describes how to reproduce the results achieved in this paper. Note that most of the steps listed here can be skipped by downloading the saved [datasets](https://drive.google.com/drive/folders/14-MN0aemA_ebeMs2yjwZTBNKQ65Om6vM?usp=share_link) and trained [models](https://drive.google.com/drive/folders/1bv_5zA95u5yGyVChOiKOROiICScUrPBh?usp=share_link) provided in our Google Drive [folder](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link).

## 0) Set Up Codebase

### 0a. Clone this repository

Clone this repository:
```
git clone https://github.com/dbl-blnd/parce.git
```

### 0b. Set up the source directory

It's recommended that you create an environment with Python 3.8:
```
conda create -n parce python=3.8
```

Then, in the main folder (`parce`), run the following command:
```
pip install -e .
```

## 1) Setup Training Dataset

### 1a. Download the dataset files

To replicate the results presented in the paper, download the lunar, speed, and pavilion dataset files from the `data` folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link). Create a folder called `data` in the main directory (`parce`) and subfolders called `lunar`, `speed`, and `pavilion`. Place the dataset files you downloaded into the corresponding subfolders. If you simply want to use the default datasets, you can skip to step 2. If you want to create a new dataset, proceed through the remaining substeps in this section.

### 1b. Set up directory structure

By default, datasets are assumed to be saved in the following structure:

|-- data  
&emsp;|-- dataset1  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images  
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted  
&emsp;|-- dataset2  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images   
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted 

The unsorted folder should contain in-distribution training images that have not been labeled, while the ID folder contains all labeled in-distribution images organized by their class labels. If you already have a labeled dataset, you can organize them in the ID folder and skip to step 1d. If you only have unlabeled data, you can place it all in the unsorted folder and proceed to step 1c. The OOD folder should contain all out-of-distribution images. If this data is labeled, it can be orgnized into its class labels. If it is unlabeled, you can place it all into the same subfolder within the OOD folder. A dataset that has already been set up (following step 1d) will be saved in a compressed NumPy file called dataset.npz in the main dataset folder.

### 1c. Cluster unlabeled data

If you have labeled data, skip to the next step. If you have unlabeled in-distribution data saved in the unsorted directory, you can cluster these images using the create_dataset script:

```
python src/datasets/create_dataset.py <path_to_dataset> --cluster_data
```

This command will cluster the unsorted images and save them in subfolders within the ID folder.

### 1d. Save custom dataset

Once you have existing classes of in-distribution data, you can save a dataset of training, test, and OOD data using the create_dataset script:

```
python src/datasets/create_dataset.py <path_to_dataset> --save_data
```

Note that this step can be combined with the previous one. By separating these two steps, you can validate the generated clusters before saving your dataset. You can also use to height and width arguments to resize your images if desired. This script will save a compressed NumPy file called dataset.npz in your dataset directory.

### 1e. Update dataloader setup script

Use the existing cases in the setup_dataloader script to enable the use of your custom dataset. You will need to add a section to the get_class_names, get_num_classes, and the setup_loader functions.

## 2) Generate Classification Model

### 2a. Download the classification model files

To replicate the results presented in the paper, download the lunar, speed, and pavilion classification models from the models folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place them in a folder called `models` in the main directory (`parce`) with the same folder structure provided in Drive. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the classification model, see substep 2e. Otherwise, you can skip to step 3. 

### 2b. Define the classification model architecture

Create a JSON file defining your model architecture using the example given in `src/networks/classification/layers.json`. Currently, you can define simple model architectures composed of convolutional, pooling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add a flattening layer in between other layers. For convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and amount of zero padding. For pooling layers, you must specify the pooling function (max or average) and the kernel size. Finally, for fully-connected layers, you must specify the number of input and output nodes.

### 2c. Define the classification training parameters

Create a configuration file defining your training parameters using the example given in `src/networks/classification/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 2d. Train the classification model

You can train your model using the train script in the networks classification folder:

```
python src/networks/classification/train.py --train_data <dataset> --output_dir models/<dataset>/classify/
```

The argument train_data is used to indicate which dataset should be used to train your classification model, which should be lunar, speed, or pavilion if you are using the default training datasets. The argument output_dir is used to define where your trained classification model will be saved. (This is `models/<dataset>/classify` for the default models downloaded in 2a.) The arguments network_file and train_config can be used to specify the location of your model architecture JSON file (created in 2b) and training parameter config file (created in 2c) if you are not using ones contained in output_dir. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 2e. Evaluate the classification model

You can evaluate your model using the test script in the networks classification folder:

```
python src/networks/classification/test.py --test_data <dataset> --model_dir models/<dataset>/classify/
```

The argument test_data is used to indicate which dataset should be used to evaluate your classification model, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save a confusion matrix to the model_dir directory.

## 3) Design Overall Competency Estimator

### 3a. Download the overall competency model files

If you have not done so already, download the reconstruction models from the models folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place them in the appropriate folders in the main directory (`parce`). The trained overall competency estimators used in the paper are saved in files called `parce.p` in these reconstruction folders. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the reconstruction model, see substep 3e. To evaluate the overall competency estimator, see 3g. To visualize examples of overall model competency estimates, see substep 3h. Otherwise, you can skip to step 4. 

### 3b. Define the reconstruction model architecture

Create a JSON file defining your model architecture using the example given in `src/networks/reconstruction/layers.json`. The reconstruction model used by the overall competency estimator is meant to reconstruct the input image. Currently, you can define simple model architectures composed of convolutional, pooling, transposed convolutional, unsampling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add an unflattening layer in between other layers. For transposed convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and the input/output zero padding. For unsampling layers, you must specify the scale factor or the target output size. If the unsampling mode is not specified, then the 'nearest' unsampling technique will be used. For fully-connected layers, you must specify the number of input and output nodes. Finally, for unflattening, the number of output channels, as well as the resulting height and width, must be provided.

### 3c. Define the reconstruction training parameters

Create a configuration file defining your training parameters using the example given in `src/networks/reconstruction/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 3d. Train the reconstruction model

To train the image reconstruction model, you can use the train script in the networks reconstruction folder:

```
python src/networks/reconstruction/train.py reconstruct --architecture autoencoder --train_data <dataset> --model_dir models/<dataset>/classify/ --output_dir models/<dataset>/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train your reconstruction model, which should be lunar, speed, or pavilion if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d. The argument output_dir is used to define where your trained reconstruction model will be saved. (This is `models/<dataset>/reconstruct` for the default models.) The arguments network_file and train_config can be used to specify the location of your model architecture JSON file (created in 3b) and training parameter config file (created in 3c) if you are not using ones contained in output_dir. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 3e. Evaluate the reconstruction model

To evaluate the image reconstruction model, you can use the test script in the networks reconstruction folder:

```
python src/networks/reconstruction/test.py reconstruct --architecture autoencoder --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate your reconstruction model, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save several figures (displaying the original and reconstructed images, along with the reconstruction loss) to a folder called `reconstruction` in decoder_dir.

### 3f. Train the overall competency estimator

You can train an overall competency estimator for your model using the train script in the competency folder:

```
python src/competency/train.py overall --train_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train the overall competency estimator, which should be lunar, speed, or pavilion if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 3g. Evaluate the overall competency estimator 

You can evaluate your overall competency estimator using the test script in the competency folder:

```
python src/competency/test.py overall --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate the overall competency estimator, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will generate plots of the reconstruction loss distributions and probabilistic competency estimates for the correctly classified and misclassified in-distribution data, as well as the out-of-distribution data, and save them to the decoder_dir directory.

### 3h. Visualize the competency estimates

You can visualize the overall competency estimates for each test image using the visualize script in the competency folder:

```
python src/competency/visualize.py overall --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used for visualization, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to visualize the model estimates using a GPU. This script will save figures of the input image and overall competency score to subfolders (correct, incorrect, and ood) in a folder called `competency` in decoder_dir.

## 4) Compare Overall Competency Estimator to Existing Methods

### 4a. Obtain ensemble of classification models

The classification folders referenced in step 2a contain multiple model files to be used as an ensemble of classifiers. If you want to replicate our results, make sure you have all of these files downloaded in the `classify` folder. If you are working with new datasets and want to analyze the performance of ensembling, you should generate an ensemble of classification models, following the instructions in step 2.

### 4b. Evaluate competency methods for manually collected data

We compare our overall competency scores to a number of uncertainty quantification (UQ) and out-of-distribution (OOD) detection methods, which are implemented in `src/comparison/overall/methods.py`. We can collect results for each method individually using the evaluate script in that same folder:

```
python src/comparison/overall/evaluate.py parce --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/ --save_file results/<dataset>/competency/unmodified/data/parce.csv
```

In the command above, you can replace `parce` with each of the available methods (softmax, temperature, dropout, ensemble, energy, odin, openmax, dice, kl, mahalanobis, and knn) and select the CSV file where you would like to save the results for the particular method. The argument test_data is used to indicate which dataset should be used for evaluation, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag to run evaluations on your GPU. Note that when evaluating the ensemble method, you will need to specify the model_dir that contains all of the model path files for the ensemble. This command will save a CSV file of results to the file indicated by the save_file argument. You can also specify a file to save/load trained estimators using the estimator_file argument.

To run the evaluations for all of the existing UQ and OOD detection methods, you can use the evaluation bash script in the comparison folder:

```
./src/comparison/overall/evaluate.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/comparison/overall/evaluate.sh`. This script will save a CSV file for all currentlly implemented methods to a folder called `results/<dataset>/competency/unmodified/data/`.

### 4c. Compare competency methods for manually collected data

After running evaluations for all of the existing UQ and OOD detection methods, you can compare them using the compare script in the overall comparison folder:

```
python src/comparison/overall/compare.py --data_dir results/<dataset>/competency/unmodified/data/ --plot_dir results/<dataset>/competency/unmodified/plots/
```

You should specify the `data_dir` where your evaluations were saved in the previous step and the `plot_dir`, where you want the generated plots to be saved. This command will pull all of the CSV files from the given folder, read the results, calculate a number of performance metrics for each method, print a table comparing the methods to the terminal, and save the same table to a CSV file. It will also save figures of the score distributions for each method to the provided folder, along with ROC curves.

### 4d. Evaluate competency methods across image properties

You can change various images properties of the in-distribution validation set and evaluate the impact on prediction accuracy and competency estimates using the `evaluate` script in the analysis folder:

```
python src/analysis/evaluate.py --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/ --property <property> --factor <factor> --data_dir results/<dataset>/competency/modified/data/
```

The argument test_data is used to indicate which dataset should be used to evaluate the overall competency estimator, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You should specify the image property you wish to change and the corresponding factor. In our results, we modify the following properties: saturation, contrast, brightness, noise, and pixelate. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script with save classification model outputs and competency estimates to a pickle file called `<factor>.p` to the folder `<data_dir>/<property>.`

To run this script for various image properties and factors, you can use the evaluation bash script in the `analysis` folder:

```
./src/analysis/evaluate.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/analysis/evaluate.sh`. This script will create subfolders within a folder called `results/<dataset>/data` for each image property and generate a pickle file corresponding to each image property factor.

### 4e. Compare competency methods across image properties

After running evaluations for all of the image modifications of interest, you can compare them using the compare script in the analysis folder:

```
python src/analysis/compare.py --data_dir results/<dataset>/competency/modified/data/ --plot_dir results/<dataset>/competency/modified/plots/
```

You should specify the `data_dir`, where your evaluations were saved in the previous step and the `plot_dir`, where you want the generated plots to be saved. This command will pull all of the pickle files from the given folder, read the results, calculate a number of performance metrics for each method, print a table comparing the methods to the terminal and save the same table to a CSV file. It will also save figures of the score distributions for each method to the provided folder, along with ROC curves.

## 5) Design Regional Competency Estimator

### 5a. Download the regional competency model files

If you have not done so already, download the inpainting models from the models folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place them in the appropriate folders in the main directory (`parce`). The trained regional competency estimators used in the paper are contained in the inpainting folders, along with labels for the segmented OOD datasets provided. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the inpainting model, see substep 5e. To evaluate the regional competency estimator, see 5h. To visualize examples of regional model competency maps, see substep 5i. Otherwise, you can skip to step 6. 

### 5b. Define the inpainting model architecture

Create a JSON file defining your model architecture using the example given in `src/networks/reconstruction/layers.json`. The inpainting model used by the regional competency estimator is meant to fill in missing pixels of the input image. Currently, you can define simple model architectures composed of convolutional, pooling, transposed convolutional, unsampling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add an unflattening layer in between other layers. For transposed convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and the input/output zero padding. For unsampling layers, you must specify the scale factor or the target output size. If the unsampling mode is not specified, then the 'nearest' unsampling technique will be used. For fully-connected layers, you must specify the number of input and output nodes. Finally, for unflattening, the number of output channels, as well as the resulting height and width, must be provided.

### 5c. Define the inpainting training parameters

Create a configuration file defining your training parameters using the example given in `src/networks/reconstruction/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 5d. Train the inpainting model

To train the image inpainting model, you can use the train script in the networks reconstruction folder:

```
python src/networks/reconstruction/train.py inpaint --architecture autoencoder --train_data <dataset> --model_dir models/<dataset>/classify/ --init_model models/<dataset>/reconstruct/ --output_dir models/<dataset>/inpaint/
```

The argument train_data is used to indicate which dataset should be used to train your inpainting model, which should be lunar, speed, or pavilion if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d. The argument output_dir is used to define where your trained inpainting model will be saved. (This is `models/<dataset>/inpaint` for the default models.) The arguments network_file and train_config can be used to specify the location of your model architecture JSON file (created in 5b) and training parameter config file (created in 5c) if you are not using ones contained in output_dir. It is recommended that you initialize the training of this model with the reconstruction model trained in step 3d for the dataset of interest. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 5e. Evaluate the inpainting model

To evaluate the image inpainting model, you can use the test script in the networks reconstruction folder:

```
python src/networks/reconstruction/test.py inpaint --architecture autoencoder --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/inpaint/
```

The argument test_data is used to indicate which dataset should be used to evaluate your inpainting model, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save several figures (displaying the original, masked, and reconstructed images, along with the reconstruction loss) to a folder called `reconstruction` in decoder_dir.

### 5f. Train the regional competency estimator

You can train a regional competency estimator for your model using the train script in the competency folder:

```
python src/competency/train.py regional --train_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/inpaint/
```

The argument train_data is used to indicate which dataset should be used to train the regional competency estimator, which should be lunar, speed, or pavilion if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 5g. Generate true labels of familiar/unfamiliar regions in image

To evaluate the performance of the regional competency estimator, you should generate labels of the regions in the OOD images that are familiar or unfamiliar to the perception model using the create_labels script:

```
python src/utils/create_labels.py --test_data <dataset> --decoder_dir models/<dataset>/inpaint/
```

Each image in the OOD set of the test_data dataset will be segmented, and you will be shown each segmented region with the prompt: "Does this segment contain a structure not present in the training set?" Answering yes (y) will indicate that this region is unfamiliar to the model, while answering no (n) will indicate that it is familiar. You can also answer that you are unsure (?). These responses will be saved to a pickle file called ood_labels.p in the decoder_dir directory. Note that you can also review these labels using the test flag and begin relabeling from the middle of the OOD set using the start_idx parameter.

### 5h. Evaluate the regional competency estimator 

You can evaluate your regional competency estimator using the test script in the competency folder:

```
python src/competency/test.py regional --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/inpaint/
```

The argument test_data is used to indicate which dataset should be used to evaluate the regional competency estimator, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will generate plots of the reconstruction loss distributions and probabilistic competency estimates for regions in in-distribution data, as well as both familiar and unfamiliar regions in OOD images, and save them to the decoder_dir directory.

### 5i. Visualize the competency estimates

You can visualize the regional competency estimates for each test image using the visualize script in the competency folder:

```
python src/competency/visualize.py regional --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/inpaint/
```

The argument test_data is used to indicate which dataset should be used for visualization, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag if you want to visualize the model estimates using a GPU. This script will save figures of the input image and regional competency image to subfolders (correct, incorrect, and ood) in a folder called `competency` in decoder_dir.

## 6) Compare Regional Competency Estimator to Existing Methods

### 6a. Evaluate competency methods for manually collected data

We compare our regional competency scores to a number of anomaly detection and localization methods, which are implemented in `src/comparison/regional/methods.py`. We can collect results for each method individually using the evaluate script in that same folder:

```
python src/comparison/regional/evaluate.py parce --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/inpaint/ --save_file results/<dataset>/competency/regional/data/parce.csv
```

In the command above, you can replace `parce` with each of the available methods (ganomaly, draem, fastflow, padim, patchcore, reverse, and stfpm) and select the CSV file where you would like to save the results for the particular method. The argument test_data is used to indicate which dataset should be used for evaluation, which should be lunar, speed, or pavilion if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag to run evaluations on your GPU. This command will save a CSV file of results to the file indicated by the save_file argument. You can also specify a file to save/load trained estimators using the estimator_file argument.

To run the evaluations for all of the existing anomaly detection and localization methods, you can use the evaluation bash script in the comparison folder:

```
./src/comparison/regional/evaluate.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/comparison/regional/evaluate.sh`. This script will save a CSV file for all currentlly implemented methods to a folder called `results/<dataset>/competency/regional/data/`.

### 6b. Compare competency methods for manually collected data

After running evaluations for all of the existing anomaly detection and localization methods, you can compare them using the compare script in the regional comparison folder:

```
python src/comparison/regional/compare.spy --data_dir results/<dataset>/competency/regional/data/ --plot_dir results/<dataset>/competency/regional/plots/ --decoder_dir models/<dataset>/inpaint/
```

You should specify the `data_dir` where your evaluations were saved in the previous step and the `plot_dir`, where you want the generated plots to be saved. You should also provide the `decoder_dir`, where the OOD segment labels were saved, along with the `height` and `width` of the competency images (if they are not the default size). This command will pull all of the CSV files from the given folder, read the results, calculate a number of performance metrics for each method, print a table comparing the methods to the terminal, and save the same table to a CSV file. It will also save figures of the score distributions for each method to the provided folder, along with ROC curves.

### 6c. Visualize competency methods for manually collected data

You can also visualize the competency maps for each evaluated method using the visualize script:
```
python src/comparison/regional/visualize.py --test_data <dataset> results/<dataset>/competency/regional/data --save_dir results/<dataset>/competency/regional/images/ --example <image_id>
```

You should specify the dataset you're working with, the folder where all of the evaluation files are stored, the folder where visualizations of the compentency maps will be saved, and the example image of interest from the given dataset. This command will pull all of the data files from the given folder, visualize the generated competency maps for the selected example, and save the figures to the folder specified by save_dir.
