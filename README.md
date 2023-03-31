## Author

David Maiolo

This project was created by David Maiolo on March 30, 2023. For any inquiries or issues related to the project, 
you can contact me through my GitHub profile or the repository's issue tracker.

# Mars Terrain Analysis

This project, created by David Maiolo on March 30, 2023, is designed to analyze the terrain of Mars by processing images 
and their corresponding masks. The project uses a Convolutional Neural Network (CNN) model to identify and classify 
different geological formations on the Martian surface.

## Project Structure

The project consists of the following files:

1. `main.py` - The main script that preprocesses the images, loads the dataset, trains the CNN model, and validates the model's performance.
2. `model.py` - Contains the `create_model` function that defines the architecture of the CNN model.
3. `utils.py` - Contains utility functions for loading and preprocessing image data.

## Dataset

The dataset is a collection of images and their corresponding masks. Each image is a 260x260 grayscale representation 
of a Martian terrain, while the mask is a binary image highlighting the geological formations.

- Images: NLA_397586934EDR_F0010008AUT_04096M1.JPG, ...
- Masks: NLA_397586934EDR_F0010008AUT_04096M1.png, ...

The dataset is divided into training and validation sets using an 80-20 split.

## Using the RGB Key

The model uses the RGB key to interpret and assign labels to the terrain features in the Mars images based on their 
pixel values. The RGB key is as follows:

- 0,0,0 - soil
- 1,1,1 - bedrock
- 2,2,2 - sand
- 3,3,3 - big rock
- 255,255,255 - NULL (no label)

The program reads an RGB image, applies the masks (rover and range-30m), and replaces the pixel values according to the 
RGB key. This process helps to convert the original image into a labeled image that indicates different types of terrain 
or features, such as soil, bedrock, sand, and big rocks.

## AI4MARS Dataset

This project uses a subset of the AI4MARS dataset for terrain-aware autonomous driving on Mars. The AI4MARS dataset was 
created by R. Michael Swan, Deegan Atha, Henry A. Leopold, Matthew Gildner, Stephanie Oij, Cindy Chiu, and Masahiro Ono 
at the Jet Propulsion Laboratory, California Institute of Technology.

The AI4MARS dataset is a large-scale dataset consisting of approximately 326,000 semantic segmentation full image labels 
on 35,000 images from the Curiosity, Opportunity, and Spirit rovers. These labels were collected through crowdsourcing, 
and each image was labeled by around 10 people to ensure greater quality and agreement.

Additionally, the dataset includes around 1,500 validation labels annotated by rover planners and scientists from NASA's 
MSL (Mars Science Laboratory) mission, which operates the Curiosity rover, and MER (Mars Exploration Rovers) mission, 
which operated the Spirit and Opportunity rovers.

The full dataset is available for download as a 5.72 GB zip file called ai4mars-dataset-merged-0.1.zip and can be found on 
the NASA data portal [here](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix/data).

To use the full dataset in this project, download the zip file, extract its contents, and replace the images and masks folders in 
the project directory with the corresponding folders from the full dataset.

### Citation

If you use the AI4MARS dataset in your work, please cite the following paper:

Swan, R. M., Atha, D., Leopold, H. A., Gildner, M., Oij, S., Chiu, C., & Ono, M. (2021). AI4MARS: 
A Dataset for Terrain-Aware Autonomous Driving on Mars. Jet Propulsion Laboratory, California Institute of Technology, 
Pasadena, CA 91109, USA.

## Model

The CNN model consists of the following layers:

1. Input layer with the shape of (260, 260, 1)
2. Conv2D layer with 32 filters, kernel size of (3, 3), and ReLU activation
3. MaxPooling2D layer with pool size of (2, 2)
4. Conv2D layer with 64 filters, kernel size of (3, 3), and ReLU activation
5. MaxPooling2D layer with pool size of (2, 2)
6. Conv2DTranspose layer with 64 filters, kernel size of (3, 3), and ReLU activation
7. UpSampling2D layer with a size of (2, 2)
8. Conv2DTranspose layer with 32 filters, kernel size of (3, 3), and ReLU activation
9. UpSampling2D layer with a size of (2, 2)
10. Conv2D layer with 1 filter, kernel size of (1, 1), and sigmoid activation

The model is trained using the Adam optimizer and binary_crossentropy as the loss function. The training is carried out 
for 10 epochs with a batch size of 2.

## Integrating the Model into Raspberry Pi / Car Kit

To integrate the model into the Raspberry Pi / Car Kit, follow these steps:

1. Convert the h5 file into a format suitable for TensorFlow Lite. This conversion will reduce the model size and make it compatible with the Raspberry Pi.
```tflite_convert --keras_model_file model.h5 --output_file model.tflite```
2. Transfer the model.tflite file to the Raspberry Pi.
3. Install TensorFlow Lite for Python on the Raspberry Pi:
```pip3 install tflite-runtime```
4. Install the Freenove car kit's required libraries, if not already installed:
```pip3 install RPi.GPIO```
```pip3 install smbus2```
5. Write a Python script on the Raspberry Pi to load the model.tflite file, preprocess the input images from the car kit's camera, and run inference using the TensorFlow Lite interpreter. Additionally, integrate the Freenove car kit's control code to control the car's motors based on the detected terrain feature.

## Structure of the h5 File
The h5 file is a Hierarchical Data Format (HDF5) file that contains the trained model's architecture, weights, and training 
configuration. The file has the following structure:

- model_config: A JSON object that describes the model's architecture
- optimizer_config: A JSON object that contains the optimizer configuration used during training
- layer_names: A list of layer names in the model
- layer_weights: A group containing the weight values for each layer
- training_config: A JSON object that contains information about the training process, such as the loss function, metrics, and batch size

## Usage

To run the project, follow these steps:

1. Clone the repository to your local machine:
```git clone https://github.com/dmaiolo/mars_terrain_analysis2.git```
2. Change into the project directory:
```cd mars_terrain_analysis2```
3. Install the required Python packages:
```pip install -r requirements.txt```
4. Run the main script:
```python main.py```

Upon running the script, the images and masks will be loaded, preprocessed, and divided into training and validation sets. 
The model will then be trained and validated using the provided dataset. The training progress, along with the loss and 
accuracy, will be displayed in the console.

## Contributing

Feel free to fork the repository, create a branch for your changes, and submit a pull request. Contributions, suggestions, 
and improvements are always welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
