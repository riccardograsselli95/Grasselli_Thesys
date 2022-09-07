## Multimodal NN for reconstruct radial and lowpass filtered MRI images
### Riccardo Grasselli Thesis
### Relator: Prof. Aurelio Uncini
### La Sapienza University

This repository contains source code to reproduce thesys experiment. 
The Multimodal Neural Network files are based on "A Wide Multimodal Dense U-Net for Fast Magnetic Resonance Imaging" (https://ieeexplore.ieee.org/document/9287519): this network is able to reconstruct undersampled MS images taken from several datasets (Brats2021 and OpenMS 2018).
I have studied the possibility to use this network also with radial and lowpass undersampled MRI images to simulate a different sampling applied on images.
In "Utils" folder there are codes to undersampling the datasets according the two modalities: lowpass and radial.


Step 1: download dataset of interesting (OpenMS or Brats2021) from following links:
 - Brats 2021: http://www.braintumorsegmentation.org/
 - Open MS data: https://github.com/muschellij2/open_ms_data

Step 2: create filtered images from TF1 data. "Utils" folder contains python file to filter dataset. In particular there are two type of filter: radial and lowpass to simulate a radial subsampling and a TimeResolved MRI.

Step 3: put images in Dataset folder

Step 4: launch main.py

Every main.py is configured for the a singular dataset and will save results in Results folder. 

With print_results.py you can load results.json file and visualize results with Matplotlib. 
