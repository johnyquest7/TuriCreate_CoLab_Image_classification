# TuriCreate_CoLab_Image_classification
<img align="right" src="https://docs-assets.developer.apple.com/turicreate/turi-dog.svg" alt="Turi Create" width="100">
Using Google's Colab to create a deep learning model employing Apple's TuriCreate just using your browser.

You do not need to own an expensive GPU to create deep learning models. 

This tutorial demonstrates creation of an image recognition classifier. Later you can download the model to create an iPhone app!

All of this costs you nothing!

## Objectives

1) Create an iPython/Jupyter notebook in Google Drive using CoLab.

2) Install Apple’s TuriCreate deep learning package with GPU support using the iPython notebook.

3) Create a deep learning model to distinguish between cats and dogs with high accuracy, using the free GPU available through CoLab.

4) Download and use this model to create an iOS app which can classify dogs and cats pictures.


## Step 1: Create an iPython notebook in Google Drive using CoLab.

Sign in to your google drive. Click ‘New’, go down the menu and click 'more'. Then click 'connect more apps'. Search for CoLab and install it.


Add CoLaboratory
Once you have CoLab installed in your drive. Click on the ‘Colab’ menu under ‘New’. A new window will open. Click — ‘Edit’ — ‘Notebook settings’ 

**iPython Notebook setting.**

Select runtime as Python 3 and choose GPU under hardware accelerator and click save.

## Step 2 & 3: Install Apple’s TuriCreate deep learning package with GPU support using this notebook and create the deep learning model.


Enter the following code in the iPython notebook
```
!pip install turicreate
```

To execute a code you have to press SHIFT + ENTER

This will install turicreate in your virtual machine provided by Google.

**Get access to your Google Drive files.**

Copy and paste the following code block into your iPython notebook. You will be asked to click a link to generate a secret key to access your Google Drive. Copy and paste secret key it into the space provided with the notebook.

```
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```

Mount your google drive to access files from it.
```
!mkdir -p drive
!google-drive-ocamlfuse drive
```
The image folder that I am using for this tutorial is located in my google drive under ‘app’ folder. So we need to change directory to the ‘app’ folder.

```python
import os
os.chdir(“drive/app”)
```

Loading the TuriCreate framework into our notebook.
```python
import turicreate as tc
```
Next step is to load the images from ImageFolder in your google drive and store it into a data frame — ‘data’.

This ‘ImageFolder’ has 2 folders — ‘Cat’ and ‘Dog’.

Each folder has about 1000 images. Original images were obtained from https://www.kaggle.com/c/dogs-vs-cats/data

The whole data set from kaggle website was downloaded to my desktop. After that I created a folder called ‘ImageFolder’ in my desktop with sub folders ‘Cat’ & ‘Dog’. First 1000 images of cats and dogs were copied into the individual folders. Later the ‘ImageFolder’ was uploaded to the ‘app’ folder in my google drive.
```python
data = tc.image_analysis.load_images(‘ImageFolder’, with_path=True)
```

Now we are going to label each image in the SFrame using the folder name
```python
data[‘label’] = data[‘path’].apply(lambda path: ‘dog’ if ‘/Dog’ in path else ‘cat’)
```

Split the image data set into training (80%) and testing (20%) data.
```python
train_data, test_data = data.random_split(0.8)
```

We are ready to train our model. Before we do that, we need to uninstall the default MXNET package and install the latest version.

Switch to the root directory in yor google virtual machine
```python
os.chdir(“/usr”)
```
Uninstalling previous versions of mxnet
```
!pip uninstall -y mxnet
```
Installing the latest version of mxnet to work with turicreate
```
!pip install mxnet==1.1.0
```
Make sure that mxnet is working by trying to import it.

If you get an error here, you need to fix it before going forward.
```python
import mxnet as mx
```
Now to the most important step — creating your deep learning model.

TuriCreate automatically picks the right model based on your data.

With GPU support this should only take few minutes. Without GPU it may take hours.
```python
model = tc.image_classifier.create(train_data, target=’label’)
```
Evaluating the model

```python
metrics = model.evaluate(test_data)
print(metrics[‘accuracy’])
```

Export the model for use in Core ML
```python
model.export_coreml(‘MyCustomImageClassifier.mlmodel’)
```
The following code will download the model to your local hard disk. It may take few minutes.

```
from google.colab import files
files.download(‘MyCustomImageClassifier.mlmodel’)
```

## Step 4: Deploying the model

You can import the model to Xcode and create an app to classify dogs and cats. 

