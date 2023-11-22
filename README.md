[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# XAI Inference Engine
This package is a wrapper for inferencing with CNN-based PyTorch models. It takes a trained PyTorch CNN model and returns the predictions, sorted prediction indices and saliency maps. For the saliency maps the library uses the [FM-G-CAM](https://pypi.org/project/xai-inference-engine/) method. More type of saliency map generation methods will be added in the future. The package also provides a method to superimpose the saliency maps on the input image.

Users can also use the package to create their own inference engine by extending the `XAIInferenceEngine` class.

Advanced Tutorials: Coming Soon...

[Github](https://github.com/SuienS/xai-inference-engine) | [PyPi](https://pypi.org/project/xai-inference-engine/)

# Requirements
- Python 3.8+
- PyTorch 2.0+


# Installation
Execute the following command in your terminal to install the package.
```python
pip install xai-inference-engine
```

# Usage
Follow the example below to use the package. Copy and paste the code into a python script and run it. Make sure you have the requirements installed. 😊

```python

print("[INFO]: Testing XAIInferenceEngine...")

print("[INFO]: Importing Libraries...")
from xai_inference_engine import XAIInferenceEngine
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO]: Device: {}".format(device))

print("[INFO]: Loading Model...")
# Model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Model config
# Set model to eval mode
model.eval()
last_conv_layer = model.layer4[2].conv3
class_count = 5
class_list = weights.meta["categories"]
img_h = 224

print("[INFO]: Image Preprocessing...")
# Image Preprocessing
url = "https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/cat_dog.png"
r = requests.get(url, allow_redirects=True)
open("dog-and-cat-cover.jpg", "wb").write(r.content)
img = Image.open("dog-and-cat-cover.jpg")
img = img.resize((img_h, img_h), resample=Image.BICUBIC)
img_tensor = preprocess(img).to(device)


print("[INFO]: Creating XAIInferenceEngine...")
xai_inferencer = XAIInferenceEngine(
    model=model,
    last_conv_layer=last_conv_layer,
    device=device,
)

print("[INFO]: Running XAIInferenceEngine.predict()...")
preds, sorted_pred_indices, super_imp_img, saliency_maps = xai_inferencer.predict(
    img=img,
    img_tensor=img_tensor,
)

print("[INFO]: Saving Results to the root folder...")
super_imp_img.save("super_imp_img.jpg")
saliency_maps.save("saliency_maps.jpg")

print("[INFO]: Displaying Results...")
print("        Predictions: {}".format(preds.shape))
print("        Sorted Prediction Indices: {}".format(sorted_pred_indices.cpu().numpy()[:10]))
print("        Heatmaps shape: {}".format(saliency_maps))
print("        Super Imposed Image: {}".format(super_imp_img))

```
# Results
Following image shows comparison between the saliency maps generated by the FM-G-CAM method and the Grad-CAM method.

![FM-G-CAM Comparison with Grad-CAM](https://raw.githubusercontent.com/SuienS/xai-inference-engine/main/docs/images/fmgcam_vs_gradcam.png)

# Author
- [Ravidu Silva](https://www.linkedin.com/in/ravidu-silva/)
