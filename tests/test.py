from PIL import Image
import torch
from torch.nn import functional as F

import requests

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

def run_test():
    
    print("[INFO]: Running test.py")

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
    preds, sorted_pred_indices, super_imp_img, heatmaps = xai_inferencer.predict(
        img=img,
        img_tensor=img_tensor,
    )

    print("[INFO]: Saving Results...")
    super_imp_img.save("super_imp_img.jpg")
    heatmaps.save("heatmaps.jpg")

    print("[INFO]: Displaying Results...")
    print("        Predictions: {}".format(preds.shape))
    print("        Sorted Prediction Indices: {}".format(sorted_pred_indices.cpu().numpy()[:10]))
    print("        Heatmaps shape: {}".format(heatmaps))
    print("        Super Imposed Image: {}".format(super_imp_img))


if __name__ == "__main__":
    run_test()