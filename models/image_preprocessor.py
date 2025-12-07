import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
from config import IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD
import requests
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImagePreprocessor:
    def __init__(self, IMAGE_SIZE=IMAGE_SIZE, NORMALIZATION_MEAN=NORMALIZATION_MEAN, NORMALIZATION_STD=NORMALIZATION_STD):
        ## Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## load model and make it ready for inference
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()

        ### preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
        ])


    def encode_image(self, image_input, is_url=True):
        """Convert an image (from a URL or local file) into two key components:
         a base64-encoded string for LLM input and a numerical feature vector for similarity comparisons
        """ 
        try:
            if is_url:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")

            ## Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            ## Preprocess the image for model input
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            ## Extract feature vector
            with torch.no_grad():
                features = self.model(input_tensor)

            
            # Convert features to a list
            feature_vector = features.cpu().numpy().flatten()
            return {"base64": base64_str, "vector": feature_vector}
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}
        

    def find_similar_items(self, user_vector, dataset):
        """Find similar items in the dataset based on the user_vector."""
        try:
            dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
            similiarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)

            ## Find the index of the most similar items
            closest_index = np.argmax(similiarities)
            similiarity_score = similiarities[0][closest_index]

            ## Retrieve the most similar item details
            similar_item = dataset.iloc[closest_index]
            return similar_item, similiarity_score

        except Exception as e:
            print(f"Error finding similar items: {e}")
            return None