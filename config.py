### Define the OPENAI Vision model name
import os

OPENAI_VISION_MODEL = "gpt-4o"
# ============================================
DATASET_PATH = os.getenv("DATASET_PATH", "swift-style-embeddings.pkl")

# Image processing settings
IMAGE_MAX_SIZE = (1024, 1024)  # Max size for image resizing
IMAGE_FORMAT = "JPEG"  # Image format for saving processed images
IMAGE_SIZE = (224,224)  # Size for model input
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
# ============================================

## Define Similarity threshold for fashion item matching
SIMILARITY_THRESHOLD = 0.8
# ============================================

# Number of top similar items to retrieve
TOP_K_SIMILAR_ITEMS = 5
# ============================================
