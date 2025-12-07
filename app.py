"""
Style Finder Main Application
Updated for OpenAI + LangChain
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import gradio as gr
import pandas as pd
import logging
from typing import Optional, Tuple

# Import project modules
from models.image_preprocessor import ImagePreprocessor as ImageProcessor
from models.llm_service import OpenAIService as LlamaVisionService
from utils.helpers import get_all_items_for_image, process_response
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StyleFinderApp:
    """Main Style Finder application"""
    
    def __init__(self, dataset_path: str):
        """Initialize the application"""
        logger.info("Initializing Style Finder App...")
        
        # Load dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        self.data = pd.read_pickle(dataset_path)
        logger.info(f"Loaded dataset with {len(self.data)} items")
        
        # Initialize components
        self.image_processor = ImageProcessor(
            IMAGE_SIZE=config.IMAGE_SIZE,
            NORMALIZATION_MEAN=config.NORMALIZATION_MEAN,
            NORMALIZATION_STD=config.NORMALIZATION_STD
        )
        
        self.llm_service = LlamaVisionService(
            open_vision_model=config.OPENAI_VISION_MODEL
        )
        
        logger.info("App initialization complete")
    
    def process_image(self, image) -> str:
        """Process uploaded image and generate fashion analysis"""
        try:
            # Handle image input
            if not isinstance(image, str):
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                image.save(temp_file.name)
                image_path = temp_file.name
            else:
                image_path = image
            
            # Step 1: Encode image
            user_encoding = self.image_processor.encode_image(
                image_path,
                is_url=False
            )
            
            if user_encoding["vector"] is None:
                return "❌ Error: Unable to process the image. Please try another image."
            
            # Step 2: Find closest match
            closest_row, similarity_score = self.image_processor.find_similar_items(
                user_encoding["vector"],
                self.data
            )
            
            if closest_row is None:
                return "❌ Error: Unable to find a match. Please try another image."
            
            logger.info(
                f"Closest match: {closest_row['Item Name']} "
                f"(similarity: {similarity_score:.2f})"
            )
            
            # Step 3: Get related items
            all_items = get_all_items_for_image(
                closest_row["Image URL"],
                self.data
            )
            
            if all_items.empty:
                return "❌ Error: No items found for the matched image."
            
            # Step 4: Generate fashion response
            bot_response = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding["base64"],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score,
                threshold=config.SIMILARITY_THRESHOLD
            )
            
            # Cleanup temp file
            if not isinstance(image, str):
                try:
                    os.unlink(image_path)
                except:
                    pass
            
            return process_response(bot_response)
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Fashion Style Analyzer"
        ) as demo:
            
            # Title
            gr.Markdown("""
            # 👗 Fashion Style Analyzer
            Upload an image to analyze fashion elements and get detailed information.
            """)
            
            # Example images section
            gr.Markdown("### 📸 Example Images")
            
            with gr.Row():
                with gr.Column():
                    gr.Image(
                        value="examples/test-1.png",
                        label="Example 1",
                        show_label=True,
                        scale=1
                    )
                with gr.Column():
                    gr.Image(
                        value="examples/test-2.png",
                        label="Example 2",
                        show_label=True,
                        scale=1
                    )
                with gr.Column():
                    gr.Image(
                        value="examples/test-3.png",
                        label="Example 3",
                        show_label=True,
                        scale=1
                    )
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="📤 Upload Fashion Image"
                    )
                    submit_btn = gr.Button(
                        "🔍 Analyze Style",
                        variant="primary"
                    )
                    status = gr.Markdown("✅ Ready to analyze...")
                
                with gr.Column(scale=2):
                    output = gr.Markdown(
                        label="📝 Style Analysis Results",
                        height=700
                    )
            
            # Event handlers
            submit_btn.click(
                lambda: "⏳ Analyzing image...",
                inputs=None,
                outputs=status
            ).then(
                self.process_image,
                inputs=image_input,
                outputs=output
            ).then(
                lambda: "✅ Analysis complete!",
                inputs=None,
                outputs=status
            )
            
            # Information section
            gr.Markdown("""
            ### ℹ️ How It Works
            1. **Image Encoding** - Converts your fashion image to numerical vectors
            2. **Similarity Matching** - Finds visually similar items in our database
            3. **AI Analysis** - Generates detailed fashion descriptions using GPT-4 Vision
            """)
            
            return demo


def main():
    """Main entry point"""
    try:
        # Create app
        app = StyleFinderApp(config.DATASET_PATH)
        
        # Create and launch interface
        interface = app.create_interface()
        interface.launch(
            server_name="Localhost",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
