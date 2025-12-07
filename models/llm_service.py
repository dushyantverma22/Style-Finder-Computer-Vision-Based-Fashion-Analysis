
import logging
from dotenv import load_dotenv
from config import OPENAI_VISION_MODEL
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd

load_dotenv()

## set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self, open_vision_model=OPENAI_VISION_MODEL, temp=0.7, logger=logger, top_p=0.6, max_tokens=1000):
        self.model_name = open_vision_model
        self.logger = logger
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("API key is required to initialize OpenAIService.")
        
        self.model = ChatOpenAI(
            model_name=self.model_name,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=self.api_key
        )

        logger.info(f"Initialized OpenAIService with model: {self.model_name}")

    def generate_response(self, prompt, encoded_image):
        """ Generate a textual response from a vision-language model using a base64-encoded image and a user-provided prompt."""
        try:
            self.logger.info(f"Sending request to LLM with prompt length: {len(prompt)}")
            
            # Create message with image and text
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            )
            
            # Invoke model
            response = self.model.invoke([message])
            content = response.content
            
            self.logger.info(f"Received response with length: {len(content)}")
            
            # Check if response might be truncated
            if len(content) > 7900:
                self.logger.warning(
                    f"Response may be truncated. Length: {len(content)}"
                )
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {e}"
    

    def generate_fashion_response(
            self,
            user_image_base64: str,
            matched_row: dict,
            all_items: 'pd.DataFrame',
            similarity_score: float,
            threshold: float = 0.8
        ) -> str:
            """
            Generate a fashion-specific response.
            
            Args:
                user_image_base64: Base64-encoded user image
                matched_row: Best matching item row
                all_items: DataFrame with all related items
                similarity_score: Similarity score of the match
                threshold: Threshold for exact match
                
            Returns:
                Formatted fashion analysis response
            """
            # Format items list
            items_list = []
            items_to_show = all_items.head(5)  # Only 5 items
            for _, row in items_to_show.iterrows():
                item_str = f"{row['Item Name']} - {row['Price']} - {row['Link']}"
                items_list.append(item_str)
            
            items_description = "\n".join(
                f"- {item}" for item in items_list
            )
            
            # Choose prompt based on similarity score
            if similarity_score > threshold:
                section_header = "ITEM DETAILS"
                assistant_prompt = f"""You are conducting a professional retail catalog analysis.
                This image shows standard clothing items available in department stores.
                Focus exclusively on professional fashion analysis for a clothing retailer.

                {section_header} - always include this section in your response:
                {items_description}

                Please:
                1. Identify and describe the clothing items objectively (colors, patterns, materials)
                2. Categorize the overall style (business, casual, etc.)
                3. Include the {section_header} section at the end

                Use formal, clinical language. This is for a professional retail catalog."""
            else:
                section_header = "SIMILAR ITEMS"
                assistant_prompt = f"""You are conducting a professional retail catalog analysis.
                This image shows standard clothing items available in department stores.
                Focus exclusively on professional fashion analysis for a clothing retailer.

                {section_header} - always include this section in your response:
                {items_description}

                Please:
                1. Note these are similar but not exact items
                2. Identify clothing elements objectively (colors, patterns, materials)
                3. Include the {section_header} section at the end

                Use formal, clinical language. This is for a professional retail catalog."""
            
            # Generate response
            response = self.generate_response(user_image_base64, assistant_prompt)
            
            # Ensure items section is included
            if len(response) < 100:
                self.logger.info("Response appears incomplete, creating basic response")
                response = f"""Fashion Analysis: This outfit features a collection of carefully coordinated pieces.

                {section_header}:
                {items_description}"""
            
            elif section_header not in response and f"{section_header}:" not in response:
                self.logger.info("Item details section missing, appending it")
                response = f"{response}\n\n{section_header}:\n{items_description}"
            
            return response