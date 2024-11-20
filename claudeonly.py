from typing import Dict, List, TypedDict
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
import json
import os
from dotenv import load_dotenv
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AgentState(TypedDict):
    messages: List[Dict]
    should_end: bool
    context: Dict
    product_db: Dict
    history: List[Dict] 

PRODUCT_DB = {
    "laptop": {
        "name": "ProBook X1",
        "price": 999.99,
        "specs": "13-inch, 16GB RAM, 512GB SSD",
        "availability": "In Stock",
        "category": "Electronics"
    },
    "headphones": {
        "name": "SoundMax Pro",
        "price": 149.99,
        "specs": "Wireless, Noise-cancelling",
        "availability": "Limited Stock",
        "category": "Electronics"
    },
    "smartphone": {
        "name": "GalaxyTech Pro",
        "price": 899.99,
        "specs": "6.7-inch OLED, 256GB Storage, 5G",
        "availability": "In Stock",
        "category": "Electronics"
    },
    "smartwatch": {
        "name": "FitTrack Elite",
        "price": 299.99,
        "specs": "Always-on Display, Heart Rate, GPS",
        "availability": "In Stock",
        "category": "Electronics"
    },
    "tablet": {
        "name": "SlateBook Air",
        "price": 649.99,
        "specs": "10.9-inch, 128GB, WiFi+5G",
        "availability": "Limited Stock",
        "category": "Electronics"
    },
    "camera": {
        "name": "PhotoPro X100",
        "price": 1299.99,
        "specs": "24MP, 4K Video, Mirrorless",
        "availability": "In Stock",
        "category": "Electronics"
    }
}

class RetailChatAgent:
    def __init__(self, api_key: str = None):
        """Initialize the retail chat agent"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=self.api_key)
        self.product_db = PRODUCT_DB
        self.history = []  
        self.state = self._initialize_state()

    def _initialize_state(self) -> AgentState:
        """Initialize the agent state"""
        return {
            "messages": [],
            "should_end": False,
            "context": {"product_db": self.product_db},
            "product_db": self.product_db,
            "history": []
        }

    def create_prompt(self, user_message: str) -> str:
        """Create a prompt for the agent"""
        conversation_history = "\n".join([
            f"{'Customer' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in self.history[-5:]  
        ])
        
        prompt = f"""You are a helpful retail assistant for an electronics store. Help customers with product information, 
        availability, and general inquiries. Be concise but friendly. When mentioning prices, always include the currency symbol ($).

        Available Products:
        {json.dumps(self.product_db, indent=2)}

        Previous Conversation:
        {conversation_history}

        Customer: {user_message}"""
        
        return prompt

    def chat(self, message: str) -> str:
        """Process a single message and return the response"""
        try:
            prompt = self.create_prompt(message)
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the response
            assistant_message = response.content[0].text
            
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def run_interactive_chat():
    """Run an interactive chat session in the terminal"""
    try:
        agent = RetailChatAgent()
        logger.info("Retail chat agent initialized")
        
        print("\n=== Welcome to the Electronics Store Assistant ===")
        print("Available products: laptops, smartphones, headphones, smartwatches, tablets, and cameras")
        print("Type 'quit' to end the conversation")
        print("================================================\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\nThank you for shopping with us! Goodbye!")
                    break
                
                response = agent.chat(user_input)
                print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for shopping with us! Goodbye!")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error during conversation: {str(e)}")
                print("An error occurred. Please try again.")
                
    except Exception as e:
        print(f"An error occurred during setup: {str(e)}")

if __name__ == "__main__":
    run_interactive_chat()