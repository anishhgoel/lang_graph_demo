from typing import Dict, List, TypedDict, Tuple, Optional
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

class AgentState(TypedDict):
    messages: List[Dict]
    current_step: str
    context: Dict
    product_db: Dict
    query_type: Optional[str]
    mentioned_products: List[str]
    completed: bool

class RetailChatAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = Anthropic(api_key=self.api_key)
        self.product_db = PRODUCT_DB
        self.graph = self._create_chat_graph()
        self.chain = self.graph.compile()

    def process_step(self, state: AgentState) -> AgentState:
        """Process a single step of the conversation"""
        try:
            prompt = f"""You are a helpful retail assistant. Help customers with product information, 
            availability, and general inquiries. Be concise but friendly.

            Available Products:
            {json.dumps(self.product_db, indent=2)}

            Previous Conversation:
            {self._format_conversation_history(state['messages'][:-1])}

            Current Customer Message:
            {state['messages'][-1]['content']}"""

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content[0].text
            })
            state["completed"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error. Please try again."
            })
            state["completed"] = True
            return state

    def _format_conversation_history(self, messages: List[Dict]) -> str:
        """Format conversation history"""
        return "\n".join([
            f"{'Customer' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in messages
        ])

    def should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        if state.get("completed", False):
            return "end"
        return "process"

    def _create_chat_graph(self) -> StateGraph:
        """Create a simple conversation flow graph"""
        workflow = StateGraph(AgentState)
        
        # single processing node
        workflow.add_node("process", self.process_step)
        
        #  conditional edges
        workflow.add_conditional_edges(
            "process",
            self.should_continue,
            {
                "process": "process",
                "end": END
            }
        )
        
        workflow.set_entry_point("process")
        return workflow

    def chat(self, message: str, state: Optional[AgentState] = None) -> AgentState:
        """Handle a single message"""
        if state is None:
            state = {
                "messages": [],
                "current_step": "process",
                "context": {"product_db": self.product_db},
                "product_db": self.product_db,
                "query_type": None,
                "mentioned_products": [],
                "completed": False
            }
        else:
            state["completed"] = False
        
        state["messages"].append({
            "role": "user",
            "content": message
        })
        
        return self.chain.invoke(state)

def run_interactive_chat():
    """Run an interactive chat session in the terminal"""
    try:
        agent = RetailChatAgent()
        logger.info("Retail chat agent initialized")
        
        print("\n=== Welcome to the Electronics Store Assistant ===")
        print("Available products: laptops, smartphones, headphones, smartwatches, tablets, and cameras")
        print("Type 'quit' to end the conversation")
        print("================================================\n")
        
        state = None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\nThank you for shopping with us! Goodbye!")
                    break
                
                state = agent.chat(user_input, state)
                print(f"\nAssistant: {state['messages'][-1]['content']}")
                    
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