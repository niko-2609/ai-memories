from dotenv import load_dotenv
from mem0 import MemoryClient, Memory
from openai import OpenAI
import os 

# Load env variables
load_dotenv()


# Create openAI client
open_ai_client = OpenAI()


# Create mem0 client 
client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))


# Create a config for memory client
config = {
    # Vector database
       "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "collection_name": "test",
            "port": 6333
        }
    }, 
    # LLM to use 
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    },
    # Model to use for creating vector embeddings
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
}


# Create memory using config
m = Memory.from_config(config)



# Chatbot
def chat():
    while True:
        user_query = input("> ")
        response = open_ai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": user_query}
            ]
        )
        print(f"🤖: {response.choices[0].message.content} ")


chat()






