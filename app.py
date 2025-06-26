import json
from dotenv import load_dotenv
from mem0 import MemoryClient, Memory
from openai import OpenAI
import os

# Load env variables
load_dotenv()


# Create openAI client
open_ai_client = OpenAI()


# # Create mem0 client
# mem_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))


# Create a config for memory client
config = {
    # Vector database
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "collection_name": "test", "port": 6333},
    },
    # LLM to use
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4.1-nano", "api_key": os.getenv("OPENAI_API_KEY")},
    },
    # Model to use for creating vector embeddings
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
}


# Create memory using config
mem_client = Memory.from_config(config)


# Chatbot
def chat():
    while True:
        user_query = input("> ")

        # Fetch all relevant memories of the user
        # Returns an object containing an array under "results" key
        # Each item in the array has a memory id, actual memory and a user id
        relevant_memories = mem_client.search(query=user_query, user_id="rishabh")

        # Format the memories with required data and store in an array
        memories = [
            f"ID: {mem.get("id")} Message: {mem.get("memory")}"
            for mem in relevant_memories["results"]
        ]

        SYSTEM_PROMPT = f"""
            You're a memory aware AI assistant that responds to user queries with context.
            You're given with past memories and facts about the user.

            Memory of the user: {json.dumps(memories)}
        """


        response = open_ai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
        print(f"🤖: {response.choices[0].message.content} ")

        # This will now internally add this message to the memory.
        # Memory client will create vector embeddings of the messages and store them in a vector db.
        mem_client.add(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response.choices[0].message.content},
            ],
            user_id="rishabh",
        )


chat()
