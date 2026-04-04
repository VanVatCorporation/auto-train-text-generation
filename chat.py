import torch
from model import BigramLanguageModel, device, block_size
from data import Tokenizer
from memory_system import MemoryManager
import argparse
import sys

def chat():
    # 1. Load Checkpoint
    try:
        checkpoint = torch.load('model_ckpt.pt', map_location=device)
        chars = checkpoint['chars']
        vocab_size = checkpoint['vocab_size']
        model = BigramLanguageModel(vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        tokenizer = Tokenizer("".join(chars))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Checkpoint not found. Please run trainer.py first!")
        return

    # 2. Init Memory
    memory = MemoryManager()

    print("\n--- AI Chat with Persistent Learning ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # 3. Retrieve Memory
        facts = memory.retrieve_relevant_facts(user_input, n_results=1)
        memory_context = ""
        if facts:
            memory_context = f"Recall: {facts[0]}. "
        
        # 4. Prepare Context for Model
        # We'll feed: Facts : Input : 
        full_context_text = f"{memory_context}{user_input} : "
        
        # Encode
        context_ids = torch.tensor([tokenizer.encode(full_context_text)], dtype=torch.long, device=device)
        
        # 5. Generate Response
        # We generate until we see a newline or a certain length
        print("AI: ", end="", flush=True)
        # Using a simplistic generation loop to show characters as they appear
        generated_ids = model.generate(context_ids, max_new_tokens=50)
        
        # Decode only the NEW tokens
        new_tokens = generated_ids[0][len(context_ids[0]):]
        response_text = tokenizer.decode(new_tokens.tolist())
        
        # Simple cleanup - stop at first newline
        clean_response = response_text.split("\n")[0]
        print(clean_response)

        # 6. Auto-Learn
        if memory.auto_detect_fact(user_input):
            print("(AI has updated its knowledge base)")

if __name__ == "__main__":
    chat()
