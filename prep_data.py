import random

def prepare_instruction_data(output_file='data_instruct.txt', num_rows=500):
    print(f"Generating {num_rows} synthetic reasoning & chat samples...")
    
    greetings = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there", "Hi! What's on your mind?"),
        ("Good morning", "Good morning! I am ready to assist you."),
        ("Who are you?", "I am a text generation AI built from scratch."),
        ("How are you?", "I am just a computer program, but I'm doing great!"),
        ("What can you do?", "I can answer questions, do math, and learn facts."),
        ("Hey", "Hey there! Need any help?"),
        ("Thanks", "You're welcome! Let me know if you need anything else.")
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0
        while count < num_rows:
            # Randomly pick math or chat (50/50)
            if random.random() < 0.5:
                # Chat logic
                prob, sol = random.choice(greetings)
                problem = prob
                solution = sol
                thinking = "The user is greeting me or asking a simple conversational question. I should respond politely."
                ans = sol # For fake recall compatibility
            else:
                # Math logic
                op = random.choice(['+', '-', '*'])
                a = random.randint(1, 20)
                b = random.randint(1, 20)
                
                if op == '+':
                    ans = a + b
                    thinking = f"First, I need to add {a} and {b}. The sum of {a} and {b} is {ans}."
                elif op == '-':
                    if a < b: a, b = b, a # keep it positive
                    ans = a - b
                    thinking = f"I need to subtract {b} from {a}. {a} minus {b} equals {ans}."
                else:
                    a = random.randint(1, 10) # keep it small
                    b = random.randint(1, 10)
                    ans = a * b
                    thinking = f"The operation is multiplication. {a} times {b} equals {a*b}."
                    
                problem = f"What is {a} {op} {b}?"
                solution = f"The answer is {ans}."

            # Occasionally add a "Fake Recall" so the model learns to use it
            recall_text = ""
            if random.random() < 0.3:
                recall_text = f"<|recall|> Information: The result involves {ans}... <|end|> "

            # Special Tokens Template:
            # <|user|> prompt <|end|> <|thought|> reasoning <|end|> <|assistant|> answer <|end|>
            entry = (
                f"{recall_text}"
                f"<|user|> {problem} <|end|> "
                f"<|thought|> {thinking} <|end|> "
                f"<|assistant|> {solution} <|end|>\n"
            )
            f.write(entry)
            count += 1
            
    print(f"Successfully prepared {count} rows in {output_file}")

if __name__ == "__main__":
    prepare_instruction_data()
