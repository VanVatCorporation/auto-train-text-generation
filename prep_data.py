import random

def prepare_instruction_data(output_file='data_instruct.txt', num_rows=100):
    print(f"Generating {num_rows} synthetic reasoning samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0
        while count < num_rows:
            # Generate random math problems
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
