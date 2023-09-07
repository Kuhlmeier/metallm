import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
        self.model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16).to(self.device)

    def generate_text(self, input_text, max_len=500):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, max_length=max_len)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    engine = InferenceEngine()
    init_text = """CREATE TABLE work_order (
    id NUMBER,
    property_id NUMBER,
    cost FLOAT,
    invoice_amount FLOAT,i
    entered_date DATE,
    due_date DATE,
    complete_date DATE,
    cancel_date DATE,
    is_canceled BOOLEAN,
    is_completed BOOLEAN,
    is_open BOOLEAN,
    order_type TEXT,
    assigned_to TEXT
    )

    CREATE TABLE property (
        id NUMBER,
        property_name TEXT,
        area FLOAT,
        owner_id NUMBER,
        city TEXT,
        country TEXT
    )

    CREATE TABLE owner (
        id NUMBER,
        name TEXT,
        salary FLOAT
    )
    -- Using valid PostgreSQL, answer the following questions for the tables provided above.

    """
    query = f"{init_text}    -- What is the completion percentage in the work orders for each order type?" + "\n"+"SELECT"
    result = engine.generate_text(query)
    
    #input("Press Enter to continue...")

    queries = [
        f"{init_text} -- How many open work orders for the largest property in Chicago?" + "\n"+"SELECT",
        f"{init_text} -- What is the average work order cost for the properties by each owner?" + "\n"+"SELECT",
        f"{init_text} -- Who has the property that has the most number of work orders?" + "\n"+"SELECT",
        f"{init_text} -- How many work order got cancelled within 3 months before the due date?" + "\n"+"SELECT",
        f"{init_text} -- What is the expected profit of open work orders?" + "\n"+"SELECT",
        f"{init_text} -- What is the expected profit of open work orders? Profit is total difference of invoice and cost across orders." + "\n"+"SELECT"
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        result = engine.generate_text(query)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Query {i + 1} \nTime Taken: {elapsed_time:.4f} seconds\n{'-'*80}\n")
