from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 5
MODEL = "qwen3:latest"

@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
    }
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier."""
    discounts = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount_rate = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount_rate/100), 2)

#Agent loop
def run_agent(question: str):
   tools = [get_product_price, apply_discount]
   tools_dict = {t.name: t for t in tools}
   llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
   llm_with_tools = llm.bind_tools(tools)
   messages =[
       SystemMessage(
           content=("You are a helpful shopping assistant."
                    "You have access to product catalog tool and discount tool."
                    
                    "Strict rules - You must follow these exactly:"
                    "1. Never guess or assume any product price."
                    "you must call get_product_price tool to get the price of a product."
                    "2. Only call apply_discount AFTER you have received"
                    "the price of a product from get_product_price tool. do not pass any made up number"
                    "3. Never calculate the discount yourself using math."
                    "Always call the apply_discount tool."
                    "4. If the user does not specify a discount tier, ask them to choose one - do not assume")
       ),
       HumanMessage(content=question)
   ]

   for iteration in range(1, MAX_ITERATIONS + 1):
       print(f"\n--- Iteration {iteration} ---")
       ai_message = llm_with_tools.invoke(messages)
       tool_calls = ai_message.tool_calls
       if not tool_calls:
           print("No tool calls made. Ending loop.")
           return ai_message.content
       for tool_call in tool_calls:
           tool_name = tool_call.get("name")
           tool_args = tool_call.get("args",{})
           tool_id = tool_call.get("id")
           print(f"Tool called: {tool_name} with args: {tool_args}")
           tool_to_use = tools_dict.get(tool_name)
           if tool_to_use is None:
               raise ValueError(f"Tool {tool_name} not found.")
           
           observation = tool_to_use.invoke(tool_args)
           print(f"Observation from tool: {observation}")
           
           messages.append(ai_message)
           messages.append(ToolMessage(content=observation, tool_call_id=tool_id))
    
print("ERROR:Max iterations reached without a final answer.")

def main():
    print("Hello from langchain-course! (1_agent_loop_langchain_tool_calling.py)")
    print("This example demonstrates an agent loop that calls tools to answer a question.")
    question = "What is the price of a laptop after applying a gold discount?"
    result = run_agent(question)
    print(f"Final answer: {result}")


if __name__ == "__main__":
    main()
