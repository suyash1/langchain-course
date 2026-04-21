from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, ai
from langsmith import traceable

MAX_ITERATIONS = 10

MODEL = 'qwen3.5:2b'

# ------------ LangChain Tool Calling ------------
@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product"""
    print(">>> Executing get_product_price tool... for product: {product_name}")
    prices = {'laptop': 1000, 'phone': 500, 'tablet': 300}
    return prices.get(product_name, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount ties to a price and retuen the final discounted price"""
    print(">>> Executing apply_discount tool... for price: {price} with discount tier: {discount_tier}")
    discount_percentages = {
        'bronze': 5,
        'silver': 12,
        'gold': 23,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


#----------Agent Loop ------------
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm = init_chat_model(f"gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Question: {question}")
    messages = [
        SystemMessage(content=
        "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool. Search for the product in the catalog and apply the discount to the price.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
                "5. If the product is not found in the catalog, say 'Product not found in catalog.' and not proceed to apply discount."
                "You can identify if the product is not found in the catalog by checking if the price is 0."),
        HumanMessage(content=question),
    ]
    for iteration in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')

        print(f"\nTool Call Iteration:{iteration + 1}\n Tool Selected {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"\nTool {tool_name} not found in tools dictionary. Skipping...")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"\nTool Result: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

    print("Max Iterations Reached without final answer.")
            


if __name__ == "__main__":
    print("Hello from LangChain Agent (.bind_tools)")
    print("----------------------------------------\n")
    result = run_agent("What is the price of 'phone' after bronze discount?")
    print(result)