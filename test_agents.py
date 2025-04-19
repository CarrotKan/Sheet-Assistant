import pandas as pd
from agents.tools import SheetTools

def test_tools():
    # Create a sample DataFrame
    data = {
        'Name': ['John', 'Jane', 'Bob'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']
    }
    df = pd.DataFrame(data)
    
    # Get tools
    tools = SheetTools.get_tools(df)
    
    # Test exact match search
    exact_match_tool = tools[0]
    result = exact_match_tool.run("John")
    print("Exact Match Result:", result)
    
    # Test column values
    column_values_tool = tools[1]
    result = column_values_tool.run("City")
    print("Column Values Result:", result)

if __name__ == "__main__":
    test_tools() 