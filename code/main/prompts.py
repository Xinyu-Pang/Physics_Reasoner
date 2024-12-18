prompts = {
    # extract variables
    "prompt_ev": """Given a physics problem, please extract all known variables in the form of Python code. Encase the Python code within triple backticks for clarity. The problem will specify the unit of measurement. Annotate with variable meanings and units afterward.""",
    
    # retrieve formulas
    "prompt_fr": """Given a physics problem, please choose possible related formula(s) from all formulas, and return in the format of Python code annotation. Encase the Python code within triple backticks for clarity. If there exists no related formulae, just return an empty string.""", 

    # generate python code
    "prompt_gc": """Given a physics problem and its corresponding incomplete Python code, please generate the next line of the code. Ensure the Python code is encased within triple backticks for clarity. You can either generate an intermediate variable or print the target variable if the problem has been solved. And make sure each variable or function has been defined or imported.""",

    # review and refine
    "prompt_review": """Given the generated Python code for a physics problem, please verify the following:
1. Check if all formulas are correctly implemented.
2. Ensure that all parameters required for the calculations are included.
3. Confirm that the code logic is sound and adheres to standard practices.
4. Encase the output is transferred to target unit.

If there are any errors or missing parameters, correct the code and provide the updated code. Else, if the code is already correct, return the correct code. Encase the Python code within triple backticks for clarity.
""" 
}