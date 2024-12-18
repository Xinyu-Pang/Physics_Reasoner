import io
import os
import ast
import sys
import pdb
import json
import time
import math
import numpy as np
import subprocess
import requests
from openai import OpenAI

from process import *


class Caller():
    def __init__(self, args):
        self.model = args.engine
        self.KEY = args.KEY
    
    def call_engine(self, messages, n=1, temperature=1e-30):
        """Fetch response from LLM with prompt."""
        client = OpenAI(api_key = self.KEY)
        try:
            response = client.chat.completions.create(
                    messages = messages,
                    model = self.model,
                    n=n,
                    temperature= temperature)
            if n == 1:
                p = response.choices[0].message.content.strip()
            else:
                p = [choice.message.content.strip() for choice in response.choices]
            return p
        except Exception as e:
            print(e)
            pdb.set_trace()
            return


class Solver():
    """Modular physics reasoning framework."""

    def __init__(self, args, verifier):
        """Init Solver class."""
        self.caller = Caller(args)

        # score
        self.verifier = verifier

        # load prompt
        self.load_prompts(args)

        # file path
        self.data_file = os.path.join(args.data_dir, f"scibench/{args.dataset}.json")    # test problems
        time_now = time.strftime("%m%d", time.localtime())
        if args.debug:
            os.makedirs(os.path.join(args.result_dir, time_now), exist_ok=True)
        self.cache_file = f"{args.result_dir}/{time_now}/{args.dataset}_cache_{time_now}_{args.engine}.json"    # existing caches
        self.result_file = f"{args.result_dir}/{time_now}/{args.dataset}_result_{time_now}_{args.engine}.json"    # existing results

        # accuracy calculation
        for k in ["count", "correct", "wrong"]:
            setattr(self, k, 0)
        self.caches = []

        # whether to store
        self.debug = args.debug
        
        # load data
        self.example_num = 4 if args.dataset != "thermo" else 3
        self.load_data(args)

        # max calculation steps
        self.max_steps = 1
    
    def load_data(self, args):
        """Load existing result and problems."""
        # load problems
        self.problems = json.load(open(self.data_file, "r", encoding="utf-8"))
        self.pids = list(range(len(self.problems)))
        
        # load existing results
        if os.path.exists(self.cache_file) and args.load_results:
            self.caches = json.load(open(self.cache_file, "r", encoding="utf-8"))
            solved_pids = [i['pid'] for i in self.caches]
        else:
            solved_pids = []
        
        if os.path.exists(self.result_file) and args.load_results:
            results = json.load(open(self.result_file, "r", encoding="utf-8")).split("|")
            self.count = int(results[2].replace("COUNT:", "").strip())
            self.correct = int(results[0].replace("Correct:", "").strip())
            self.wrong = int(results[1].replace("WRONG:", "").strip())
        
        self.pids = [p for p in self.pids if p not in solved_pids]
        
        # print number of problems
        print(f"# Number of test problems: {len(self.pids)}\n")

        # load formulas
        self.formulas = json.load(open(os.path.join(args.formulas_dir, args.dataset, '{}_formulas.json'.format(args.dataset)), "r", encoding="utf-8"))

    def load_prompts(self, args):
        """Load system prompts."""
        self.prompts = json.load(open(os.path.join(args.prompt_solver_dir, 'prompt_solver_{}.json'.format(args.dataset)), 'r', encoding='utf-8'))

    def build_state(self, pid):
        """Build state for the problem to be solved."""
        self.state = {"pid": pid, "problem": self.problems[pid], "previous actions": {}, "history": {}}
        
        # process correct answer with exponentiation
        if remove_not(self.state['problem']["unit"]):
            self.state['problem']["answer_number"] = cal_not((self.state['problem']["answer_number"], self.state['problem']["unit"]))    # 结合10的次幂的结果
        
        # process unit in latex format
        self.state['problem']['unit'] = process_unit(self.state['problem']['unit'])

        # annotate unit in the problem text
        self.state['problem']['problem_text'] += ' The unit of the answer is {}'.format(self.state['problem']['unit'])

    def extract_variables(self):
        """Extract all known variables from problem text."""
        print("Extracting variables...")
        
        messages = self.prompts['prompt_ext_vars'][:]
        messages += [
            {"role": "user", "content": "**Problem**: {}".format(self.state["problem"]["problem_text"])}
        ]

        variables = self.caller.call_engine(messages)
        variables = process_code(variables)
        
        # review
        review, out_code, _, _ = self.verifier.review_and_revise(self.state["problem"]["problem_text"], variables, flag='variables')
        # review = ""
        # out_code = variables
        self.state['history']['variables'] = {'review': review, 'previous code': variables, 'refined code': out_code}
        
        # update state
        if out_code != "":
            self.state["code"] = out_code
        else:
            self.state["code"] = variables
        return 

    def retrieve_formula(self):
        """Find possible related formulae."""
        print("Retrieving formulas...")
        
        # subfileds
        messages = self.prompts["prompt_retr_subfields"][:]
        messages += [
            {"role": "user", "content": "**Problem**: {}".format(self.state["problem"]["problem_text"])}]
        subfield = self.caller.call_engine(messages)

        # retrieve formula
        messages = self.prompts['prompt_retr_for'][:]
        formulas_contents = {}
        for f in self.formulas.keys():
            if f.lower() in subfield.lower():
                try:
                    formulas_contents[f] = extract_formula_content(self.formulas[f])
                except:
                    pdb.set_trace()
        if formulas_contents != {}:
            messages += [
                {"role": "user", "content": "**Problem**: {} **Possible formulas**: {}".format(self.state["problem"]["problem_text"], formulas_contents)}]
            r = self.caller.call_engine(messages)
            formula = process_code(r)
        else:
            formula = ""
        # update state
        self.state["code"] += "\n" + formula

        return 

    def stepwise_calculate(self):
        try:
            print("Generating code...")
            code = self.generate_python_code()  
            self.state['code'] = code

        except Exception as e:
            print(e)
            pdb.set_trace()
            return

        return

    def generate_python_code(self):
        """Generate Python code."""
        messages = self.prompts['prompt_gen_code'][:]
        messages += [
            {"role": "user", "content": "**Problem**: {} **Incomplete python code**: ```{}```"
            .format(self.state["problem"]["problem_text"], self.state["code"])}]
        
        code = self.caller.call_engine(messages)
        code = process_code(code)
        
        return code
    
    def verify_code(self, code):
        """Verify whether the generated code can be executed successfully."""
        prefix = "import numpy as np\nimport math\nimport sympy as sp\nfrom scipy.optimize import fsolve\n"
        
        if not isinstance(code, str) or "print(" not in code:
            return None
        
        code = prefix + code

        old_stdout = sys.stdout
        captured_output = sys.stdout = io.StringIO()
        try:
            exec(code, globals())
            output_str = captured_output.getvalue().strip()
            pred_ans = process_output(output_str)
        except Exception as _: 
            print("Problem {} fail!".format(self.state["pid"]))
            return None
        finally: 
            sys.stdout = old_stdout
        
        try:
            return float(pred_ans)
        except:
            return "None"

    def terminate(self):
        """Judge whether to terminate process."""

        if "print(" in self.state["code"]:
            return True
        else:
            return False

    def judge_result(self):
        """Judge prediction accuracy with correct answer of the problem using Numbat."""
        
        # verify with checklist
        review, out_code, _, _ = self.verifier.review_and_revise(self.state["problem"]["problem_text"], self.state["code"], flag='code')
        self.state['history']['code'] = {'review': review, 'previous code': self.state['code'], 'refined code': out_code}
        if out_code != "":
            self.state["code"] = out_code

        self.state["problem"]["pred_ans"] = self.verify_code(self.state["code"])

        # evaluate accuracy
        try:
            correct_ans = float(self.state["problem"]["answer_number"])
            pred_ans = float(self.state["problem"]["pred_ans"])
            flag_c = True
        except:
            flag_c = False

        try:
            flag_p = math.isclose(pred_ans, correct_ans, rel_tol=0.05)
        except:
            flag_p = False
        
        # update state
        flag = flag_c and flag_p
        self.state["problem"]["correctness"] = flag
        self.caches.append(self.state)
        self.count += 1
        if flag:
            self.correct += 1
        else:
            self.wrong += 1

        if self.debug:
            with open(self.cache_file, "w", encoding="utf-8") as f_c:
                json.dump(self.caches, f_c, indent=4, ensure_ascii=False)
            with open(self.result_file, "w", encoding="utf-8") as f_r:
                json.dump("Correct: {} | WRONG: {} | COUNT: {} | ACC: {}\n".format(self.correct, self.wrong, self.count, self.correct/self.count), f_r, indent=4, ensure_ascii=False)
        return

    def call(self, action):
        """Call corresponding function.
        
        Args:
            action: a string representing the type of next action."""
        # call function
        getattr(self, action)()

        # update state
        self.state["previous actions"][len(self.state["previous actions"])] = action

        return


class Controller():
    """Controller module of scientific reasoning framework.

    Attributes:
        actions: list of possible actions.
    """
    def __init__(self, args):
        self.actions = args.actions

    def predict_action(self, solver):
        """Based on the observed state, predict the next action.
        
        Args:
            solver.state: a dict including previous actions(list), problem, and process.
        
        Returns:
            action: predicted next action. If the problem is solved, return None.
        """
        executed_actions = list(solver.state["previous actions"].values())
        
        # first, extract variables and retrieve formula
        for action in ["extract_variables", "retrieve_formula"]:
            if action not in executed_actions:
                return action
            
        # then, calculate step by step, using python or wolfram
        if "judge_result" not in executed_actions:
            if not solver.terminate():
                if "stepwise_calculate" not in executed_actions:
                    return "stepwise_calculate"
            return 'judge_result'
        else:
            return None


class Verifier():
    """Judge distance between intermediate variable and target variable."""
    def __init__(self, args):
        self.prompt = json.load(open(os.path.join(args.prompt_verifier_dir, 'prompt_verifier_{}.json'.format(args.dataset)), 'r', encoding='utf-8'))
        self.caller = Caller(args)
    
    def review_and_revise(self, problem_text, code, flag):
        prompt_key = 'prompt_{}'.format(flag)
        try:
            messages = self.prompt[prompt_key][:]
        except Exception as e:
            print(e)
            pdb.set_trace()
            return
        messages += [
            {'role': 'user', 'content': '**Problem**: {} **code**: ```{}```'.format(problem_text, code)}
        ]
        r = self.caller.call_engine(messages)

        review, out_code = process_review(r)

        return review, out_code, messages, r