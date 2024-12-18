import os
import pdb
import time
import json
import tqdm
import argparse

from openai import OpenAI
from model import Solver, Controller, Caller, Verifier
from process import str2bool


def parse_args():
    """Add args."""
    parser = argparse.ArgumentParser()

    # LLM
    parser.add_argument("--KEY", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n", type=int, default=1)

    # files
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--dataset", type=str, default="fund")
    parser.add_argument("--result_dir", type=str, default="../../data/results")
    parser.add_argument("--formulas_dir", type=str, default="../../textbooks")
    # parser.add_argument("--examples_dir", type=str, default="../../data/ours")
    parser.add_argument("--prompt_verifier_dir", type=str, default="prompts")
    parser.add_argument("--prompt_solver_dir", type=str, default="prompts")

    # method
    parser.add_argument("--actions", type=list, default=["extract_variables", "retrieve_formula", 
                                                         "stepwise_calculate", "terminate", "judge_result"])
    
    # whether to store
    parser.add_argument("--debug", action="store_true")

    # whether to load existing results
    parser.add_argument("--load_results", type=str2bool, default=True)

    args = parser.parse_args()
    return args

def reason():
    args = parse_args()

    # build framework
    verifier = Verifier(args)
    solver = Solver(args, verifier)
    controller = Controller(args)

    # iteratively reason
    for pid in tqdm.tqdm(solver.pids):
        # build problem and clear cache
        solver.build_state(pid)
        print('\nSolving problem {}...'.format(solver.state['pid']))

        # iterate until the problem is solved
        while True:
            # controller: predict next action
            action = controller.predict_action(solver)

            # acter: call corresponding functions
            if action:
                solver.call(action)
            else:
                time.sleep(0.1)
                break


if __name__ == "__main__":
    reason()