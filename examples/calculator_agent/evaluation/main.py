import os
from dotenv import load_dotenv

from evaluation.runner import EvalRunner, EvaluationTask
from evaluation.calc_evals import calculator_task_qwen3_0_6b, calculator_tool_parser


OUTPUT_DIRECTORY = "./evaluation/results"

if __name__ == "__main__":
    load_dotenv()

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    current_task: EvaluationTask = calculator_task_qwen3_0_6b
    current_parser = calculator_tool_parser

    runner = EvalRunner(tool_call_parser=current_parser)

    try:
        print(f"Running evaluation for task: {current_task.task_name}")
        results = runner.run_evaluation(
            task=current_task,
            output_dir=OUTPUT_DIRECTORY,
        )
        print("\nEvaluation script finished successfully.")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nEvaluation script finished with errors.")