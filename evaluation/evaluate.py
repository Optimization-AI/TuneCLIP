import argparse
import copy
import json
import os
import pickle
import re
import shutil
import time
import warnings
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import requests
import yaml
from eval_utils.main import evaluate_model
from scale_configs import available_scales, get_scale_config

warnings.filterwarnings("ignore", message="Length of IterableDataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_output_dir",
        required=True,
        help="Path to output directory from training.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
    )
    parser.add_argument(
        "--data_dir",
        help="(Optional) Path to directory containing downloaded evaluation datasets.",
        default=None,
    )
    parser.add_argument(
        "--use_model", 
        type=str, 
        required=True, 
        help="Model architecture and checkpoint, e.g., 'ViT-B/32 /path/to/checkpoint.pth'"
    )

    parser.add_argument("--model", type=str, default="ViT-B-32", help="Model architecture to use (e.g., ViT-B-32, ViT-L-14)")


    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")

    parser.add_argument("--task", default="eval", help = "name of the task")


    args = parser.parse_args()

    args.train_output_dir = Path(args.train_output_dir)
    if args.output_dir is None:
        args.output_dir = args.train_output_dir
    args.output_dir = Path(args.output_dir)

    if args.use_model is not None:
        args.train_output_dir = args.output_dir / args.task
        # Generate barebones info.pkl
        model_arch, model_checkpoint = args.use_model.split(maxsplit=1)
        Path.mkdir(args.train_output_dir, parents=True, exist_ok=True)
        with open(args.train_output_dir / "info.pkl", "wb") as f:
            pickle.dump(
                {"scale_config": {"model": model_arch}, "checkpoint": model_checkpoint},
                f,
            )


    # Read training information
    train_info_filename = args.output_dir / args.task / "info.pkl"
    train_info = pickle.load(open(train_info_filename, "rb"))

    results_filename = args.output_dir / f"{args.task}.jsonl"

    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f: 
        tasks = yaml.safe_load(f)

    # Check for cached results
    results = {}
    cached_train_info_filename = args.output_dir / args.task / "info.pkl"
    if args.output_dir.exists() and cached_train_info_filename.exists():
        # If the output directory already exists, the training information should match.
        cached_train_info = pickle.load(open(cached_train_info_filename, "rb"))
        error_message = (
            "Error: output directory exists, but the training configs do not match. "
            "If you are re-using an output directory for evals, please be sure that "
            "the training output directory is consistent."
        )
        assert cached_train_info == train_info, error_message

        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")
    else:
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        pickle.dump(train_info, open(cached_train_info_filename, "wb"))

    train_checkpoint = Path(train_info["checkpoint"])
    try:
        exists = Path(train_info["checkpoint"]).exists()
    except:
        exists = False
    if not exists and args.use_model is None:
        print(
            "Warning, did not find or could not read checkpoint at",
            train_info["checkpoint"],
        )
        default_checkpoint_name = (
            args.train_output_dir / "checkpoints" / "epoch_latest.pt"
        )
        print("Defaulting to", default_checkpoint_name)
        train_info["checkpoint"] = default_checkpoint_name

    print("Evaluating")

    starttime = int(time.time())

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(
                f"Skipping {task_name} since results are already in {results_filename}"
            )
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                train_info,
                args.data_dir,
                tasks[task_key].get("size"),
                batch_size=args.batch_size,
            )
            metrics["main_metric"] = metrics.get(
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")
    if args.submit:
        print("=====================")
        average = np.mean(
            [
                val["metrics"]["main_metric"]
                for val in results.values()
                if val["metrics"]["main_metric"] is not None
            ]
        )
        print(f"Average: {average}")