import os
import sys
import subprocess
import argparse
import numpy as np
import shutil
import glob
import torch

# ensure project root on path (parent of this 'lstm' folder)
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# import training and evaluation helpers
from lstm.lstm_transfer_train import train_baseline_and_finetune
from lstm import evaluate_on_file as eval_mod
# import the simple trainer
from lstm.lstm_train import train_lstm_and_iterative_eval

def main():
    parser = argparse.ArgumentParser(description="Train -> evaluate_on_file -> run resulttest")
    # mode: 'transfer' or 'simple'
    parser.add_argument("--mode", type=str, default="transfer", choices=["transfer","simple"], help="Training mode")
    parser.add_argument("--data_path", type=str, default="data/ohio/2018", help="Base path to ohio 2018 data")
    parser.add_argument("--train_folder", type=str, default="train_cleaned", help="Folder train_cleaned/")
    parser.add_argument("--finetune_file", type=str, default="559-ws-training.csv", help="Filename inside train_cleaned/")
    parser.add_argument("--test_folder", type=str, default="test_cleaned", help="Folder test_cleaned/")
    parser.add_argument("--test_file", type=str, default="559-ws-testing.csv", help="Filename inside test_cleaned/")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs_baseline", type=int, default=25)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    parser.add_argument("--epochs_simple", type=int, default=15, help="Epochs for simple trainer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_horizon", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default="lstm/models_lstm")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="Optional: skip training and use this checkpoint for evaluation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # determine device once and reuse for loading/evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create run folder with timestamp and subfolders for models+eval
    import datetime
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    run_base = os.path.join(run_dir, "base_line")
    run_finetune = os.path.join(run_dir, "fine_tuned")
    run_single = os.path.join(run_dir, "single")
    os.makedirs(run_base, exist_ok=True)
    os.makedirs(run_finetune, exist_ok=True)
    os.makedirs(run_single, exist_ok=True)

     # build full train/test directories and file paths
    train_dir = os.path.join(args.data_path, args.train_folder)
    test_dir = os.path.join(args.data_path, args.test_folder)
    train_file_path = os.path.join(train_dir, args.finetune_file)
    test_file_path = os.path.join(test_dir, args.test_file)

    # training phase
    if args.model_ckpt is None:
        if args.mode == "transfer":
            if not os.path.exists(train_file_path):
                print(f"Error: {args.finetune_file} not found in directory {train_dir}.")
                print("Either place the CSV there or run with --model_ckpt to skip training.")
                sys.exit(1)

            print("Starting transfer training / fine-tuning ...")
            # pass run_dir so train saves into run_dir/{base_line,fine_tuned}
            train_baseline_and_finetune(
                data_dir=train_dir,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                epochs_baseline=args.epochs_baseline,
                epochs_finetune=args.epochs_finetune,
                lr=args.lr,
                finetune_file=args.finetune_file,
                out_dir=run_dir
            )
            # check that baseline model was created in run_base
            base_model_path = os.path.join(run_base, "baseline_model.pth")
            if not os.path.exists(base_model_path):
                print("Baseline model not found after transfer training, retraining baseline only")
                train_baseline_and_finetune(
                    data_dir=train_dir,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    epochs_baseline=args.epochs_baseline,
                    epochs_finetune=0,
                    lr=args.lr,
                    finetune_file=args.finetune_file,
                    out_dir=run_dir
                )
                if not os.path.exists(base_model_path):
                    print("Error: baseline model could not be created:", base_model_path)
                else:
                    print("Baseline model:", base_model_path)
            else:
                print("Baseline model already exists:", base_model_path)

            # train the single (non-transfer) model and store in run_single
            train_lstm_and_iterative_eval(
                data_dir=train_dir,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                epochs=args.epochs_simple,
                lr=args.lr,
                target_file=args.finetune_file,
                max_horizon=args.max_horizon,
                out_dir=run_dir
            )
            single_ckpt = os.path.join(run_single, f"lstm_model_simple_{args.finetune_file.replace('.csv','')}.pth")
        else:
            # simple mode
            if not os.path.exists(train_file_path):
                print(f"Error: {args.finetune_file} not found in directory {train_dir}.")
                sys.exit(1)

            print("Starte einfaches LSTM-Training ...")
            train_lstm_and_iterative_eval(
                data_dir=train_dir,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                epochs=args.epochs_simple,
                lr=args.lr,
                target_file=args.finetune_file,
                max_horizon=args.max_horizon,
                out_dir=run_dir  # call simple trainer with out_dir=run_dir so it creates run_dir/single
            )
            ckpt_path = os.path.join(run_single, f"lstm_model_simple_{args.finetune_file.replace('.csv','')}.pth")
    else:
        ckpt_path = args.model_ckpt

    # Determine which checkpoint to use for downstream evaluation
    if args.model_ckpt is not None:
        ckpt_path = args.model_ckpt
    else:
        if args.mode == "transfer":
            # prefer fine_tuned checkpoint (created in run_finetune)
            ckpt_path = os.path.join(run_finetune, f"lstm_model_finetuned_{args.finetune_file.replace('.csv','')}.pth")
        else:
            # simple mode -> single model path
            ckpt_path = os.path.join(run_single, f"lstm_model_simple_{args.finetune_file.replace('.csv','')}.pth")

    if not os.path.exists(ckpt_path):
        print("Checkpoint not found:", ckpt_path)
        sys.exit(1)

    # evaluate available checkpoints on test_file_path
    print("Evaluating available checkpoints on test file ...")
    ckpt_candidates = {
        "base_line": os.path.join(run_base, "baseline_model.pth"),
        "fine_tuned": os.path.join(run_finetune, f"lstm_model_finetuned_{args.finetune_file.replace('.csv','')}.pth"),
        "single": os.path.join(run_single, f"lstm_model_simple_{args.finetune_file.replace('.csv','')}.pth")
    }

    eval_csvs = []
    for name, ckpt in ckpt_candidates.items():
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found for {name}: {ckpt} (skipping)")
            continue
        try:
            print(f"Loading checkpoint for {name}: {ckpt}")
            # load model directly onto selected device
            model, scaler = eval_mod._load_checkpoint_model(ckpt, device=device)
            print(f"Running evaluate_on_file for {name} -> saving into {os.path.dirname(ckpt)}")
            target_out_dir = os.path.dirname(ckpt)
            # model is already on `device`; pass same device to evaluation
            eval_mod.evaluate_on_file(model, test_file_path, scaler,
                                      seq_len=args.seq_len, device=device,
                                      max_horizon=args.max_horizon, out_dir=target_out_dir)
            # eval_on_file names: eval_{base_name}_{model_tag}_all{max_horizon}.csv
            base_name = os.path.basename(test_file_path).replace('.csv', '')
            model_tag = os.path.basename(os.path.normpath(target_out_dir))
            eval_csv = os.path.join(target_out_dir, f"eval_{base_name}_{model_tag}_all{args.max_horizon}.csv")
            if os.path.exists(eval_csv):
                eval_csvs.append(eval_csv)
            else:
                print(f"Expected eval CSV not found after evaluation: {eval_csv}")
        except Exception as e:
            print(f"Failed to evaluate checkpoint {ckpt}: {e}")

    # if a single external checkpoint was explicitly passed, evaluate it too (if not already)
    if args.model_ckpt:
        if os.path.exists(args.model_ckpt) and args.model_ckpt not in ckpt_candidates.values():
            try:
                print(f"Also evaluating provided --model_ckpt: {args.model_ckpt}")
                model, scaler = eval_mod._load_checkpoint_model(args.model_ckpt, device=device)
                eval_mod.evaluate_on_file(model, test_file_path, scaler,
                                          seq_len=args.seq_len, device=device,
                                          max_horizon=args.max_horizon, out_dir=os.path.dirname(args.model_ckpt))
                eval_csv = os.path.join(os.path.dirname(args.model_ckpt),
                                        f"eval_{os.path.basename(test_file_path).replace('.csv','')}_all{args.max_horizon}.csv")
                if os.path.exists(eval_csv):
                    eval_csvs.append(eval_csv)
            except Exception as e:
                print("Failed to evaluate provided model_ckpt:", e)

    # Run resulttest on the produced eval CSVs
    # collect any eval_*.csv files under each model folder (handles both no_future/with_future files)
    eval_csvs = []
    candidates = {}
    for name, folder in (("base_line", run_base), ("fine_tuned", run_finetune), ("single", run_single)):
        if os.path.isdir(folder):
            found = sorted(glob.glob(os.path.join(folder, "eval_*.csv")))
            candidates[name] = found
            eval_csvs.extend(found)
        else:
            candidates[name] = []
    
    pretty = {k: [os.path.basename(p) for p in v] for k, v in candidates.items()}
    print("Found eval CSVs per folder:", pretty)

    if not eval_csvs:
        print("No eval CSVs found to compare.")
    else:
        try:
            from lstm import resulttest
            comp_res = resulttest.results(eval_csvs, out_path=os.path.join(run_dir, "comparison_metrics"))
        except Exception as e:
            print("Failed to run combined resulttest (import):", e)
            # fallback to CLI invocation
            subprocess.check_call([sys.executable, os.path.join(proj_root, "lstm", "resulttest.py")] + eval_csvs)
            fallback_dir = os.path.join(os.path.dirname(eval_csvs[0]), "comparison_metrics")
            if os.path.exists(fallback_dir):
                target = os.path.join(run_dir, "comparison_metrics")
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.move(fallback_dir, target)

    print("Done.")

if __name__ == "__main__":
    main()