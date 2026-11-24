# run_ddrmgrid.py
import argparse, subprocess, os, sys, shlex

ETA_VALUES  = [0.7, 0.8, 0.9, 1.0]
ETAB_VALUES = [0.7, 0.8, 0.9, 1.0]

def parse():
    parser = argparse.ArgumentParser()
    # required (match your main.py)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deg", required=True)
    parser.add_argument("--sigma_0", type=float, required=True)

    # common passthroughs
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--verbose", type=str, default="info")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--subset_start", type=int, default=-1)
    parser.add_argument("--subset_end", type=int, default=-1)
    parser.add_argument("--ni", action="store_true")
    parser.add_argument("--noise_setting", type=str, default=None)
    parser.add_argument("--noise_model", type=str, default=None)
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )

    # defaults for the *other* parameter while sweeping
    parser.add_argument("--eta_default", type=float, default=0.85, help="Used when sweeping etaB")
    parser.add_argument("--etaB_default", type=float, default=1.0,  help="Used when sweeping eta")
    return parser.parse_args()

def run(cmd, env, exp_name):
    print(">>", shlex.join(cmd), flush=True)
    
    # Create experiment directory if it doesn't exist
    os.makedirs(exp_name, exist_ok=True)
    results_file = os.path.join(exp_name, "results.txt")
    
    # Open results.txt in the experiment directory
    with open(results_file, "w") as f:
        f.write(f"=== {exp_name} ===\n")
        f.write(f"Command: {shlex.join(cmd)}\n")
        f.flush()
        
        # Run the command with real-time output to both terminal and file
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1, universal_newlines=True)
        
        # Read output line by line and write to both terminal and file
        for line in proc.stdout:
            print(line, end='', flush=True)  # Print to terminal
            f.write(line)                    # Write to file
            f.flush()
        
        proc.wait()
        f.write(f"\nExit code: {proc.returncode}\n")
    
    return proc.returncode

def make_common_cmd(args):
    base = [
        sys.executable, "main.py",
        "--config", args.config,
        "--deg", args.deg,
        "--sigma_0", f"{args.sigma_0}",
        "--timesteps", str(args.timesteps),
        "--seed", str(args.seed),
        "--verbose", args.verbose,
        "--subset_start", str(args.subset_start),
        "--subset_end", str(args.subset_end),
    ]
    if args.comment: base += ["--comment", args.comment]
    if args.ni:      base += ["--ni"]
    if args.noise_setting is not None: base += ["--noise_setting", args.noise_setting]
    if args.noise_model   is not None: base += ["--noise_model",   args.noise_model]
    return base


def main():
    args = parse()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    results = []

    # Sweep η (eta), fix etaB to etaB_default
    for eta in ETA_VALUES:
        exp = f"ddrmgrid_imagenet_eta_{eta:.2f}"
        cmd = make_common_cmd(args) + [
            "--exp", exp,
            "--doc", exp,
            "--image_folder", exp,
            "--eta", f"{eta}",
            "--etaB", f"{args.etaB_default}",
        ]
        rc = run(cmd, env, exp)
        results.append((exp, rc))

    # Sweep η_b (etaB), fix eta to eta_default
    for etaB in ETAB_VALUES:
        exp = f"ddrmgrid_imagenet_etaB_{etaB:.2f}"
        cmd = make_common_cmd(args) + [
            "--exp", exp,
            "--doc", exp,
            "--image_folder", exp,
            "--eta", f"{args.eta_default}",
            "--etaB", f"{etaB}",
        ]
        rc = run(cmd, env, exp)
        results.append((exp, rc))

    print("\n=== SUMMARY ===")
    for exp, rc in results:
        print(f"{exp}: {'OK' if rc == 0 else f'FAIL({rc})'}")

if __name__ == "__main__":
    main()