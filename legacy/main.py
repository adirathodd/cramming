import torch
from models import *
from training import *
from utils import *
from analysis.rnn import ablation as rnn_ablation
from analysis.rnn import fourier as rnn_fourier
from analysis.rnn import svd as rnn_svd
from analysis.rnn import trig as rnn_trig
import matplotlib.pyplot as plt
import pandas as pd
import yaml, os, argparse

# argument parser
parser = argparse.ArgumentParser(description='Script to Train an RNN on Modular Operation (+, -, *, /) and perform Fourier Spectrum Analysis')
parser.add_argument('-p', '--path', default='.', type=str, help='Folder containing yaml file.')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-t', '--train', action='store_true', default=False)
parser.add_argument('-a', '--analysis', action='store_true', default=False)
parser.add_argument('-o', '--operation', default='addition', nargs="+",
                   choices=['addition', 'subtraction', 'multiplication', 'division'],
                   help='Type of modular arithmetic operation')
args = parser.parse_args()
args.operation.sort()

if args.path == ".":
    if len(args.operation) > 1:
        args.path = f"""rnn_modular_{"_".join(args.operation)}"""
    else:
        args.path = f"""rnn_modular_{args.operation[0]}"""

config_path = os.path.join(args.path, 'config.yaml')
if os.path.exists(config_path):
    with open(os.path.join(args.path, 'config.yaml'), 'r') as f:
        config_params = yaml.safe_load(f)
        config_params['training']['save_dir'] = args.path
else:
    print("No filepath given or config filepath does not exist, using default config.")
    config_params = {
        "data": create_default_data_config(args.operation),
        "model": create_default_model_config(args.operation),
        "training": create_default_opt_config(args.path)
    }

    save_dir = config_params['training']['save_dir']
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'config.yaml'),'w') as f:
        yaml.dump(config_params, f, default_flow_style=False, sort_keys=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create dataset
dataset = create_dataset(args.operation, config_params["data"], device)

# Update model config with actual dataset vocab size
config_params["model"]["vocab_size"] = dataset.vocab_size

# set up the rnn model
model = create_model(config_params['model'], device=device)

print("Created Modular " + " ".join(args.operation) + " Dataset and Model:\n")
if args.verbose:
    print_dataset_info(dataset)
    print(model)

if args.train:
    checkpoint = train(model, dataset, config_params['training'])
    if not os.path.exists(os.path.join(args.path, 'figures')):
        os.makedirs(os.path.join(args.path, 'figures'))

    df = pd.DataFrame({
        'Train Loss': checkpoint['train_losses'],
        'Test Loss': checkpoint['test_losses']
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(y=['Train Loss', 'Test Loss'], ax=ax)
    ax.set_title('Training and Test Loss Over Time')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    plt.savefig(os.path.join(args.path, 'figures', 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
if args.analysis:
    model, checkpoint = load_model(os.path.join(args.path, 'checkpoints/final.pt'), config_params['model'], device='cpu')
    dataset = dataset.to_device('cpu')

    if not os.path.exists(os.path.join(args.path, 'figures')):
        os.makedirs(os.path.join(args.path, 'figures'))
    
    print("=== Fourier Spectrum Analysis Report ===\n")
    # Basic model evaluation - in utils.py
    print("1. Model performance:")
    print(f"Accuracy on entire dataset: {evaluate_model(model, dataset.dataset, dataset.labels)}% \n\t Train set accuracy: {checkpoint['final_train_accuracy']}% \n\t Test set accuracy: {checkpoint['final_test_accuracy']}%")
    
    # For each operation in args.operation
    for op in args.operation:
        # Fourier coefficient analysis - in fourier_spectrum_analysis.py
        print(f"=== {op.upper()} ===")
        ip_elbow = rnn_fourier.compute_ip_elbow(checkpoint, dataset, op)
        print("2. Generating fourier coefficient analysis plots")
        rnn_fourier.fourier_spectrum_analysis_plotting(model, checkpoint, dataset, args.path, op)
        print()
    
        # SVD analysis - in fourier_spectrum_analysis.py
        print("3. Generating Weight SVD spectrum plots")
        rnn_svd.svd_spectrum_analysis_plotting(model, checkpoint, dataset, args.path, op, ip_elbow)
        print()
        
        # SVD ablation
        print("4. Performing SVD ablation analysis")
        rnn_ablation.svd_ablation_analysis(config_params['model'], checkpoint, dataset, op, ip_elbow)
        print()
        
        # Fourier component ablation
        print("5. Performing fourier component ablation analysis")
        rnn_ablation.fourier_ablation_analysis(config_params['model'], checkpoint, dataset, op, ip_elbow)
        print()
        # test others first.
        
        # Trigonometric identity verification
        if op == 'addition':
            print("6. Trigonometric Identity Verification:")
            rnn_trig.verify_trigonometric_identity(model, checkpoint, dataset)
            print()
        
    print("=== Analysis Complete ===")
