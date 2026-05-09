import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from sonar.data.asvspoof import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_ASVspoof2019_val, Dataset_in_the_wild_eval
from sonar.guided_model import *
from tqdm import tqdm
import torch.nn.functional as F
import plotly.graph_objects as go
from scipy.fft import fft
from scipy.signal import convolve
from sonar.orthogonal_loss import *
from sonar import eval_metric_LA as em
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sonar.utils import setup_seed, L2_regularization
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.functional import cosine_similarity
import wandb
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.multiprocessing as mp
import warnings
import copy
from datetime import timedelta
from sklearn.metrics import precision_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import sys
import logging
import warnings
import os
from sonar.srm_filters import *
from sonar.entropy_loss import *
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import plotly.graph_objects as go
from scipy.stats import lognorm
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from simple_classifier import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch
from sklearn.decomposition import PCA

logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
__author__ = "Ido Nitzan Hidekel"
_email__ = "idonithid@gmail.com"

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

def compute_cosine_similarity_histograms(dataloader, model, device, bins=50, output_file="cosine_similarity_histogram.png"):
    """
    Computes cosine similarities between z_low and z_high embeddings across a dataset
    and generates a histogram for fake and real samples.

    Parameters:
    - dataloader: DataLoader, data loader for the dataset
    - model: PyTorch model, outputs z_low and z_high embeddings
    - device: torch.device, device to perform computations on
    - bins: int, number of bins for the histogram
    - output_file: str, path to save the histogram image

    Outputs:
    - Saves a histogram showing cosine similarities for fake and real samples.
    """
    fake_cosine_similarities = []
    real_cosine_similarities = []

    # Iterate through the dataloader
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            inputs, labels = batch  # Assuming inputs and labels are returned
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.reshape(inputs.shape[0],1,64600)
            # Get z_low and z_high from the model
            z_low, z_high = model(inputs)  # Both should be of shape (B, 201, 1024)

            # Flatten across time steps
            z_low_flat = z_low.view(-1, 1024)  # Shape: (B * 201, 1024)
            z_high_flat = z_high.view(-1, 1024)  # Shape: (B * 201, 1024)
            labels_flat = labels.unsqueeze(1).repeat(1, z_low.shape[1]).view(-1).cpu().numpy()  # Shape: (B * 201,)

            # Compute cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(z_low_flat, z_high_flat, dim=1).cpu().numpy()

            # Separate by labels
            fake_cosine_similarities.extend(cosine_sim[labels_flat == 0].tolist())
            real_cosine_similarities.extend(cosine_sim[labels_flat == 1].tolist())

 # Generate density plot using Plotly
    fig = go.Figure()

    # Add density for fake samples
    fig.add_trace(go.Histogram(
        x=fake_cosine_similarities,
        nbinsx=bins,
        histnorm='probability density',  # Normalize to density
        opacity=0.7,
        name="Fake (Label 0)"
    ))

    # Add density for real samples
    fig.add_trace(go.Histogram(
        x=real_cosine_similarities,
        nbinsx=bins,
        histnorm='probability density',  # Normalize to density
        opacity=0.7,
        name="Real (Label 1)"
    ))

    # Update layout
    fig.update_layout(
        title="Cosine Similarity Density Function of the embeddings (Z_low and Z_high)",
        xaxis_title="Cosine Similarity",
        yaxis_title="Density",
        barmode="overlay",  # Overlap the bars for comparison
        legend=dict(x=0.5, y=1),
    )
    fig.update_traces(opacity=0.75)

    # Save the plot
    fig.write_image(output_file)

    return {
        "fake_cosine_similarities": fake_cosine_similarities,
        "real_cosine_similarities": real_cosine_similarities,
        "density_plot_file": output_file
    }



def js_divergence_loss(E_HFE, E_LFE, labels):
    # Normalize embeddings
    P = F.softmax(E_HFE, dim=2)  # Shape: B x 201 x 1024
    Q = F.softmax(E_LFE, dim=2)  # Shape: B x 201 x 1024
    
    # Compute midpoint
    M = 0.5 * (P + Q)  # Shape: B x 201 x 1024

    # Compute KL divergences
    KL_P_M = (P * (torch.log(P + 1e-6) - torch.log(M + 1e-6))).sum(dim=2)  # Shape: B x 201
    KL_Q_M = (Q * (torch.log(Q + 1e-6) - torch.log(M + 1e-6))).sum(dim=2)  # Shape: B x 201

    # Jensen-Shannon divergence
    JS_div = 0.5 * KL_P_M + 0.5 * KL_Q_M  # Shape: B x 201

    # Aggregate across the 201 vectors
    JS_div_batch = JS_div.mean(dim=1)  # Shape: B

    # Define loss based on labels
    labels = labels.float()
    loss = labels * JS_div_batch + (1 - labels) * (1 - JS_div_batch)
    return loss.mean()  # Average across the batch




def val_model_optimized(dev_loader, model, criterion, rank, main_rank, world_size=8):
    val_loss = 0.0
    val_kl_loss = 0.0
    val_ce_loss = 0.0
    num_total = 0.0
    correct = 0
    device = torch.device(f'cuda:{rank}')
    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []
    all_probs = []
    guided_loss = js_divergence_loss
    model.eval()
    with torch.no_grad():
        for batch_x, labels in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            labels = labels.view(-1).type(torch.int64).to(device)

            # Reshape if needed
            batch_x = batch_x.reshape(batch_size, 1, 64600)
        
            batch_out ,batch_x_low,batch_x_high= model(batch_x)
            
            task_loss = criterion(batch_out, labels)
            kl_loss = guided_loss(batch_x_low,batch_x_high,labels)
            batch_loss = task_loss + 0.8*kl_loss

            val_loss += (batch_loss.item() * batch_size)
            val_kl_loss += (kl_loss.item() * batch_size)
            val_ce_loss+= (task_loss.item() * batch_size)

            # Move outputs to CPU for metric calculation
            probs = F.softmax(batch_out.detach().cpu(), dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].numpy())


    # Convert results to tensors for reduction
    val_loss_tensor = torch.tensor([val_loss], device=device)
    val_task_loss_tensor = torch.tensor([val_ce_loss], device=device)
    val_kl_loss_tensor = torch.tensor([val_kl_loss], device=device)

    num_total_tensor = torch.tensor([num_total], device=device)

    # Reduce across all GPUs
    dist.reduce(val_loss_tensor, dst=main_rank, op=dist.ReduceOp.SUM)
    dist.reduce(val_task_loss_tensor, dst=main_rank, op=dist.ReduceOp.SUM)
    dist.reduce(val_kl_loss_tensor, dst=main_rank, op=dist.ReduceOp.SUM)

    dist.reduce(num_total_tensor, dst=main_rank, op=dist.ReduceOp.SUM)

    # Gather predictions and labels on rank 0
    all_labels_tensor = torch.tensor(all_labels, device=device)
    all_probs_tensor = torch.tensor(all_probs, device=device)

    if rank == main_rank:
        gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(world_size)]
        gathered_probs = [torch.zeros_like(all_probs_tensor) for _ in range(world_size)]
    else:
        gathered_labels = None
        gathered_probs = None
    
    dist.barrier()

    # Gather on rank 0
    dist.gather(all_labels_tensor, gather_list=gathered_labels, dst=main_rank)
    dist.gather(all_probs_tensor, gather_list=gathered_probs, dst=main_rank)

    if rank == main_rank:
        # Concatenate results
        all_labels = np.concatenate([g.cpu().numpy() for g in gathered_labels])
        all_probs = np.concatenate([g.cpu().numpy() for g in gathered_probs])

        val_loss = val_loss_tensor.item() / num_total_tensor.item()

        val_ce_loss = val_task_loss_tensor.item() / num_total_tensor.item()
        val_kl_loss = val_kl_loss_tensor.item() / num_total_tensor.item()

        sort_indices = np.argsort(all_probs)
        all_probs = all_probs[sort_indices]
        all_labels = all_labels[sort_indices]

                # Compute metrics
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
        print(f"EER: {eer}")
        print(f"thresholds: {thresholds}")
        return val_loss,val_kl_loss,val_ce_loss ,eer
    else:
        return None, None,None,None


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '2222'  # Replace with the chosen IP address
    os.environ['MASTER_PORT'] = '12346'           # Replace with an available port
    timeout = timedelta(minutes=10)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(f"cuda:{rank}")

def cleanup():
    dist.destroy_process_group()



def process_and_plot_distributions(model, dataloader, output_path="mean_distributions_ours"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    real_sum = torch.zeros((1024,), dtype=torch.float32, device=device)
    fake_sum = torch.zeros((1024,), dtype=torch.float32, device=device)
    real_count = 0
    fake_count = 0

    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(64,1,64600)
        with torch.no_grad():
            outputs = model(inputs)
            softmax_outputs = torch.softmax(outputs, dim=-1)

        real_mask = labels == 1
        fake_mask = labels == 0

        if real_mask.any():
            real_sum += softmax_outputs[real_mask].sum(dim=(0, 1))
            real_count += real_mask.sum().item() * 201

        if fake_mask.any():
            fake_sum += softmax_outputs[fake_mask].sum(dim=(0, 1))
            fake_count += fake_mask.sum().item() * 201

    real_mean = real_sum / real_count if real_count > 0 else None
    fake_mean = fake_sum / fake_count if fake_count > 0 else None       # Normalize to ensure probability distribution
    if real_mean is not None:
        real_mean /= real_mean.sum()
    if fake_mean is not None:
        fake_mean /= fake_mean.sum()


    # Convert to NumPy for plotting
    real_mean = real_mean.cpu().numpy() if real_mean is not None else None
    fake_mean = fake_mean.cpu().numpy() if fake_mean is not None else None

    # Create an x-axis for the 1024 dimensions
    x = np.arange(real_mean.shape[0]) if real_mean is not None else np.arange(fake_mean.shape[0])

    # Create Plotly figure
    fig = go.Figure()

    if real_mean is not None:
        fig.add_trace(go.Scatter(
            x=x,
            y=real_mean,
            mode='lines',
            name='Real Mean Distribution',
            line=dict(color='blue')
        ))

    if fake_mean is not None:
        fig.add_trace(go.Scatter(
            x=x,
            y=fake_mean,
            mode='lines',
            name='Fake Mean Distribution',
            line=dict(color='red')
        ))

    # Update layout
    fig.update_layout(
        title="Mean Distributions of Real and Fake Samples",
        xaxis_title="Feature Index",
        yaxis_title="Mean Value",
        legend_title="Distributions",
        template="plotly_white"
    )

    # Save plots
    html_path = f"{output_path}.html"
    png_path = f"{output_path}.png"
    fig.write_html(html_path)
    fig.write_image(png_path)  # Requires 'kaleido'

    # Show the plot
    fig.show()

    print(f"Plots saved as {html_path} and {png_path}")



def train_epoch_optimized(train_loader, model, optimizer, rank, criterion, sampler, epoch, main_rank):
    device = torch.device(f"cuda:{rank}")
    running_loss = 0
    num_total = 0.0
    running_task_loss = 0
    running_kl_loss = 0
    sampler.set_epoch(epoch)
    model.train()
    model.to(device)
    num = 1
    guided_loss = js_divergence_loss
    for batch_x, labels in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        labels = labels.view(-1).type(torch.int64).to(device)
        optimizer.zero_grad()
        
    # Forward pass with autocast for mixed precision
        batch_x = batch_x.reshape(batch_size, 1, 64600)
        batch_out ,batch_x_low,batch_x_high= model(batch_x)
        
        task_loss = criterion(batch_out, labels)
        kl_loss = guided_loss(batch_x_low,batch_x_high,labels)
        
        #spectral_loss = model.module.polynomial.spectrum_loss()
        
        batch_loss = task_loss + 0.8*kl_loss


        batch_loss.backward()
        optimizer.step()

        # If your model has constraints to apply after the step
        #model.module.srm_module.apply_constraints()


        running_loss += (batch_loss.item() * batch_size)

        running_task_loss += (task_loss.item() * batch_size)

        running_kl_loss += (kl_loss.item() * batch_size)

        num += 1

    running_loss_tensor = torch.tensor([running_loss], device=device)
    running_kl_loss_tensor = torch.tensor([running_kl_loss], device=device)
    running_task_loss_tensor = torch.tensor([running_task_loss], device=device)

    num_total_tensor = torch.tensor([num_total], device=device)


    # Reduce the loss across GPUs
    torch.distributed.reduce(running_loss_tensor, dst=main_rank, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(running_kl_loss_tensor, dst=main_rank, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce(running_task_loss_tensor, dst=main_rank, op=torch.distributed.ReduceOp.SUM)
    
    torch.distributed.reduce(num_total_tensor, dst=main_rank, op=torch.distributed.ReduceOp.SUM)

    if rank == main_rank:
        running_loss = running_loss_tensor.item() / num_total_tensor.item()
        running_kl_loss = running_kl_loss_tensor.item() / num_total_tensor.item()
        running_task_loss = running_task_loss_tensor.item() / num_total_tensor.item()

    return (running_loss,running_kl_loss,running_task_loss) if rank == main_rank else (None,None,None)


def get_args():
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--train_data_path', type=str, default=os.environ.get('ASVSPOOF2019_ROOT','data/ASVspoof2019/LA') + '/ASVspoof2019_LA_train/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''
    parser.add_argument('--eval_LA_database_path', type=str, default=os.environ.get('ASVSPOOF2021_ROOT','data/ASVspoof2021') + '/LA/ASVspoof2021_LA_eval/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    '''
    parser.add_argument('--eval_DF_database_path', type=str, default=os.environ.get('ASVSPOOF2021_ROOT','data/ASVspoof2021') + '/DF/ASVspoof2021_DF_eval/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_DF_eval/flac
    '''
    parser.add_argument('--train_protocols_path', type=str, default='data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', help='Change with path to user\'s LA database protocols directory address')
    '''
    % pפrotocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''
    parser.add_argument('--eval_LA_protocols_path', type=str, default="data/ASVspoof2021/LA/ASVspoof2021_LA_eval/ASVspoof2021_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt", help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt

    '''
    parser.add_argument('--eval_DF_protocols_path', type=str, default="data/ASVspoof2021/DF/ASVspoof2021_DF_eval/ASVspoof2021_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt", help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt

    '''
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--wandb_title',type=str, default='SRM-WAV2VEC')
    # model
    parser.add_argument('--seed', type=int, default=12345, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default="checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/KL_Loss_lambda_1_seed_12345/best_run_epoch_14_run_loss_0.04574163804143112_ITW_eer_0.060991379310344825.pth", help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default="checkpoints/Twin-Wav2Vec-SRM-Scores/scores_asv2021/eval_df.txt",
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 

    parser.add_argument('--model_save_path', type=str,default="checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/", help='Checkpoint save path')

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=4, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')


    ##===================================================new loss==================================================================================
    parser.add_argument('--loss', type=str, default="ce",
                        choices=['ce','oc', 'toc'], help="loss function")

    ##===================================================guided Mode ==================================================================================
    parser.add_argument('--guided_mode', type=bool, default=True,
                        choices=[False,True], help="Setting mode for guided model")
    
    args = parser.parse_args()

    # setting One class or two class
    if args.loss != 'ce':
        args.binary_class = False
    else:
        args.binary_class = True
    return args

def create_dataloaders(train_protocols_path, train_data_path, val_protocol_path, val_data_path, args, world_size, local_rank):
    # Define train dataloader
    d_label_trn, file_train = genSpoof_list(dir_meta = train_protocols_path, is_train=True, is_eval=False)
    print('no. of training trials', len(file_train))

    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=train_data_path, algo=args.algo)
    sampler_train = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank,seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=28, num_workers=10, shuffle=False, drop_last=True,pin_memory=True ,sampler=sampler_train)

    # Define validation dataloader
    d_label_dev, file_dev = genSpoof_list(dir_meta = val_protocol_path, is_train=False, is_eval=False)
    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_ASVspoof2019_val(args, list_IDs=file_dev, labels=d_label_dev, base_dir=val_data_path, algo=args.algo)
    dev_sampler = DistributedSampler(dev_set, num_replicas=world_size, rank=local_rank, shuffle=False)
    dev_loader = DataLoader(dev_set, batch_size=28, num_workers=10, shuffle=False, drop_last=True, sampler=dev_sampler)


    # create dataloader 
    file_eval = genSpoof_list('data/IDW/meta.csv',is_train=False,is_eval=True)
    indw_set=Dataset_in_the_wild_eval(list_IDs = file_eval[1:],base_dir = 'data/IDW/wav')


    in_the_wild_loader = DataLoader(indw_set, batch_size=32, num_workers=10, shuffle=False, drop_last=True, sampler=dev_sampler)
    
    return train_loader, dev_loader,sampler_train,in_the_wild_loader



##################################################### MAIN RUN #####################################################

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    set_random_seed(args.seed,rank,args=args)
    main_rank = 0
    log_wandb = True
    test_loaded_model = False

    if rank ==main_rank and log_wandb:
        wandb.login(key="Insert_your_key")
        wandb.init(project="SRM-audio-df-detection", name=args.wandb_title,settings=wandb.Settings(console="off"))
        wandb.config.update(args)  # Log hyperparameters

    device = torch.device(f"cuda:{rank}")
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)

    print(f"Rank {rank} random check: {torch.randint(0, 100, (1,))}")

    # Initialize classifier model on DataParallel if multiple GPUs are available
    model = GuidedModel(args,device)
    model = model.to(device)

    # testing the model forward with batch
    x = torch.randn(8,1,64600).to(device)
    model(x)
    del x

    #evaluation 
    if args.is_eval:
        #args.model_path = 'checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/ablations/ablations_without_srm_with_js/best_run_epoch_6_run_loss_0.13832336331006162_ITW_eer_0.08728448275862069.pth'
        
        #args.model_path = 'checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/KL_Loss_lambda_1_seed_123456/best_run_epoch_15_run_loss_0.022803038543795947_ITW_eer_0.056896551724137934.pth'
        args.model_path = 'checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/lambda_0.8_exp/best_run_epoch_8_run_loss_0.04241523387269204_ITW_eer_0.06325431034482759.pth'
        
        name = "DF_best"
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path,map_location=device),strict=False)
            print('Model loaded : {}'.format(args.model_path))
        
        eval_protocols_path = getattr(args, f"eval_{args.track}_protocols_path")
        database_path = getattr(args, f"eval_{args.track}_database_path")
        eval_output_scores_path = f"eval_{args.track}_score_{name}_.txt"

        keys_path = f"data/ASVspoof2021/{args.track}/{args.track}-keys-stage-1/keys/CM/trial_metadata.txt"
        
        file_eval = genSpoof_list(dir_meta = eval_protocols_path,is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        
        eval_set= Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = database_path)
        eval_loader = DataLoader(eval_set, batch_size=16, num_workers=10, shuffle=False, drop_last=False)
        
        produce_evaluation_file(eval_loader, model, device, eval_output_scores_path)
        
        sys.exit(0)
   
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    for parameter in ddp_model.module.ssl_model.model.parameters():
        parameter.requires_grad = True
    for parameter in ddp_model.module.noise_ssl_model.model.parameters():
        parameter.requires_grad = True
    

    optimizer = torch.optim.AdamW(ddp_model.module.parameters(), lr=1e-5, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)  # T_max = total epochs

    print("Criterion: WCE")
    # define train dataloader and validation dataloader
    val_protocol_path = 'data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'    
    val_data_path ='data/ASVspoof2019/LA/ASVspoof2019_LA_dev/'
    train_loader, dev_loader,sampler_train,in_the_wild_loader = create_dataloaders(args.train_protocols_path, args.train_data_path, val_protocol_path, val_data_path, args, world_size, rank)


    if test_loaded_model:
        test_path = 'checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/simple_cls/best_run_epoch_13_run_loss_0.006360377584184919_ITW_eer_0.06928879310344828.pth'
        state_dict = torch.load(test_path,map_location=f'cuda:{rank}')
        model.load_state_dict(state_dict,strict=False)
        model.eval()
        del state_dict

        sys.exit()
    
    print("Train!")
    best_val_eer = float('inf')
    epochs_without_improvement = 0
    early_stopping_threshold = 3  # Number of epochs without improvement for early stopping
    for epoch in range(0,args.num_epochs):
            with tqdm(total=2, desc=f'Epoch {epoch + 1}') as pbar:
                print("Train!")
                total_loss_train,kl_loss_train,ce_loss_train = train_epoch_optimized(train_loader, ddp_model, optimizer,rank,criterion,sampler_train,epoch,main_rank)
                print("Validate")
                val_loss,val_kl_loss,val_ce_loss ,val_eer = val_model_optimized(dev_loader, ddp_model, criterion, rank,main_rank,world_size)
                print("IDW Validate")
                itw_loss,itw_kl_loss,itw_ce_loss ,itw_eer = val_model_optimized(in_the_wild_loader, ddp_model, criterion, rank,main_rank,world_size)
                #itw_val_loss,itw_eer,eer,val_loss = 0,0,0,0

                if rank == main_rank:
                    metric_dict = {
                        "Total-loss-train": total_loss_train,
                        "CE-loss-train": ce_loss_train,
                        "KL-loss-train": kl_loss_train,

                        "Total-loss-val": val_loss,
                        "CE_loss_val": val_ce_loss,
                        "KL-loss_val": val_kl_loss,
                        "EER-val":val_eer,

                        "Total-loss-itw": itw_loss,
                        "CE_loss_itw": itw_ce_loss,
                        "KL-loss_itw": itw_kl_loss,
                        "EER-itw":itw_eer

                    }
                    scheduler.step()
                    wandb.log(metric_dict)
                              # Early stopping logic
                    if val_eer < best_val_eer:
                        best_val_eer = val_eer
                        epochs_without_improvement = 0
                        if rank==0:
                            best_dict = ddp_model.module.state_dict()
                            torch.save(best_dict,os.path.join(args.model_save_path, 'best_run_epoch_{}_run_loss_{}_ITW_eer_{}.pth'.format(epoch,ce_loss_train,itw_eer)))
                            

                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_threshold:
                        print("Early stopping triggered. Saving the best model from early stopping.")
                        break
    cleanup()

def spawn_workers(world_size, args):
    mp.spawn(main_worker,
             args=(world_size,args),
             nprocs=world_size,
             join=True)


def main(world_size,args):
    spawn_workers(world_size, args)


def set_random_seed(random_seed, rank=None, args=None):
    """
    Set the random seed for numpy, python, and cudnn, with rank awareness for DDP.

    Args:
        random_seed (int): The base random seed.
        rank (int, optional): The rank of the current process in DDP. Defaults to None.
        args (argparse.Namespace, optional): Argument parser with optional cudnn settings.
    """
    # Adjust the seed for each rank to ensure unique but reproducible seeds per rank

    seed = random_seed

    # Set seeds for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For PyTorch's CUDA backend
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Configure cuDNN for deterministic behavior
        if args is None:
            cudnn_deterministic = True
            cudnn_benchmark = False
        else:
            cudnn_deterministic = args.cudnn_deterministic_toggle
            cudnn_benchmark = args.cudnn_benchmark_toggle

        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f"Random seed {seed} set for rank {rank}")


if __name__ == '__main__':
    # get args
    args = get_args()
    print(args)
    data_dir = 'checkpoints/ssl/trained_models/checkpoints/ido_checkpoint/'
    args.model_save_path = os.path.join(data_dir, args.model_save_path)
    args.seed = 123456
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
        print(f"Directory '{args.model_save_path}' created.")
    else:
        print(f"Directory '{args.model_save_path}' already exists.")    # Training and validation 
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(args.model_save_path)
    set_random_seed(args.seed)
    world_size = torch.cuda.device_count()
    print(f"Visible GPUs: {world_size}")
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    main(world_size,args)