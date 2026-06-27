import os
import random
import torch
import datetime
import argparse
import numpy as np
import logging

def set_logger(log_path, log_name='seqrec', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def parse_args():

    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="../BSARec/src/data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--res_scale", default=1.0, type=float,
                        help="residual-dominance study: global scale for ALL +x residuals "
                             "(freq, attn, FFN). 1.0 = original BSARec.")
    parser.add_argument("--res_scale_freq", default=None, type=float,
                        help="per-path override: freq-branch residual scale (else res_scale).")
    parser.add_argument("--res_scale_attn", default=None, type=float,
                        help="per-path override: attention residual scale (else res_scale).")
    parser.add_argument("--res_scale_ffn", default=None, type=float,
                        help="per-path override: FFN residual scale (else res_scale).")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)
    parser.add_argument("--dump_ranks", action="store_true",
                        help="dump per-instance true-item rank to output/<train_name>_ranks.csv (eval only)")
    parser.add_argument("--full_softmax", action="store_true",
                        help="SASRec: use full-vocab softmax CE instead of BCE+1neg (loss-match study)")
    parser.add_argument("--seq_augment", action="store_true",
                        help="PROPOSED: expand input sequences with co-occurrence+direction associates")
    parser.add_argument("--aug_k", default=3, type=int,
                        help="number of associate items to insert per sequence (--seq_augment)")
    parser.add_argument("--aug_no_mask", action="store_true",
                        help="do NOT mask inserted associate items at eval (retrieval-augmented; "
                             "inserted items incl. a surfaced target remain recommendable)")
    parser.add_argument("--aug_anchor", default="random", choices=["random", "last"],
                        help="seq_augment anchor: 'random' item each insertion, or 'last' "
                             "(append the LAST item's top-k associates after it)")

    # model args
    parser.add_argument("--model_type", default='BSARec', type=str)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int, help="embedding dimension")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of blocks")
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)

    args, _ = parser.parse_known_args()

    if args.model_type.lower() == 'bsarec':
        parser.add_argument("--c", default=3, type=int)
        parser.add_argument("--alpha", default=0.9, type=float)

    elif args.model_type.lower() == 'bert4rec':
        parser.add_argument("--mask_ratio", default=0.2, type=float)

    elif args.model_type.lower() == 'caser':
        parser.add_argument("--nh", default=8, type=int)
        parser.add_argument("--nv", default=4, type=int)
        parser.add_argument("--reg_weight", default=1e-4, type=float)

    elif args.model_type.lower() == 'duorec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)

    elif args.model_type.lower() == 'fearec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default='us_x', type=str)
        parser.add_argument("--fredom", default='True', type=str) # use eval function to use as boolean

    elif args.model_type.lower() == 'gru4rec':
        parser.add_argument("--gru_hidden_size", default=64, type=int, help="hidden size of GRU")

    elif args.model_type.lower() == 'mamba4rec':
        parser.add_argument("--d_state", default=16, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")

    elif args.model_type.lower() == 'sigma':
        parser.add_argument("--d_state", default=32, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")

    elif args.model_type.lower() == 'matrrec':
        parser.add_argument("--d_state", default=32, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")

    elif args.model_type.lower() == 'hlsdmrec':
        parser.add_argument("--d_state", default=16, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")
        parser.add_argument("--hlsd_q", default=2, type=int, help="short-term window length")

    elif args.model_type.lower() == 'echomamba4rec':
        parser.add_argument("--d_state", default=32, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")

    elif args.model_type.lower() == 'echomambasa':
        parser.add_argument("--d_state", default=32, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")
        parser.add_argument("--echo_sa_alpha", default=-1.0, type=float,
                            help="fixed mamba-branch weight in [0,1]; <0 = learnable gate")
        parser.add_argument("--echo_sa_gate", default="scalar", type=str,
                            choices=["scalar", "seq", "pos", "channel"],
                            help="fusion gate: scalar(static) | seq | pos | channel (input-dependent)")
        parser.add_argument("--echo_sa_r", default=0, type=int,
                            help="two-path attention: 0=single; r>0 reverse last r items for 2nd path; -1=full reverse")

    elif args.model_type.lower() in ('echocodiremb', 'echocodirscore'):
        parser.add_argument("--d_state", default=16, type=int, help="Mamba state dimension")
        parser.add_argument("--d_conv", default=4, type=int, help="Mamba conv kernel size")
        parser.add_argument("--expand", default=2, type=int, help="Mamba expansion factor")
        parser.add_argument("--codir_window", default=3, type=int, help="co-occurrence window length")
        parser.add_argument("--codir_cap", default=50, type=int, help="seq cap for matrix building (0=full)")
        parser.add_argument("--codir_rank", default=64, type=int, help="SVD rank for co/dir embeddings (Model A)")
        parser.add_argument("--codir_norm", default="l2", type=str, choices=["l2", "ppmi", "raw"],
                            help="co-occurrence C normalization: l2(row L2) | ppmi(popularity-corrected) | raw")
        parser.add_argument("--codir_dir_norm", default="ratio_l2", type=str, choices=["ratio", "ratio_l2"],
                            help="directional D normalization: ratio((F-F^T)/(F+F^T)) | ratio_l2(+row L2)")
        parser.add_argument("--echo_attn_raw", action="store_true",
                            help="feed the attention branch the RAW (pre-FFT) input x instead of the "
                                 "frequency-filtered xf, i.e. (Freq->Mamba) in series PARALLEL attention(x). "
                                 "Default off = both branches share xf (current structure).")

    elif args.model_type.lower() == 'icsrec':
        parser.add_argument("--temperature", default=1.0, type=float)
        parser.add_argument("--sim", default="dot", type=str, help="dot or cos")
        parser.add_argument("--intent_num", default=512, type=int)
        parser.add_argument("--cl_mode", default="cf", type=str, help="c|f|cf")
        parser.add_argument("--f_neg", action="store_true", help="False-negative masking")
        parser.add_argument("--rec_weight", default=1.0, type=float)
        parser.add_argument("--lambda_0", default=0.1, type=float, help="CICL weight")
        parser.add_argument("--beta_0", default=0.1, type=float, help="FICL weight")

    elif args.model_type.lower() == 'iclrec':
        parser.add_argument("--temperature", default=1.0, type=float)
        parser.add_argument("--sim", default="dot", type=str)
        parser.add_argument("--num_intent_clusters", default="256", type=str,
                            help="comma-separated cluster counts, e.g. '256' or '64,128,256'")
        parser.add_argument("--contrast_type", default="Hybrid", type=str,
                            help="InstanceCL | IntentCL | Hybrid")
        parser.add_argument("--seq_representation_type", default="mean", type=str,
                            help="mean | concatenate")
        parser.add_argument("--warm_up_epoches", default=0, type=int)
        parser.add_argument("--de_noise", action="store_true")
        parser.add_argument("--rec_weight", default=1.0, type=float)
        parser.add_argument("--cf_weight", default=0.1, type=float,
                            help="InstanceCL weight")
        parser.add_argument("--intent_cf_weight", default=0.3, type=float,
                            help="IntentCL (PCL) weight")

    return parser.parse_args()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model, epoch=None):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.best_epoch = epoch
            self.save_checkpoint(score, model, epoch)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, epoch)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            tag = f' (epoch {epoch})' if epoch is not None else ''
            self.logger.info(f'Validation score increased{tag}.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
