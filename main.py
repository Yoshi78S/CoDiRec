import os
import time
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

def main():

    args = parse_args()
    check_path(args.output_dir)            # ensure output/ exists before logging (standalone)
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    n_params = sum(p.nelement() for p in model.parameters())
    logger.info(f"TIMING n_params {n_params}")

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)

            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
            # torch.save(trainer.model.state_dict(), args.checkpoint_path)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        wall_start = time.perf_counter()
        completed_epochs = 0
        for epoch in range(args.epochs):

            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            completed_epochs = epoch + 1
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model, epoch=epoch)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        wall_elapsed = time.perf_counter() - wall_start
        logger.info(f"TIMING wall_train {wall_elapsed:.4f}s")
        logger.info(f"TIMING epochs_run {completed_epochs}")

        train_times = trainer.train_epoch_times
        if train_times:
            logger.info(f"TIMING train_epoch_mean {np.mean(train_times):.4f}s")
            logger.info(f"TIMING train_epoch_sum {np.sum(train_times):.4f}s")
        eval_times = trainer.eval_epoch_times
        if eval_times:
            logger.info(f"TIMING valid_epoch_mean {np.mean(eval_times):.4f}s")

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        best_epoch = early_stopping.best_epoch if early_stopping.best_epoch is not None else 0
        logger.info(f"Loaded best-val checkpoint from epoch {best_epoch}")
        scores, result_info = trainer.test(best_epoch)

    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == "__main__":
    main()
