import time
import tqdm
import torch
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

        self.train_epoch_times = []
        self.eval_epoch_times = []
        self.test_time = None

    def _sync(self):
        if self.cuda_condition:
            torch.cuda.synchronize()

    def train(self, epoch):
        if hasattr(self.model, "on_epoch_start"):
            self.model.on_epoch_start(epoch, self.train_dataloader, self.device)
        self._sync()
        t0 = time.perf_counter()
        self.iteration(epoch, self.train_dataloader, train=True)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.train_epoch_times.append(elapsed)
        self.logger.info(f"TIMING train_epoch {epoch} {elapsed:.4f}s")

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        self._sync()
        t0 = time.perf_counter()
        result = self.iteration(epoch, self.eval_dataloader, train=False)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.eval_epoch_times.append(elapsed)
        self.logger.info(f"TIMING valid_epoch {epoch} {elapsed:.4f}s")
        return result

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        self._sync()
        t0 = time.perf_counter()
        result = self.iteration(epoch, self.test_dataloader, train=False)
        self._sync()
        elapsed = time.perf_counter() - t0
        self.test_time = elapsed
        self.logger.info(f"TIMING test {elapsed:.4f}s")
        return result

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            if 'beta' in key:
                # print(key)
                # new_key = key.replace('beta', 'sqrt_beta')
                # original_state_dict[new_key] = new_dict[key]
                original_state_dict[key]=new_dict[key]
            else:
                original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                user_ids, input_ids, answers, neg_answer, same_target = batch
                loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, _, _ = batch
                recommend_output = self.model.predict(input_ids, user_ids)
                recommend_output = recommend_output[:, -1, :]# 推荐的结果
                
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                try:
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                except: # bert4rec
                    rating_pred = rating_pred[:, :-1]
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # PROPOSED: by default mask augmented/context items in the (expanded) input
                # so they are never recommended (prevents trivially "predicting" an inserted
                # item). With --aug_no_mask, leave inserted items recommendable (retrieval-aug).
                if getattr(self.args, "seq_augment", False) and not getattr(self.args, "aug_no_mask", False):
                    inp = input_ids.cpu().numpy()
                    rating_pred[np.arange(rating_pred.shape[0])[:, None], inp] = -1e9

                # --- rank dump (opt-in; uses the SAME masked rating_pred as HR) ---
                if getattr(self.args, "dump_ranks", False):
                    ans = answers.cpu().data.numpy()
                    true_score = rating_pred[np.arange(len(rating_pred)), ans]
                    rank = (rating_pred > true_score[:, None]).sum(1) + 1   # competition rank
                    if not hasattr(self, "_rank_dump"):
                        self._rank_dump = []
                    for ui, rk in zip(batch_user_index, rank):
                        self._rank_dump.append((int(ui), int(rk)))

                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # Take the corresponding values from the corresponding dimension 
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            if getattr(self.args, "dump_ranks", False) and getattr(self, "_rank_dump", None):
                import csv, os
                path = os.path.join(self.args.output_dir, f"{self.args.train_name}_ranks.csv")
                with open(path, "w", newline="") as fp:
                    w = csv.writer(fp); w.writerow(["user_index", "rank"])
                    w.writerows(self._rank_dump)
                self.logger.info(f"dumped {len(self._rank_dump)} ranks -> {path}")
                self._rank_dump = []

            return self.get_full_sort_score(epoch, answer_list, pred_list)
