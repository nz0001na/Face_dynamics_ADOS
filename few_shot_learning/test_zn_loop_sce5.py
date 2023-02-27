import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import csv
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from apl import models
from apl import memory_store
from datasets import omniglot

os.environ["CUDA_VISIBLE_DEVICES"]="3"

check_list = [ # 'RSAFF_3_2021-12-02-23-43_fusion'
    'RSAFF_3_2021-12-02-23-43_sce5'
    # 'RSAFF_3_2021-12-04-02-31_top3'
            ]

# IN = 5
N_CLASSES = 3
N_PER_CLASS = 8   # 154  106  193  150 30 100

N_NEIGHBOURS = 3
MAX_BATCHES = 3000
MEMORY_SIZE = 10000
SIGMA_RATIO = 0.75
QUERY_EMBED_DIM = 64
LABEL_EMBED_DIM = 32
KEY_SIZE = 256
VALUE_SIZE = 256
N_HEADS = 2
NUM_LAYERS = 5
USE_CUDA = True
SAVE_FREQUENCY = 100


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.long().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot

def split_batch(batch, nshot, n_classes, n_per_class):
    context = []
    query = []
    for i in range(n_classes):
        class_start = i * n_per_class
        context.extend(
            [batch[b] for b in range(class_start, class_start + nshot)])
        query.extend(
            [batch[b] for b in range(class_start + nshot, class_start + n_per_class)])
    return context, query

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=N_CLASSES)
    parser.add_argument("--n_per_classes", type=int, default=N_PER_CLASS)
    parser.add_argument("--n_neighbours", type=int, default=N_NEIGHBOURS)
    parser.add_argument("--memory_size", type=int, default=MEMORY_SIZE)
    parser.add_argument("--sigma_ratio", type=float, default=SIGMA_RATIO)
    parser.add_argument("--query_embed_dim", type=int, default=QUERY_EMBED_DIM)
    parser.add_argument("--label_embed_dim", type=int, default=LABEL_EMBED_DIM)
    parser.add_argument("--key_size", type=int, default=KEY_SIZE)
    parser.add_argument("--value_size", type=int, default=VALUE_SIZE)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--use_cuda", type=bool, default=USE_CUDA)
    return parser.parse_args()

# main
def test_checkpoint(model_path, dst_path):
    model_list = os.listdir(model_path)
    final_acc1 = []
    final_acc1.append(['model', 'acc', 'acc_kernel', 'acc_top1', 'mem'])
    acc_name1 = dst_path + 'test_acc_online.csv'
    online_acc = dst_path + 'online_acc/'
    online_acc_kernel = dst_path + 'online_acc_kernel/'
    if os.path.exists(online_acc) is False: os.makedirs(online_acc)
    if os.path.exists(online_acc_kernel) is False: os.makedirs(online_acc_kernel)

    # final_acc2 = []
    # final_acc2.append(['model', 'acc', 'acc_kernel', 'acc_top1'])
    # acc_name2 = dst_path + 'test_acc_fixed.csv'
    # fixed_acc = dst_path + 'fixed_acc/'
    # fixed_acc_kernel = dst_path + 'fixed_acc_kernel/'
    # if os.path.exists(fixed_acc) is False: os.makedirs(fixed_acc)
    # if os.path.exists(fixed_acc_kernel) is False: os.makedirs(fixed_acc_kernel)

    args = get_arguments()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    enc = models.Encoder()
    dec = models.RSAFFDecoder(
        args.n_classes, args.query_embed_dim, args.label_embed_dim,
        args.n_neighbours, args.key_size, args.value_size, args.n_heads,
        args.num_layers)
    enc.to(device)
    dec.to(device)
    memory = memory_store.MemoryStore(
        args.memory_size, args.n_classes,
        args.n_neighbours, args.query_embed_dim, device)

    # train_dataset = omniglot.RestrictedOmniglot(
    #     "data/Omniglot", args.n_classes, train=True, noise_std=0.1)
    test_dataset = omniglot.RestrictedOmniglot(
        "data/Omniglot", args.n_classes, train=False, noise_std=0)
    nll_threshold = args.sigma_ratio * np.log(args.n_classes)

    for mo in model_list:
        ckpt = model_path + mo
        checkpoint = torch.load(ckpt)
        enc.load_state_dict(checkpoint['encoder_state'])
        dec.load_state_dict(checkpoint['decoder_state'])

        memory.flush()
        enc.eval()
        dec.eval()

        memory_size = []
        top1_matches = []
        loss_list = []

        # Pick a batch with n_classes classes and 20 examples per class.
        # Test it on the online setting.
        test_dataset.shuffle_classes()
        batch = list(test_dataset)
        shuffled_batch = random.sample(batch, len(batch))

        # online acc
        final_data1 = []
        final_data1.append(['id', 'target_label', 'pred_label',
                            # 'pred_prob0', 'pred_prob1', 'pred_logprob0','pred_logprob1',
                            'acc'])
        target_labels = []
        pred_labels1 = []
        # online kernel
        final_data2 = []
        final_data2.append(['id', 'target_label', 'pred_label',
                            # 'pred_prob0', 'pred_prob1',
                            'acc'])
        pred_labels2 = []
        for batch_idx, (data, target) in enumerate(shuffled_batch):
            target_label = target
            target = torch.Tensor([target]).long()
            data, target = data.to(device), target.to(device)
            query_embeds = enc(data.unsqueeze(0))
            buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(query_embeds)
            top1_labels = buffer_labels[:, 0]
            top1_match = float(torch.mean((top1_labels == target).double()))

            # all
            logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
            preds = torch.argmax(logprob, dim=1)
            prob = torch.exp(logprob)
            pred = torch.argmax(prob, dim=1)
            pred_label = pred.cpu().data.numpy()[0]
            # pred_prob = prob.cpu().data.numpy()[0].flatten()
            # pred_prob0 = pred_prob[0]
            # pred_prob1 = pred_prob[1]
            # pred_logprob = logprob.cpu().data.numpy()[0].flatten()
            # pred_logprob0 = pred_logprob[0]
            # pred_logprob1 = pred_logprob[1]
            acc = float(torch.mean((preds == target).double()))
            batch_loss = F.cross_entropy(logprob, target, reduce=False)
            final_data1.append([str(batch_idx), str(target_label), str(pred_label),
                               # str(pred_prob0), str(pred_prob1), str(pred_logprob0), str(pred_logprob1),
                               str(acc) ])
            target_labels.append(target_label)
            pred_labels1.append(pred_label)

            # kernel
            n_classes = memory.n_classes
            dist_probs = F.softmax(-distances, dim=1)
            ker_probs = to_one_hot(
                buffer_labels, n_dims=n_classes + 1)[:, :, :n_classes] * dist_probs.unsqueeze(-1)
            ker_probs = torch.sum(ker_probs, dim=1)
            # kpred_prob = ker_probs.cpu().data.numpy()[0].flatten()
            # kpred_prob0 = kpred_prob[0]
            # kpred_prob1 = kpred_prob[1]
            ker_pred = torch.argmax(ker_probs, dim=1)
            kpred_label = ker_pred.cpu().data.numpy()[0]
            ker_acc = float(torch.mean((ker_pred == target).double()))
            final_data2.append([str(batch_idx), str(target_label), str(kpred_label),
                               # str(kpred_prob0), str(kpred_prob1),
                                str(ker_acc)])
            pred_labels2.append(kpred_label)

            # memory
            surprise_indices = torch.nonzero(batch_loss > nll_threshold)
            for idx in surprise_indices:
                memory.add_entry(query_embeds[idx], target[idx])
            #
            # accuracy.append(acc)
            # ker_accuracy.append(ker_acc)
            memory_size.append(len(memory))
            top1_matches.append(top1_match)
            loss_list.append(float(torch.mean(batch_loss)))

        # accuracy = np.array(accuracy)
        # ker_accuracy = np.array(ker_accuracy)
        top1_matches = np.array(top1_matches)
        # save data of each pths
        with open(online_acc + mo.split('.')[0] + '.csv', 'w', newline='') as ff:
            fft = csv.writer(ff)
            fft.writerows(final_data1)
        with open(online_acc_kernel + mo.split('.')[0] + '.csv', 'w', newline='') as ff2:
            fft2 = csv.writer(ff2)
            fft2.writerows(final_data2)

        # calculate acc
        count1 = 0
        count2 = 0
        for jj in range(len(target_labels)):
            t = target_labels[jj]
            p = pred_labels1[jj]
            p2 = pred_labels2[jj]
            if t == p: count1 += 1
            if t == p2: count2 += 1
        accc1 = float(count1/len(target_labels)*1.0)
        accc2 = float(count2/len(target_labels) * 1.0)
        final_acc1.append([mo, str(accc1), str(accc2), str(np.mean(top1_matches[-n_classes:])),
                           int(np.mean(memory_size[-n_classes:]))])


        # # Now test the same batch but with a fixed context size.
        # # f.write('Now test the same batch but with a fixed context size.\n')
        # memory.flush()
        # context, query = split_batch(batch, nshot=1, n_classes=n_classes, n_per_class=N_PER_CLASS)
        # for example in context:
        #     data = example[0].unsqueeze(0)
        #     target = torch.Tensor([example[1]]).long()
        #     data, target = data.to(device), target.to(device)
        #     memory.add_entry(enc(data), target)
        #
        # top1_matches = []
        # loss_list = []
        #
        # # fix acc
        # final_data3 = []
        # final_data3.append(['id', 'target_label', 'pred_label',
        #                     # 'pred_prob0', 'pred_prob1', 'pred_logprob0', 'pred_logprob1',
        #                     'acc'])
        # target_labelsa = []
        # pred_labels1a = []
        # # online kernel
        # final_data4 = []
        # final_data4.append(['id', 'target_label', 'pred_label',
        #                     # 'pred_prob0', 'pred_prob1',
        #                     'acc'])
        # pred_labels2a = []
        # c = 0
        # for q in query:
        #     data, target = q
        #     target_labela = target
        #     data = data.unsqueeze(0)
        #     target = torch.Tensor([target]).long()
        #     data, target = data.to(device), target.to(device)
        #
        #     query_embeds = enc(data)
        #     buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(query_embeds)
        #     top1_labels = buffer_labels[:, 0]
        #     top1_match = float(torch.mean((top1_labels == target).double()))
        #
        #     # acc
        #     logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
        #     preds = torch.argmax(logprob, dim=1)
        #     prob = torch.exp(logprob)
        #     pred = torch.argmax(prob, dim=1)
        #     pred_label = pred.cpu().data.numpy()[0]
        #     # pred_prob = prob.cpu().data.numpy()[0].flatten()
        #     # pred_prob0 = pred_prob[0]
        #     # pred_prob1 = pred_prob[1]
        #     # pred_logprob = logprob.cpu().data.numpy()[0].flatten()
        #     # pred_logprob0 = pred_logprob[0]
        #     # pred_logprob1 = pred_logprob[1]
        #     acc = float(torch.mean((preds == target).double()))
        #     batch_loss = F.cross_entropy(logprob, target, reduce=False)
        #     final_data3.append([str(c), str(target_labela), str(pred_label),
        #                         # str(pred_prob0), str(pred_prob1), str(pred_logprob0), str(pred_logprob1),
        #                         str(acc)])
        #     target_labelsa.append(target_labela)
        #     pred_labels1a.append(pred_label)
        #
        #     # kernel
        #     n_classes = memory.n_classes
        #     dist_probs = F.softmax(-distances, dim=1)
        #     ker_probs = to_one_hot(
        #         buffer_labels, n_dims=n_classes + 1)[:, :, :n_classes] * dist_probs.unsqueeze(-1)
        #     ker_probs = torch.sum(ker_probs, dim=1)
        #     kpred_prob = ker_probs.cpu().data.numpy()[0].flatten()
        #     kpred_prob0 = kpred_prob[0]
        #     kpred_prob1 = kpred_prob[1]
        #     ker_pred = torch.argmax(ker_probs, dim=1)
        #     kpred_label = ker_pred.cpu().data.numpy()[0]
        #     ker_acc = float(torch.mean((ker_pred == target).double()))
        #     final_data4.append([str(c), str(target_labela), str(kpred_label),
        #                         str(kpred_prob0), str(kpred_prob1), str(ker_acc)])
        #     pred_labels2a.append(kpred_label)
        #
        #     # accuracy.append(acc)
        #     # ker_accuracy.append(ker_acc)
        #     top1_matches.append(top1_match)
        #     loss_list.append(float(torch.mean(batch_loss)))
        #     c += 1
        #
        # # accuracy = np.array(accuracy)
        # # ker_accuracy = np.array(ker_accuracy)
        # top1_matches = np.array(top1_matches)
        # # save data of each pths
        # with open(fixed_acc + mo.split('.')[0] + '.csv', 'w', newline='') as ffa:
        #     ffta = csv.writer(ffa)
        #     ffta.writerows(final_data3)
        # with open(fixed_acc_kernel + mo.split('.')[0] + '.csv', 'w', newline='') as ffa2:
        #     ffta2 = csv.writer(ffa2)
        #     ffta2.writerows(final_data4)

        # # calculate acc
        # count1 = 0
        # count2 = 0
        # for ii in range(len(target_labelsa)):
        #     t = target_labelsa[ii]
        #     p = pred_labels1a[ii]
        #     p2 = pred_labels2a[ii]
        #     if t == p: count1 += 1
        #     if t == p2: count2 += 1
        # accc1 = float(count1/len(target_labelsa)*1.0)
        # accc2 = float(count2/len(target_labelsa) * 1.0)
        # final_acc2.append([mo, str(accc1), str(accc2), str(np.mean(top1_matches[-n_classes:]))])

    # f.close()
    with open(acc_name1, 'w', newline='') as ff0f:
        fft0 = csv.writer(ff0f)
        fft0.writerows(final_acc1)
    # with open(acc_name2, 'w', newline='') as ff0f1:
    #     fft01 = csv.writer(ff0f1)
    #     fft01.writerows(final_acc2)

# main
if __name__ == "__main__":
    check_m = check_list[0]
    model_path = 'data/checkpoints/' + check_m + '/'
    print(check_m)
    dst_path = 'result/sce_5_v2/'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)
    test_checkpoint(model_path, dst_path)
