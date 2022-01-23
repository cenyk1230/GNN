from torch import optim
import wandb
import torch
import time
import copy
import argparse
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import actnn
from actnn import config
from actnn.controller import Controller  # import actnn controller
from actnn.utils import get_memory_usage, compute_tensor_bytes

from utils import AverageMeter

from cogdl.datasets.ogb import OGBArxivDataset
from cogdl.datasets.saint_data import FlickrDataset, RedditDataset, YelpDataset
from models import GCN, SAGE, GAT, GCNII


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def train(args):
    set_random_seed(args.seed)

    if not args.actnn:
        args.level = "L0"
    exp_name = f"{args.model}_{args.dataset}_{args.level}_{args.num_layers}_{args.hidden_size}_{args.dropout}"
    wandb.init(project="ActNN-Graph-Exp", name=exp_name)
    wandb.config.update(args)

    quantize = args.actnn
    get_mem = args.get_mem

    device = torch.device("cuda:0")

    if args.dataset == "ogbn-arxiv":
        dataset = OGBArxivDataset()
    elif args.dataset == "flickr":
        dataset = FlickrDataset()
    elif args.dataset == "reddit":
        dataset = RedditDataset()
    elif args.dataset == "yelp":
        dataset = YelpDataset()

    graph = dataset[0]
    graph.add_remaining_self_loops()
    graph.apply(lambda x: x.to(device))

    if args.model == "gcn":
        model = GCN(
            in_feats=dataset.num_features,
            hidden_size=args.hidden_size,
            out_feats=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
        )
    elif args.model == "sage":
        model = SAGE(
            in_feats=dataset.num_features,
            out_feats=dataset.num_classes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
        )
    elif args.model == "gat":
        model = GAT(
            in_feats=dataset.num_features,
            out_feats=dataset.num_classes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
            nhead=args.nhead,
        )
    elif args.model == "gcnii":
        model = GCNII(
            in_feats=dataset.num_features,
            out_feats=dataset.num_classes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        raise NotImplementedError
    print(model)
    model.to(device)

    if args.level == "L2.0":
        actnn.set_optimization_level("L2")
        config.bit = 3
    elif args.level == "L2.1":
        actnn.set_optimization_level("L2")
        config.bit = 2
    else:
        actnn.set_optimization_level(args.level)
    controller = Controller(model)

    if hasattr(model, "get_optimizer"):
        optimizer = model.get_optimizer(args)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    def pack_hook(tensor):  # quantize hook
        if quantize:
            return controller.quantize(tensor)
        return tensor


    def unpack_hook(tensor):  # dequantize hook
        if quantize:
            return controller.dequantize(tensor)
        return tensor


    def accuracy(y_pred, y_true):
        y_true = y_true.squeeze().long()
        preds = y_pred.max(1)[1].type_as(y_true)
        correct = preds.eq(y_true).double()
        correct = correct.sum().item()
        return correct / len(y_true)


    def multilabel_f1(y_pred, y_true, sigmoid=False):
        if sigmoid:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        else:
            y_pred[y_pred > 0] = 1
            y_pred[y_pred <= 0] = 0
        tp = (y_true * y_pred).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = (2 * precision * recall) / (precision + recall + epsilon)
        return f1.item()


    # install hook
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        total_mem = AverageMeter('Total Memory', ':.4e')
        peak_mem = AverageMeter('Peak Memory', ':.4e')
        activation_mem = AverageMeter('Activation Memory', ':.4e')

        end = time.time()

        best_model = None
        best_acc = 0
        patience = 0
        epoch_iter = tqdm(range(args.epochs))

        for i in epoch_iter:
            # measure data loading time
            data_time.update(time.time() - end)
            if get_mem and i > 0:
                print("===============After Data Loading=======================")
                init_mem = get_memory_usage(True)  # model size + data size
                torch.cuda.reset_peak_memory_stats()

            model.train()
            graph.train()
            # compute output
            output = model(graph)
            if args.dataset == "yelp":
                loss = F.binary_cross_entropy_with_logits(
                    output[graph.train_mask], graph.y[graph.train_mask].float())
            else:
                loss = F.cross_entropy(
                    output[graph.train_mask], graph.y[graph.train_mask])
            # measure accuracy and record loss
            losses.update(loss.detach().item())

            if get_mem and i > 0:
                print("===============Before Backward=======================")
                before_backward = get_memory_usage(True)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            graph.eval()
            for name, child in model.named_modules():
                if isinstance(child, torch.nn.BatchNorm1d):
                    child.train()
                    
            with torch.no_grad():
                logits = model(graph)
                if args.dataset == "yelp":
                    val_loss = F.binary_cross_entropy_with_logits(
                        output[graph.val_mask], graph.y[graph.val_mask].float()).item()
                    val_acc = multilabel_f1(logits[graph.val_mask], graph.y[graph.val_mask])
                else:
                    val_loss = F.cross_entropy(
                        logits[graph.val_mask], graph.y[graph.val_mask]).item()
                    val_acc = accuracy(logits[graph.val_mask], graph.y[graph.val_mask])

            epoch_iter.set_description(
                f"Epoch: {i}" + " val_loss: %.4f" % val_loss + " val_acc: %.4f" % val_acc)
            wandb.log({"train_loss": loss.item(),
                    "val_loss": val_loss, "val_acc": val_acc})
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                best_model = copy.deepcopy(model)
            else:
                patience += 1
                if patience >= args.patience:
                    break

            if get_mem and i > 0:
                print("===============After Backward=======================")
                after_backward = get_memory_usage(True)  # model size
                # init : weight + optimizer state + data size
                # before backward : weight + optimizer state + data size + activation + loss + output
                # after backward : init + grad
                # grad = weight
                # total - act = weight + optimizer state + data size + loss + output + grad
                total_mem.update(before_backward + (after_backward - init_mem))
                peak_mem.update(
                    torch.cuda.max_memory_allocated())
                activation_mem.update(
                    before_backward - after_backward)
                break

            del loss
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            def get_grad():
                output = model(graph)
                if args.dataset == "yelp":
                    loss = F.binary_cross_entropy_with_logits(
                        output[graph.train_mask], graph.y[graph.train_mask].float())
                else:
                    loss = F.cross_entropy(
                        output[graph.train_mask], graph.y[graph.train_mask])
                optimizer.zero_grad()
                loss.backward()
                return loss, output

            if quantize:
                controller.iterate(get_grad)

        model = best_model
        model.eval()
        graph.eval()
        with torch.no_grad():
            logits = model(graph)
            if args.dataset == "yelp":
                test_acc = multilabel_f1(logits[graph.test_mask], graph.y[graph.test_mask])
            else:
                test_acc = accuracy(logits[graph.test_mask], graph.y[graph.test_mask])
            print("Final Test Acc:", test_acc)
            wandb.log({"test_acc": test_acc})

        print(batch_time.summary())
        print(data_time.summary())
        print(losses.summary())
        if get_mem:
            print("Peak %d MB" % (peak_mem.get_value() / 1024 / 1024))
            print("Total %d MB" % (total_mem.get_value() / 1024 / 1024))
            print("Activation %d MB" % (activation_mem.get_value() / 1024 / 1024))
    
    return activation_mem.get_value() / 1024 / 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN (ActNN)")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--model", type=str, default="gcn")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--level", type=str, default="L2.1")
    parser.add_argument("--nhead", type=int, default=3)
    parser.add_argument("--norm", type=str, default="batchnorm")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--wd1", type=float, default=0)
    parser.add_argument("--wd2", type=float, default=0)
    parser.add_argument("--actnn", action="store_true")
    parser.add_argument("--get-mem", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mem = train(args)
    if args.get_mem:
        with open("act_memory.txt", "a") as f:
            f.write(f"{args.model}\t{args.dataset}\t{args.level}\t{mem:.2f}\n")
