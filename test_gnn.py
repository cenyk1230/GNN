import wandb
import torch
import time
import copy
import argparse
import torch.nn.functional as F

from tqdm import tqdm

import actnn
from actnn import config
from actnn.controller import Controller  # import actnn controller
from actnn.utils import get_memory_usage, compute_tensor_bytes
# from actnn.utils import get_memory_usage, compute_tensor_bytes, set_seeds, error_rate, get_flatten_gradient

from utils import AverageMeter

from cogdl.datasets.ogb import OGBArxivDataset
from cogdl.models.nn.gcn import GCN

wandb.init(project="ActNN-Graph")
parser = argparse.ArgumentParser(description="GNN (ActNN)")
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--actnn", action="store_true")
parser.add_argument("--get-mem", action="store_true")
args = parser.parse_args()


quantize = args.actnn
get_mem = args.get_mem
# set_seeds()

device = torch.device("cuda:0")

dataset = OGBArxivDataset()
graph = dataset[0]
graph.apply(lambda x: x.to(device))

model = GCN(
    in_feats=dataset.num_features,
    hidden_size=args.hidden_size,
    out_feats=dataset.num_classes,
    num_layers=args.num_layers,
#    num_layers = 2,
    dropout=args.dropout,
    activation="relu",
).to(device)
print(model)

actnn.set_optimization_level("L2")
controller = Controller(model)

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
    epoch_iter = tqdm(range(args.epochs))

    for i in epoch_iter:
        # measure data loading time
        data_time.update(time.time() - end)
        if get_mem and i > 0:
            print("===============After Data Loading=======================")
            init_mem = get_memory_usage(True)  # model size + data size
            torch.cuda.reset_peak_memory_stats()

        model.train()
        # compute output
        output = model(graph)
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
        with torch.no_grad():
            logits = model(graph)
            val_loss = F.cross_entropy(
                logits[graph.val_mask], graph.y[graph.val_mask]).item()
            val_acc = accuracy(logits[graph.val_mask], graph.y[graph.val_mask])

        epoch_iter.set_description(
            f"Epoch: {i}" + " val_loss: %.4f" % val_loss + " val_acc: %.4f" % val_acc)
        wandb.log({"train_loss": loss.item(),
                  "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

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
            loss = F.cross_entropy(
                output[graph.train_mask], graph.y[graph.train_mask])
            optimizer.zero_grad()
            loss.backward()
            return loss, output

        if quantize:
            controller.iterate(get_grad)

        # controller.iterate()
        # if i == 40:
        #     exit(0)

    model = best_model
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        test_acc = accuracy(logits[graph.test_mask], graph.y[graph.test_mask])
        print("Final Test Acc:", test_acc)

    print(batch_time.summary())
    print(data_time.summary())
    print(losses.summary())
    if get_mem:
        print("Peak %d MB" % (peak_mem.get_value() / 1024 / 1024))
        print("Total %d MB" % (total_mem.get_value() / 1024 / 1024))
        print("Activation %d MB" % (activation_mem.get_value() / 1024 / 1024))
