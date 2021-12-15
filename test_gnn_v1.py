import torch
import time
import copy
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from cogdl.datasets.ogb import OGBArxivDataset
from cogdl.models.nn.gcn import GCN
from cogdl.models import BaseModel
from cogdl.layers.actgcn_layer import ActGCNLayer

import actnn
from actnn.conf import config

wandb_log = True
try:
    import wandb
except Exception:
    wandb_log = False

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


class ActGCN(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.activation,
            args.residual,
            args.norm,
            args.rp_ratio,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout,
        activation="relu",
        residual=False,
        norm=None,
        rp_ratio=1,
    ):
        super(ActGCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                ActGCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    residual=residual if i != num_layers - 1 else None,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                    rp_ratio=rp_ratio,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def embed(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.layers[i](graph, h)
        return h

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN (ActNN)")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--actnn", action="store_true")
    parser.add_argument("--get-mem", action="store_true")
    args = parser.parse_args()

    if wandb_log:
        wandb.init(project="ActNN-Graph")

    get_mem = args.get_mem

    if args.actnn:
        actnn.set_optimization_level("L3")

    device = torch.device("cuda:0")

    dataset = OGBArxivDataset()
    graph = dataset[0]
    graph.apply(lambda x: x.to(device))

    model_class = ActGCN if args.actnn else GCN
    model = model_class(
        in_feats=dataset.num_features,
        hidden_size=args.hidden_size,
        out_feats=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation="relu",
    ).to(device)

    print("Model Parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def accuracy(y_pred, y_true):
        y_true = y_true.squeeze().long()
        preds = y_pred.max(1)[1].type_as(y_true)
        correct = preds.eq(y_true).double()
        correct = correct.sum().item()
        return correct / len(y_true)


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
        loss = F.cross_entropy(output[graph.train_mask], graph.y[graph.train_mask])

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
            val_loss = F.cross_entropy(logits[graph.val_mask], graph.y[graph.val_mask]).item()
            val_acc = accuracy(logits[graph.val_mask], graph.y[graph.val_mask])

        epoch_iter.set_description(f"Epoch: {i}" + " loss: %.4f" % loss +  " val_loss: %.4f" % val_loss + " val_acc: %.4f" % val_acc)
        if wandb_log:
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_acc": val_acc})

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

        del loss
        del output

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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
