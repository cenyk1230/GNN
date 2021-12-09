import torch
import time
import torch.nn.functional as F

from actnn.controller import Controller # import actnn controller
from actnn import get_memory_usage, compute_tensor_bytes

from utils import AverageMeter

from cogdl.datasets.ogb import OGBArxivDataset
from cogdl.models.nn.gcn import GCN

actnn = True
get_mem = True

controller = Controller(default_bit=4, swap=False, debug=False, prefetch=False)

device = torch.device("cuda:0")

dataset = OGBArxivDataset()
graph = dataset[0]
graph.apply(lambda x: x.to(device))

model = GCN(
    in_feats=dataset.num_features,
    hidden_size=512,
    out_feats=dataset.num_classes,
    num_layers=3,
    dropout=0.0,
    activation=None,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

controller.filter_tensors(model.named_parameters()) # do not quantize parameters

def pack_hook(tensor): # quantize hook
    if actnn:
        return controller.quantize(tensor)
    return tensor

def unpack_hook(tensor): # dequantize hook
    if actnn:
        return controller.dequantize(tensor)
    return tensor

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook): # install hook

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    total_mem = AverageMeter('Total Memory', ':.4e')
    peak_mem = AverageMeter('Peak Memory', ':.4e')
    activation_mem = AverageMeter('Activation Memory', ':.4e')

    end = time.time()

    for i in range(10):
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
            print(val_loss, val_acc)

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

        controller.iterate()

    print(batch_time.summary())
    print(data_time.summary())
    print(losses.summary())
    print("Peak %d MB" % (peak_mem.get_value() / 1024 / 1024))
    print("Total %d MB" % (total_mem.get_value() / 1024 / 1024))
    print("Activation %d MB" % (activation_mem.get_value() / 1024 / 1024))