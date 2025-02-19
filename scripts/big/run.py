import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from math import degrees
from re import sub
import torch
import dgl

def run(args):
    from dgl.data import (
        RedditDataset,
        FlickrDataset,
    )

    g = locals()[args.dataset]()[0]
    g = dgl.remove_self_loop(g)
    # g.ndata["feat"] = g.ndata["feat"] / (g.ndata["feat"].norm(dim=-1, keepdim=True) + 1e-10)

    from rum.models import RUMModel
    model = RUMModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max()+1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_samples=args.num_samples,
        length=args.length,
        temperature=args.consistency_temperature,
        dropout=args.dropout,
        num_layers=1,
        consistency_weight=args.consistency_weight,
        self_supervise=False,
        degrees=True,
        binary=False,
        activation=getattr(torch.nn, args.activation)(),
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda:0")
    
    optimizer = getattr(
        torch.optim,
        args.optimizer,
    )(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    acc_vl_max, acc_te_max = 0, 0
    for idx in range(args.n_epochs):
        model.train()
        nodes = g.ndata["train_mask"].nonzero().flatten()[torch.randperm(g.ndata["train_mask"].sum())]
        for i in range(0, g.ndata["train_mask"].sum(), args.batch_size):
            subsample = nodes[i:i+args.batch_size]
            subsample = subsample.to(g.device)
            optimizer.zero_grad()
            h, loss = model(g, g.ndata["feat"], subsample=subsample)
            h = h.mean(0).log()
            loss = loss + torch.nn.NLLLoss()(
                h, 
                g.ndata["label"][subsample],
            ) 
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            nodes = g.nodes()
            h = []
            for i in range(0, g.num_nodes(), args.batch_size):
                subsample = nodes[i:i+args.batch_size]
                subsample = subsample.to(g.device)
                _h, _ = model(g, g.ndata["feat"], subsample=subsample)
                _h = _h.mean(0).argmax(-1)
                h.append(_h)
            subsample = nodes[i+args.batch_size:]
            if subsample.numel() > 0:
                subsample = subsample.to(g.device)
                _h, _ = model(g, g.ndata["feat"], subsample=subsample)
                _h = _h.mean(0).argmax(-1)
                h.append(_h)
            
            h = torch.cat(h, dim=0)
            acc_tr = (
                h[g.ndata["train_mask"]] == g.ndata["label"][g.ndata["train_mask"]]
            ).float().mean().item()
            acc_vl = (
                h[g.ndata["val_mask"]] == g.ndata["label"][g.ndata["val_mask"]]
            ).float().mean().item()
            acc_te = (
                h[g.ndata["test_mask"]] == g.ndata["label"][g.ndata["test_mask"]]
            ).float().mean().item()

            if __name__ == "__main__":
                print(
                    f"Epoch: {idx+1:03d}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Train Acc: {acc_tr:.4f}, "
                    f"Val Acc: {acc_vl:.4f}, "
                    f"Test Acc: {acc_te:.4f}"
                )

            # else:
            #     ray.train.report(dict(acc_vl=acc_vl, acc_te=acc_te))

            if acc_vl > acc_vl_max:
                acc_vl_max = acc_vl
                acc_te_max = acc_te
                
            if early_stopping([-acc_vl]):
                break

    print(acc_vl_max, acc_te_max, flush=True)
    return acc_vl_max, acc_te_max

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Node Classification')
    parser.add_argument('--dataset', type=str, default='FlickrDataset')
    parser.add_argument('--hidden_features', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--consistency_temperature', type=float, default=1.0)
    parser.add_argument('--consistency_weight', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='SiLU')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--n_epochs', type=int, default=1000)
    args = parser.parse_args()
    run(args)