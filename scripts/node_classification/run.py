import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import copy
import numpy as np
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)

def get_graph(data):
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
        CornellDataset,
        TexasDataset,
        FlickrDataset,
    )

    g = locals()[data](verbose=False)[0]
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)

    if "train_mask" not in g.ndata:
        g.ndata["train_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["val_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["test_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)

        train_idxs = torch.tensor([], dtype=torch.int32)
        val_idxs = torch.tensor([], dtype=torch.int32)
        test_idxs = torch.tensor([], dtype=torch.int32)

        n_classes = g.ndata["label"].max() + 1
        for idx_class in range(n_classes):
            idxs = torch.where(g.ndata["label"] == idx_class)[0]
            # print(idxs)
            assert len(idxs) > 50
            idxs = idxs[torch.randperm(len(idxs))]
            _train_idxs = idxs[:20]
            _val_idxs = idxs[20:50]
            _test_idxs = idxs[50:]
            train_idxs = torch.cat([train_idxs, _train_idxs])
            val_idxs = torch.cat([val_idxs, _val_idxs])
            test_idxs = torch.cat([test_idxs, _test_idxs])

        g.ndata["train_mask"][train_idxs] = True
        g.ndata["val_mask"][val_idxs] = True
        g.ndata["test_mask"][test_idxs] = True
    return g

def run(args):
    g = get_graph(args.data)

    if args.directed:
        _g = g
        g = dgl.to_bidirected(g, copy_ndata=True)
        src, dst = g.edges()
        has_fwd_edge = _g.has_edges_between(src.flatten(), dst.flatten()) * 1.0
        has_bwd_edge = _g.has_edges_between(dst.flatten(), src.flatten()) * 1.0
        has_edge = has_fwd_edge - has_bwd_edge
        e = has_edge.unsqueeze(-1)
    else:
        e = None

    if args.split_index >= 0:
        g.ndata["train_mask"] = g.ndata["train_mask"][:, args.split_index]
        g.ndata["val_mask"] = g.ndata["val_mask"][:, args.split_index]
        g.ndata["test_mask"] = g.ndata["test_mask"][:, args.split_index]

    
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
        num_layers=args.num_layers,
        self_supervise_weight=args.self_supervise_weight,
        consistency_weight=args.consistency_weight,
        activation=getattr(torch.nn, args.activation)(),
        edge_features=e.shape[-1] if e is not None else 0,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda")
        if e is not None:
            e = e.cuda()

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
        optimizer.zero_grad()
        h, loss = model(g, g.ndata["feat"], e=e)
        h = h.mean(0).log() # This mean is different it's over the 3rd dimension
        loss = loss + torch.nn.NLLLoss()(
            h[g.ndata["train_mask"]], 
            g.ndata["label"][g.ndata["train_mask"]],
        ) 
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            h, _ = model(g, g.ndata["feat"], e=e)
            h = h.mean(0) # This mean is different it's over the 3rd dimension
            acc_tr = (
                h.argmax(-1)[g.ndata["train_mask"]] == g.ndata["label"][g.ndata["train_mask"]]
            ).float().mean().item()
            acc_vl = (
                h.argmax(-1)[g.ndata["val_mask"]] == g.ndata["label"][g.ndata["val_mask"]]
            ).float().mean().item()
            acc_te = (
                h.argmax(-1)[g.ndata["test_mask"]] == g.ndata["label"][g.ndata["test_mask"]]
            ).float().mean().item()

            # if __name__ == "__main__":
            #     print(
            #         f"Epoch: {idx+1:03d}, "
            #         f"Loss: {loss.item():.4f}, "
            #         f"Train Acc: {acc_tr:.4f}, "
            #         f"Val Acc: {acc_vl:.4f}, "
            #         f"Test Acc: {acc_te:.4f}"
            #     )

            # scheduler.step(acc_vl)

            # if optimizer.param_groups[0]["lr"] < 1e-6:
            #     break

            if acc_vl > acc_vl_max:
                acc_vl_max = acc_vl
                acc_te_max = acc_te
                
            if early_stopping([-acc_vl]):
                break
    
    print(acc_vl_max, acc_te_max, flush=True)
    return acc_vl_max, acc_te_max
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_epochs", type=int, default=1000)
    # parser.add_argument("--factor", type=float, default=0.5)
    # parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--self_supervise_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1)
    parser.add_argument("--consistency_temperature", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--directed", type=int, default=0)
    args = parser.parse_args()
    run(args)
