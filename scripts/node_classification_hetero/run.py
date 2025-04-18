import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from ray import train

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
        WisconsinDataset,
        FlickrDataset,
        ActorDataset,
        SquirrelDataset,
        ChameleonDataset,
    )

    g = locals()[data](verbose=False)[0]
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)
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
        self_supervise=False,
        consistency_weight=args.consistency_weight,
        activation=getattr(torch.nn, args.activation)(),
        edge_features=e.shape[-1] if e is not None else 0,
    )

    # print(model)
    # number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of parameters:", number_of_parameters, flush=True)

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


    # for _ in range(1000):
    #     optimizer.zero_grad()
    #     _, loss = model(g, g.ndata["feat"], consistency_weight=0.0)
    #     loss.backward()
    #     optimizer.step()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode="max",
    #     factor=args.factor,
    #     patience=args.patience,
    # )

    from rum.utils import EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience)

    acc_vl_max, acc_te_max = 0, 0
    for idx in range(args.n_epochs):
        optimizer.zero_grad()
        h, loss = model(g, g.ndata["feat"], e=e)
        h = h.mean(0).log()
        loss = loss + torch.nn.NLLLoss()(
            h[g.ndata["train_mask"][:,0]],  # Taking only the first row
            g.ndata["label"][g.ndata["train_mask"][:,0]], # Taking only the first row
        ) 
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            h, _ = model(g, g.ndata["feat"], e=e)
            h = h.mean(0)
            acc_tr = (
                h.argmax(-1)[g.ndata["train_mask"][:,0]] == g.ndata["label"][g.ndata["train_mask"][:,0]]
            ).float().mean().item()
            acc_vl = (
                h.argmax(-1)[g.ndata["val_mask"][:,0]] == g.ndata["label"][g.ndata["val_mask"][:,0]]
            ).float().mean().item()
            acc_te = (
                h.argmax(-1)[g.ndata["test_mask"][:,0]] == g.ndata["label"][g.ndata["test_mask"][:,0]]
            ).float().mean().item()

            # if __name__ == "__main__":
            #     print(
            #         f"Epoch: {idx+1:03d}, "
            #         f"Loss: {loss.item():.4f}, "
            #         f"Train Acc: {acc_tr:.4f}, "
            #         f"Val Acc: {acc_vl:.4f}, "
            #         f"Test Acc: {acc_te:.4f}"
            #     )
            # else:  # Add if you want all the values
            #     train.report(dict(acc_tr=acc_tr, acc_vl=acc_vl, acc_te=acc_te))

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
    parser.add_argument("--data", type=str, default="WisconsinDataset")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--n_epochs", type=int, default=10000)
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
