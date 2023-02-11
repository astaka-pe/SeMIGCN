import numpy as np
import torch
import torch.nn as nn
import datetime
import argparse
import os
import copy
import wandb
import random
from tqdm import tqdm

import util.loss as Loss
import util.models as Models
import util.datamaker as Datamaker
from util.mesh import Mesh
from util.networks import SingleScaleGCN
import check.dist_check as DIST

def torch_fix_seed(seed=314):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def get_parser():
    parser = argparse.ArgumentParser(description="Self-supervised Mesh Completion")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-pos_lr", type=float, default=0.01)
    parser.add_argument("-iter", type=int, default=300)
    parser.add_argument("-iter_refine", type=int, default=1000)
    parser.add_argument("-k1", type=float, default=4.0)
    parser.add_argument("-k2", type=float, default=4.0)
    parser.add_argument("-dm_size", type=int, default=40)
    parser.add_argument("-net", type=str, default="single")
    parser.add_argument("-activation", type=str, default="lrelu")
    parser.add_argument("-ant", type=int, default=0)
    parser.add_argument("-kn", type=int, nargs="*", default=[4])
    parser.add_argument("-batch", type=int, default=5)
    parser.add_argument("-skip", action="store_true")
    parser.add_argument("-drop", type=int, default=1)
    parser.add_argument("-drop_rate", type=float, default=0.6)
    parser.add_argument("-rot", type=int, default=0)
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("-cache", action="store_true")
    parser.add_argument("-CAD", action="store_true")
    parser.add_argument("-real", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def main():
    args = get_parser()
    mesh_dic, dataset = Datamaker.create_dataset(args.input, dm_size=args.dm_size, kn=args.kn, cache=args.cache)
    dataset_rot = copy.deepcopy(dataset)
    ini_file, smo_file, v_mask, f_mask, mesh_name = mesh_dic["ini_file"], mesh_dic["smo_file"], mesh_dic["v_mask"], mesh_dic["f_mask"], mesh_dic["mesh_name"]
    ini_mesh, smo_mesh, out_mesh = mesh_dic["ini_mesh"], mesh_dic["smo_mesh"], mesh_dic["out_mesh"]
    rot_mesh = copy.deepcopy(ini_mesh)
    dt_now = datetime.datetime.now()

    vmask_dummy = mesh_dic["vmask_dummy"]
    fmask_dummy = mesh_dic["fmask_dummy"]

    """ --- wandb settings --- """
    wandb.init(project="inpaint_mgcn", group=mesh_name, name=dt_now.isoformat(),
            config={
                "dm_size": args.dm_size,
                "kn": args.kn,
                "batch": args.batch,
                "rot": args.rot,
                "net": args.net,
                "annotation": args.ant,
                "activation": args.activation,
                "iter": args.iter,
                "skip": args.skip,
                "drop": args.drop,
                "drop_rate": args.drop_rate,
            })


    """ --- create model instance --- """
    torch_fix_seed()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    posnet = SingleScaleGCN(device, skip=args.skip).to(device)
    optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=args.pos_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_pos, step_size=50, gamma=0.5)

    os.makedirs("{}/output/{}_sgcn".format(args.input, dt_now), exist_ok=True)

    """ --- learning loop --- """
    with tqdm(total=args.iter) as pbar:
        """ --- training --- """
        for epoch in range(1, args.iter+1):
            n_data = vmask_dummy.shape[1]
            batch_index = torch.randperm(n_data).reshape(-1, args.batch)
            epoch_loss_p = 0.0
            epoch_loss_n = 0.0
            epoch_loss_r = 0.0
            epoch_loss = 0.0

            for batch in batch_index:
                dm_batch = vmask_dummy[:, batch]
                posnet.train()
                optimizer_pos.zero_grad()

                for i, b in enumerate(batch):
                    dm = dm_batch[:, i].reshape(-1, 1)
                    rm = v_mask.reshape(-1, 1).float()
                    dm = rm * dm
                    ini_vs = ini_mesh.vs
                    
                    pos = posnet(dataset_rot, dm)
                    norm = Models.compute_fn(pos, ini_mesh.faces)
                    loss_p = Loss.mask_pos_rec_loss(pos, ini_vs, v_mask)
                    loss_n = Loss.mask_norm_rec_loss(norm, rot_mesh.fn, f_mask)
                    if args.CAD:
                        loss_bnf, _ = Loss.fn_bnf_detach_loss(pos, norm, ini_mesh, loop=5)
                        loss = loss_p + args.k1 * loss_n + args.k2 * loss_bnf
                        epoch_loss_r += loss_bnf.item()
                    else:
                        #loss_lap = Loss.mesh_laplacian_loss(pos, ini_mesh)
                        loss = loss_p + args.k1 * loss_n#+ 0.0 * loss_lap
                    epoch_loss_p += loss_p.item()
                    epoch_loss_n += loss_n.item()
                        
                    loss.backward()
                    epoch_loss += loss.item()

                optimizer_pos.step()
            scheduler.step()

            epoch_loss_p /= n_data
            epoch_loss_n /= n_data
            epoch_loss_r /= n_data
            epoch_loss /= n_data
            wandb.log({"loss_p": epoch_loss_p}, step=epoch)
            wandb.log({"loss_n": epoch_loss_n}, step=epoch)
            wandb.log({"loss_r": epoch_loss_r}, step=epoch)
            wandb.log({"loss": epoch_loss}, step=epoch)
            pbar.set_description("Epoch {}".format(epoch))
            pbar.set_postfix({"loss": epoch_loss})
        
            """ --- test --- """
            if epoch == args.iter:
                out_path = "{}/output/{}_sgcn/{}_step.obj".format(args.input, dt_now, str(epoch))
                out_mesh.vs = pos.to("cpu").detach().numpy().copy()
                Mesh.save(out_mesh, out_path)

            if epoch % 10 == 0:
                out_path = "{}/output/{}_sgcn/{}_step.obj".format(args.input, dt_now, str(epoch))

                posnet.eval()
                dm = v_mask.reshape(-1, 1).float()
                pos = posnet(dataset, dm)
                out_mesh.vs = pos.to("cpu").detach().numpy().copy()
                Mesh.save(out_mesh, out_path) 

            pbar.update(1)

    DIST.mesh_distance(mesh_dic["gt_file"], mesh_dic["org_file"], out_path, args.real)

    """ refinement """

    """ laplacian refine """
    posnet.eval()
    dm = v_mask.reshape(-1, 1).float()
    out_pos = posnet(dataset, dm).to("cpu").detach()
    ini_pos = torch.from_numpy(ini_mesh.vs).float()
    if args.CAD:
        w = 0.01
    else:
        w = 1.0
    ref_pos = Mesh.mesh_merge(ini_mesh.Lap, ini_mesh, out_pos, v_mask, w=w)
    out_path = "{}/output/{}_sgcn/refine.obj".format(args.input, dt_now, args.net)
    out_mesh.vs = ref_pos.detach().numpy().copy()
    Mesh.save(out_mesh, out_path)
    DIST.mesh_distance(mesh_dic["gt_file"], mesh_dic["org_file"], out_path, args.real)

    """ neural refine """
    # torch_fix_seed()
    # refnet = nets["single"](device, activation=args.activation, skip=args.skip).to(device)
    # optimizer_ref = torch.optim.Adam(refnet.parameters(), lr=args.pos_lr)

    # with tqdm(total=args.iter_refine) as pbar:
    #     posnet.eval()
    #     dm = v_mask.reshape(-1, 1).float()
    #     out_pos = posnet(dataset, dm).to("cpu").detach().numpy().copy()
    #     z1 = ini_mesh.vs - out_pos
    #     dataset.z1 = torch.tensor(z1, dtype=torch.float, requires_grad=True)
    #     dataset.x_pos = torch.tensor(out_pos, dtype=torch.float)
    #     for epoch in range(args.iter+1, args.iter+args.iter_refine+1):
    #         refnet.train()
    #         optimizer_ref.zero_grad()
    #         pos = refnet(dataset, dm)
    #         norm = Models.compute_fn(pos, ini_mesh.faces)
    #         loss_p = Loss.mask_pos_rec_loss(pos, ini_mesh.vs, v_mask)
    #         loss_n = Loss.mask_norm_rec_loss(norm, rot_mesh.fn, f_mask)
    #         loss = loss_p + args.k1 * loss_n
    #         loss.backward()
    #         optimizer_ref.step()
    #         pbar.set_description("Epoch {}".format(epoch))
    #         pbar.set_postfix({"loss": loss.item()})
    #         wandb.log({"loss_p": loss_p}, step=epoch)
    #         wandb.log({"loss_n": loss_n}, step=epoch)
    #         wandb.log({"loss": loss}, step=epoch)
    #         if epoch % 50 == 0:
    #             out_path = "{}/output/{}_{}/{}_refine.obj".format(args.input, dt_now, args.net, str(epoch))
    #             out_mesh.vs = pos.to("cpu").detach().numpy().copy()
    #             Mesh.save(out_mesh, out_path)
    #         pbar.update(1)

    # DIST.mesh_distance(mesh_dic["gt_file"], mesh_dic["org_file"], out_path)

if __name__ == "__main__":
    main()