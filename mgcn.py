import numpy as np
import torch
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
from util.meshnet import MGCN
import check.dist_check as DIST

def torch_fix_seed(seed=314):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def get_parse():
    parser = argparse.ArgumentParser(description="Self-supervised Mesh Completion")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-pos_lr", type=float, default=0.01)
    parser.add_argument("-iter", type=int, default=100)
    parser.add_argument("-k1", type=float, default=4.0)
    parser.add_argument("-k2", type=float, default=4.0)
    parser.add_argument("-dm_size", type=int, default=40)
    parser.add_argument("-kn", type=int, nargs="*", default=[4])
    parser.add_argument("-batch", type=int, default=5)
    parser.add_argument("-skip", action="store_true")
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("-cache", action="store_true")
    parser.add_argument("-CAD", action="store_true")
    parser.add_argument("-real", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

if __name__ == "__main__":
    args = get_parse()
    """ --- create dataset --- """
    mesh_dic, dataset = Datamaker.create_dataset(args.input, dm_size=args.dm_size, kn=args.kn, cache=args.cache)
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
                "iter": args.iter,
                "skip": args.skip,
            })


    """ --- create model instance --- """
    torch_fix_seed()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    posnet = MGCN(device, smo_mesh, ini_mesh, v_mask, skip=args.skip).to(device)
    optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=args.pos_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_pos, step_size=50, gamma=0.5)


    anss = posnet.poss
    v_masks = posnet.v_masks
    nvs = posnet.nvs
    meshes = posnet.meshes
    v_masks_list = posnet.v_masks_list
    poss_list = posnet.poss_list
    nvs_all = [len(meshes[0].vs)] + nvs
    pos_weight = [0.35, 0.3, 0.2, 0.15]
    # pos_weight = [0.5, 0.3, 0.15, 0.05]
    # pos_weight = [1.0, 0.0, 0.0, 0.0]

    os.makedirs("{}/output/{}_mgcn".format(args.input, dt_now), exist_ok=True)

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
                """ original dummy mask """
                dm_batch = vmask_dummy[:, batch]
                posnet.train()
                optimizer_pos.zero_grad()
                for i, b in enumerate(batch):
                    """ original dummy mask """
                    dm = dm_batch[:, i].reshape(-1, 1)
                    rm = v_mask.reshape(-1, 1).float()
                    dm = rm * dm

                    ini_vs = ini_mesh.vs

                    poss = posnet(dataset, dm)
                    pos = poss[0]
                    """ compute multiple norms (under construction) """
                    # st_nv = 0
                    # norms = None
                    # for res, mesh in enumerate(meshes):
                    #     pos = poss[st_nv:st_nv+len(mesh.vs)]
                    #     norm = Models.compute_fn(pos, mesh.faces)
                    #     if res == 0:
                    #         norms = norm
                    #     else:
                    #         norms = torch.cat([norms, norm], dim=0)

                    norm = Models.compute_fn(pos, ini_mesh.faces)
                    for mesh_idx, pos_i in enumerate(poss):
                        if mesh_idx == 0:
                            loss_p = Loss.mask_pos_rec_loss(pos_i, poss_list[mesh_idx], v_masks_list[mesh_idx].reshape(-1).bool()) * pos_weight[mesh_idx]
                        else:
                            loss_p = loss_p + Loss.mask_pos_rec_loss(pos_i, poss_list[mesh_idx], v_masks_list[mesh_idx].reshape(-1).bool()) * pos_weight[mesh_idx]
                    # loss_p = Loss.mask_pos_rec_loss(poss, anss, v_masks.reshape(-1).bool())
                    loss_n = Loss.mask_norm_rec_loss(norm, ini_mesh.fn, f_mask)
                    if args.CAD:
                        loss_reg, _ = Loss.fn_bnf_detach_loss(pos, norm, ini_mesh, loop=5)
                        loss = loss_p + args.k1 * loss_n + args.k2 * loss_reg
                        epoch_loss_r += loss_reg.item()
                    else:
                        # loss_reg = Loss.mesh_laplacian_loss(pos, ini_mesh)
                        loss = loss_p + args.k1 * loss_n# + 0.0 * loss_reg
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
            if epoch % 10 == 0:
                posnet.eval()
                dm = v_mask.reshape(-1, 1).float()
                poss = posnet(dataset, dm)
                st_nv = 0
                for res, mesh in enumerate(meshes):
                    out_path = "{}/output/{}_mgcn/{}_step_{}.obj".format(args.input, dt_now, str(epoch), res)
                    mesh.vs = poss[res].to("cpu").detach().numpy().copy()
                    st_nv += len(mesh.vs)
                    Mesh.save(mesh, out_path)
                out_path = "{}/output/{}_mgcn/{}_step_0.obj".format(args.input, dt_now, str(epoch))
                
            pbar.update(1)

    DIST.mesh_distance(mesh_dic["gt_file"], mesh_dic["org_file"], out_path, args.real)


    """ refinement """
    posnet.eval()
    dm = v_mask.reshape(-1, 1).float()
    poss = posnet(dataset, dm)
    out_pos = poss[0].to("cpu").detach()
    ini_pos = torch.from_numpy(ini_mesh.vs).float()
    if args.CAD:
        w = 0.01
    else:
        w = 1.0
    ref_pos = Mesh.mesh_merge(ini_mesh.Lap, ini_mesh, out_pos, v_mask)
    out_path = "{}/output/{}_mgcn/refine.obj".format(args.input, dt_now)
    out_mesh.vs = ref_pos.detach().numpy().copy()
    Mesh.save(out_mesh, out_path)

    DIST.mesh_distance(mesh_dic["gt_file"], mesh_dic["org_file"], out_path, args.real)