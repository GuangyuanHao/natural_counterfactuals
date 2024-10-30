import argparse
import copy
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from dscm import OursDSCM
from flow_pgm import BOXPGM
from layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score
from scipy.stats import gamma

from torch import Tensor, nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from train_pgm import eval_epoch, preprocess, sup_epoch
from utils_pgm import plot_cf, update_stats
import numpy as np
import itertools

sys.path.append("..")
from datasets import get_t_i_max_min
from hps import Hparams
from train_setup import setup_directories, setup_logging, setup_tensorboard
from utils import EMA, seed_all
from vae import HVAE
from simple_vae import VAE


def loginfo(title: str, logger: Any, stats: Dict[str, Any]):
    logger.info(f"{title} | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))


def inv_preprocess(pa: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # undo [-1,1] parent preprocessing back to original range
    
    for k, v in pa.items():
        
        if k!='digit':
            pa[k] = (v + 1) / 2  # [-1,1] -> [0,1]
            _max, _min = get_t_i_max_min(k)
            pa[k] = pa[k] * (_max - _min) + _min
        # print(k, pa[k][:2])
    # print('type(pa)', type(pa[k]))
    return pa


def save_plot(
    save_path: str,
    obs: Dict[str, Tensor],
    cfs: Dict[str, Tensor],
    do: Dict[str, Tensor],
    var_cf_x: Optional[Tensor] = None,
    num_images: int = 10,
) -> None:
    _ = plot_cf(
        obs["x"],
        cfs["x"],
        inv_preprocess({k: v for k, v in obs.items() if k != "x"}),  # pa
        inv_preprocess({k: v for k, v in cfs.items() if k != "x"}),  # cf_pa
        inv_preprocess(do),
        var_cf_x,  # counterfactual variance per pixel
        num_images=num_images,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_metrics(
    dataset: str, preds: Dict[str, List[Tensor]], targets: Dict[str, List[Tensor]]
) -> Dict[str, Tensor]:
    for k, v in preds.items():
        # print(k, type(targets[k]), targets[k])
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze().cpu()
    stats = {}
    for k in preds.keys():
        if "ukbb" in dataset:
            if k == "mri_seq" or k == "sex":
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            else:  # continuous variables
                preds_k = (preds[k] + 1) / 2  # [-1,1] -> [0,1]
                _max, _min = get_t_i_max_min(k)
                preds_k = preds_k * (_max - _min) + _min
                norm = 1000 if "volume" in k else 1  # for volume in ml
                stats[k + "_mae"] = (targets[k] - preds_k).abs().mean().item() / norm
        elif "mimic" in dataset:
            if k in ["sex", "finding"]:
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            elif k == "age":
                preds_k = (preds[k] + 1) * 50  # unormalize
                targets_k = (targets[k] + 1) * 50  # unormalize
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
            elif k == "race":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(),
                    preds[k].numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        elif args.dataset == "morphomnist":
            if k == "digit":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            else:  # continuous variables
                # print(targets[k].shape)
                stats[k + "_mae"] = (targets[k] - preds[k]).abs().mean().item()
        else:
            NotImplementedError
    return stats


def cf_epoch(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    elbo_fn: TraceStorage_ELBO,
    optimizers: Optional[Tuple] = None,
    split: str = "train",
):
    "counterfactual auxiliary training/eval epoch"
    is_train = split == "train"
    model.vae.train(is_train)
    model.pgm.eval()
    model.predictor.eval()
    stats = {k: 0 for k in ["loss", "aux_loss", "elbo", "nll", "kl", "n"]}
    steps_skipped = 0

    dag_vars = list(model.pgm.variables.keys())
    if is_train and isinstance(optimizers, tuple):
        optimizer, lagrange_opt = optimizers
    else:
        preds = {k: [] for k in dag_vars}
        targets = {k: [] for k in dag_vars}
        train_set = copy.deepcopy(dataloaders["train"].dataset.samples)

    loader = tqdm(
        enumerate(dataloaders[split]), total=len(dataloaders[split]), mininterval=0.1
    )

    for i, batch in loader:
        bs = batch["x"].shape[0]
        batch = preprocess(batch)

        with torch.no_grad():
            # randomly intervene on a single parent do(pa_k ~ p(pa_k))
            do = {}
            do_k = copy.deepcopy(args.do_pa) if args.do_pa else random.choice(dag_vars)
            if is_train:
                do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
            else:
                idx = torch.randperm(train_set[do_k].shape[0])
                do[do_k] = train_set[do_k].clone()[idx][:bs]
                do = preprocess(do)

        with torch.set_grad_enabled(is_train):
            # if not is_train:
            #     args.cf_particles = 5 if i == 0 else 1

            out = model.forward(batch, do, elbo_fn, cf_particles=args.cf_particles,t_abduct=0.1)

            if torch.isnan(out["loss"]):
                model.zero_grad(set_to_none=True)
                steps_skipped += 1
                continue

        if is_train:
            args.step = i + (args.epoch - 1) * len(dataloaders[split])
            optimizer.zero_grad(set_to_none=True)
            lagrange_opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if grad_norm < args.grad_skip:
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                model.lmbda.data.clamp_(min=0)
                ema.update()
            else:
                steps_skipped += 1
                print(f"Steps skipped: {steps_skipped} - grad_norm: {grad_norm:.3f}")
        else:  # evaluation
            with torch.no_grad():
                preds_cf = ema.ema_model.predictor.predict(**out["cfs"])
                for k, v in preds_cf.items():
                    preds[k].extend(inv_preprocess({k: v})[k])
                # interventions are the targets for prediction
                for k in targets.keys():
                    t_k = out["cfs"][k].clone() #do[k].clone() if k in do.keys() else 
                    targets[k].extend(inv_preprocess({k: t_k})[k])

        if i % args.plot_freq == 0:
            if is_train:
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in dag_vars + [None]:
                    args.do_pa = pa_k
                    valid_stats, valid_metrics = cf_epoch(  # recursion
                        args, model, ema, dataloaders, elbo_fn, None, split="valid"
                    )
                    loginfo(f"valid do({pa_k})", logger, valid_stats)
                    loginfo(f"valid do({pa_k})", logger, valid_metrics)
                args.do_pa = copy_do_pa
            # save_path = os.path.join(args.save_dir, f'{args.step}_{split}_{do_k}_cfs.pdf')
            # save_plot(save_path, batch, out['cfs'], do, out['var_cf_x'], num_images=args.imgs_plot)

        stats["n"] += bs
        stats["loss"] += out["loss"].item() * bs
        stats["aux_loss"] += out["aux_loss"].item() * args.alpha * bs
        stats["elbo"] += out["elbo"] * bs
        stats["nll"] += out["nll"] * bs
        stats["kl"] += out["kl"] * bs
        stats = update_stats(stats, elbo_fn)  # aux_model stats
        loader.set_description(
            f"[{split}] lmbda: {model.lmbda.data.item():.3f}, "
            + f", ".join(
                f'{k}: {v / stats["n"]:.3f}' for k, v in stats.items() if k != "n"
            )
            + (f", grad_norm: {grad_norm:.3f}" if is_train else "")
        )
    stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    return stats if is_train else (stats, get_metrics(args.dataset, preds, targets))



def preprocess(
    batch: Dict[str, Tensor], dataset: str = "ukbb", split: str = "l"
) -> Dict[str, Tensor]:
    if "x" in batch.keys():
        batch["x"] = (batch["x"].float().cuda() - 127.5) / 127.5  # [-1,1]
    # for all other variables except x
    not_x = [k for k in batch.keys() if k != "x"]
    for k in not_x:
        if split == "u":  # unlabelled
            batch[k] = None
        elif split == "l":  # labelled
            batch[k] = batch[k].float().cuda()
            if len(batch[k].shape) < 2:
                batch[k] = batch[k].unsqueeze(-1)
        else:
            NotImplementedError
    return batch

def preprocess_image_batch(args: Hparams, batch: Dict[str, Tensor], expand_pa: bool = False):
    
    # batch["x"] = (batch["x"].cuda().float() - 127.5) / 127.5  # [-1, 1]
    batch_list =[batch[k] for k in batch.keys() if k!="x"]
    batch["pa"] = torch.cat([t.unsqueeze(1) if len(t.size())==1 else t for t in batch_list], dim=1)
    batch["pa"] = batch["pa"].cuda().float()
    # print("expand_pa", expand_pa)
    if expand_pa:  # used for HVAE parent concatenation
        batch["pa"] = batch["pa"][..., None, None].repeat(1, 1, *(args.input_res,) * 2)
    return batch


def write_images(batch_images, name):
    bs, c, h, w = batch_images.shape
    # print(batch_images.shape)
    # print(bs)
    # Use torchvision.utils.make_grid to combine images in the batch into a single image
    grid_image = vutils.make_grid((batch_images[:20].cpu() + 1.0) * 127.5, nrow=10, normalize=True, padding=2)

    # # Visualize the combined image
    # plt.imshow(grid_image[0], cmap='gray')  # Use grayscale colormap
    # plt.axis('off')  # Turn off the axis
    
    # # Save the combined image
    # plt.savefig('../image/' + name + '.png', bbox_inches='tight', pad_inches=0)
    
    # Convert the PyTorch tensor to a PIL image
    composite_image = F.to_pil_image(grid_image)

    # Save the composite image as a file
    composite_image.save('./box_image/' + name + '.png')


def create_folder(folder_path):
    """
    Check if the specified folder path exists, and create it if it does not.

    Parameters:
    folder_path (str): The path of the folder to check and create.

    Returns:
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"'{folder_path}' does not exist, created")
    else:
        print(f"'{folder_path}' exists")


######## common tools 

def torch_exp(a):
    return torch.exp(torch.tensor(a))

def tail_min_max(a, inte=False):
    min_a = a.min()
    max_a = a.max()
    tail_min_a = gaussian_cdf(min_a)
    tail_max_a = 1.0-gaussian_cdf(max_a)
    if inte:
        if min_a<3 and min_a>-3:
            return [10000, min_a.cpu().numpy(), max_a.cpu().numpy(), tail_min_a.cpu().numpy(), tail_max_a.cpu().numpy()]
        elif max_a<3 and max_a>-3:
            return [min_a.cpu().numpy(), max_a.cpu().numpy(), tail_min_a.cpu().numpy(), tail_max_a.cpu().numpy(), 10000]
    return [min_a.cpu().numpy(), max_a.cpu().numpy(), tail_min_a.cpu().numpy(), tail_max_a.cpu().numpy()]

def tensor_min_max(a, inte=False):
    min_a = a.min()
    max_a = a.max()
    return [min_a, max_a]

def convert_dict_to_storable_structure(d):
    converted_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # print('dict key', key, type(value))
            converted_dict[key] = convert_dict_to_storable_structure(value)
        elif isinstance(value, torch.Tensor):
            converted_dict[key] = value.detach().cpu().numpy()
            # print('tensor key', key, type(value))
        else:
            # print('other key', key, type(value))
            converted_dict[key] = value
    return converted_dict

def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries with identical keys, where the values are PyTorch tensors (torch.Tensor).
    Tensors with the same key are concatenated along dimension 0, and the results are updated in dict1.

    Parameters:
    dict1 (dict): The first dictionary, which will receive the merged results.
    dict2 (dict): The second dictionary.

    Returns:
    None
    """

    # If dict1 has no keys, directly assign the contents of dict2 to dict1
    if len(dict1) == 0:
        dict1.update(convert_dict_to_storable_structure(dict2))
    else:
        # Iterate through the keys in the dictionary
        for key in dict2.keys():
            # Use np.concatenate to concatenate the two tensors along dimension 0, and update the result in dict1
            if key in dict1.keys():
                dict1[key] = np.concatenate((dict1[key], dict2[key].detach().cpu().numpy()), axis=0)
            else:
                dict1[key] = dict2[key].detach().cpu().numpy()


def convert_dict(d):
    d_= convert_dict_to_storable_structure(d)
    # print(d_)
    d.clear()
    d.update(d_)
    # print(d)

def merge_tensors(d, key, tensor2):

        if d[key] is None:
            d[key] = tensor2.detach().cpu().numpy()
        else:
            d[key] = np.concatenate((d[key], tensor2.detach().cpu().numpy()), axis=0)


###########common tools

def our_cf_epoch(
    args: Hparams, 
    model: nn.Module,
    dataloader: DataLoader
) -> Dict[str, float]:
    "caution: this can consume lots of memory if dataset is large"

    cond_dict={'c1':'c2', 's1':'s2', 's3':'c3'}
    rev_cond_dict={'c2':'c1', 's2':'s1', 'c3':'s3'}
    s_var={"c2", "s2", "c3", "m"}
    c_var={"c1", "s1", "s3"}

    ###models
    vae=model.vae
    pgm=model.pgm
    predictor=model.predictor

    vae.eval()
    pgm.eval()
    predictor.eval()
    

    ### common hyper
    natural_eps = torch.tensor([args.natural_eps]).cuda() # naturalness threshold
    print('natural eps: ', natural_eps.item())
    half_eps = natural_eps/2.0
    # allow_error = torch.tensor([1e-6]).cuda() # action error
    torch_zero = torch.tensor([0.0]).cuda()


    m = torch.distributions.normal.Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    def gaussian_cdf(value):
        return m.cdf(value)
    lower_bound = m.icdf(half_eps)
    upper_bound = m.icdf(1-half_eps)



    def ours_counterfactuals(var_name):
        
        preds_jd = {k: [] for k in predictor.variables.keys()} # predict  cf_input from cf_output
        targets_jd = {k: [] for k in predictor.variables.keys()} # cf_input

        preds_ours = {k: [] for k in predictor.variables.keys()}
        targets_ours = {k: [] for k in predictor.variables.keys()} # cf_input
        
        fact_dict = { # np
                    'obs_dict': {}, 
                    'u_dict': {}, 
                    'u_x': None,
                    'prob': [],
                    're_x':None,
                } #obs: t, i, digit, x; u:ut, ui; re_x: reconstruct x
        #prob: [pt, p(u_t), p(i|t), p(u_i)]

        jdcf_dict = { #np
                    'obs_dict': {},
                    'u_dict': {},
                    'u_x': None,
                    'prob': []
                } #not_x_dict: t, i*, digit; 'u_dict': ut,ui
        
        
        jdcf_result_dict = {
                   'targets_dict': {}, #np
                   'preds_dict': {}, #np
                   'index_not_tail': None, #list
                   'num_not_tail': None, # float
                   'num_tail': None, # float
                   'total_mean_dict': {}, # float
                   'not_tail_mean_dict': {}, # float
                   'tail_mean_dict': {}, # float
                   'mix_index':[], # list
                   'mix_mean_dict': {} # list
                } # jd: total_mean

        do_c_var=var_name.intersection(c_var)
        do_s_var=var_name.intersection(s_var)
        if len(do_c_var)!=0:
            ourscf_dict = { #np
                        'obs_dict': {},
                        'u_dict': {},
                        'u_x': None,
                        'prob': []
                    } #not_x_dict: t*^, i*^, digit u_dict: u_t*^ u_i*^
            
            ourscf_result_dict = {
                    'targets_dict': {},
                    'preds_dict': {}, 
                    'index_not_tail': None,
                    'num_not_tail': None,
                    'num_tail': None,
                    'total_mean_dict': {},
                    'not_tail_mean_dict': {},
                    'tail_mean_dict': {},
                    'mix_index':[],
                    'mix_mean_dict': {}
                    } # ours: 'not_tail_mean_dict'
                        # mix_index: ours row [0, 1] * jd colum [0, 1]
                        #mix_mean (ours row /jd colum)
                        #mix_mean keys: num; thickness, intensity
            
        
        str_eps="%.6f"%natural_eps.cpu().item()
        if 'box22' in args.vae_path:
            begin_str='box_s'
        else:
            begin_str='box_w'
        common_folder=begin_str+'/'+args.csm_order+'/'+args.vae+'/'+str_eps.split('.')[1]+'/'+''.join(sorted(var_name))+'/'
        create_folder(common_folder)
        common_name=common_folder+str(args.seed)+'_'

        if args.dataset == "box":
            # hyper for i
            if len(do_c_var)!=0:
                w_s = torch_exp(args.logw_s).cuda()
                w_c = torch_exp(args.logw_c).cuda()
                # w_action = torch_exp(2).cuda()

                w_n = torch_exp(args.logw_n).cuda() #15 #10
                print('log ws', torch.log(w_s).item())
                print('log wc', torch.log(w_c).item())
                print('log wn', torch.log(w_n).item())
                
                # print('wt', w_s.item())
                # print('wi', w_c.item())
                # print('wn', w_n.item())
                
            
        index=0
        if 'box2' in args.pgm_path:
            c_dataloader = itertools.chain(dataloader, dataloader) # each dataloader 2117*5=10585
        else:
            c_dataloader=dataloader
        # finish_index=10000//args.bs
        # index=0
        # for batch in tqdm(combined_dataloader): #
        for batch in tqdm(c_dataloader):
            # if index==finish_index:
            #     break
            # index +=1
            num_batch = batch['x'].size(0)

            interventions={}
            for k in var_name:
                interventions[k]=2*torch.rand(num_batch).reshape(-1, 1).cuda()-1
            
            ######fact
            batch = preprocess(batch, args.dataset, split="l") # x->-1to1

            merge_dicts(fact_dict['obs_dict'], batch)
            
            ### fact's noise
            not_x = [k for k in batch.keys() if k != "x"]
            batch_not_x={}
            for k in not_x:
                batch_not_x[k]=batch[k]
            # pa's noise
            u_fact = pgm.infer_exogeneous(batch_not_x) # only infer t,i exo noise
            merge_dicts(fact_dict['u_dict'], u_fact)
            
            # expand pa to input res, used for HVAE parent concatenation
            args.expand_pa = args.vae == "hierarchical"
            # merge fact's all parents
            batch4images=preprocess_image_batch(args, batch, expand_pa=args.expand_pa)
            # noise on fact x
            with torch.no_grad():
                u_x = vae.abduct(batch4images['x'], batch4images['pa'])
            merge_tensors(fact_dict,'u_x', u_x[0])
            merge_tensors(jdcf_dict,'u_x', u_x[0])
            # print("fact_dict['u_x']", fact_dict['u_x'], type(u_x[0]))
            if len(do_c_var)!=0:
                merge_tensors(ourscf_dict,'u_x', u_x[0])
            # reconstruct fact x
            with torch.no_grad():
                re_x, _ = vae.forward_latents(latents=u_x, parents=batch4images['pa'], t=0.1)
            # print(type(re_x))
            merge_tensors(fact_dict,'re_x', re_x)
            ########fact
            


            ###### jdcfs
            ## pa's jdcf
            pa_jd = pgm.counterfactual(obs=batch_not_x, intervention=interventions)
            merge_dicts(jdcf_dict['obs_dict'], pa_jd)
            # print(pa_jd.keys())
            u_jd_pa = pgm.infer_exogeneous(pa_jd) # 只能infer出t,i exo noise
            merge_dicts(jdcf_dict['u_dict'], u_jd_pa)
            # print(u_jd_pa.keys())

            # if len(do_c_var)!=0: # if do on intensity
            #     u_ours_init=u_jd_pa
            #     our_param = []
            #     #以0为初始值
            #     for k in do_c_var:
            #         u_ours_init[cond_dict[k]+'_base'] = nn.Parameter(torch.zeros(num_batch).reshape(-1, 1).cuda())
            #         our_param.append(u_ours_init[cond_dict[k]+'_base'])
            
            # Originally within support, directly assign it to initialization; others assign a value of 0 to t.
            if len(do_c_var)!=0: # if do on intensity
                
                u_ours_init=u_jd_pa
                our_param = []
                
                def index_not_tail_in_batch(u_jd_pa):
                    s_to_c =set()
                    result_list = [1] * num_batch
                    for k in do_c_var:
                        s_to_c.add(k)
                        s_to_c.add(cond_dict[k])
                    all_back_var=s_to_c.union(do_s_var)
                    for k in all_back_var:
                        temp_list=[1 if lower_bound <= x.item() <= upper_bound else 0 for x 
                                        in u_jd_pa[k+'_base']]
                        result_list= [a * b for a, b in zip(temp_list, result_list)]
                    return np.array(result_list)
                
                index_not_tail0=index_not_tail_in_batch(u_jd_pa)
                print('bbbbbbbb100',index_not_tail0[:100])
                para_init={}
                for k in do_c_var:
                    para_init[cond_dict[k]+'_base']=torch.zeros_like(u_jd_pa[cond_dict[k]+'_base'])
                    para_init[cond_dict[k]+'_base'][index_not_tail0==1]=u_jd_pa[cond_dict[k]+'_base'][index_not_tail0==1]
                

                for k in do_c_var:
                    # u_ours_init[cond_dict[k]+'_base'] = nn.Parameter(torch.zeros(num_batch).reshape(-1, 1).cuda())
                    u_ours_init[cond_dict[k]+'_base'] = nn.Parameter(para_init[cond_dict[k]+'_base'].cuda())
                    print('aaaaaa', u_ours_init[cond_dict[k]+'_base'].shape)
                    our_param.append(u_ours_init[cond_dict[k]+'_base'])
                ####################

                # for k in u_jd_pa.keys():
                    # print(k)

                    # if k in ['thickness_base', 'intensity_base']: # 以cfs为初始值
                    #     u_ours_init[k] = nn.Parameter(copy.deepcopy(u_jd_pa[k].detach()))
                    #     our_param.append(u_ours_init[k])

                    # if k in ['thickness_base', 'intensity_base']:#以fact为初始值
                    #     u_ours_init[k] = nn.Parameter(copy.deepcopy(u_fact[k].detach()))
                    #     our_param.append(u_ours_init[k])

                    # if k in ['thickness_base', 'intensity_base']:#以cfs和fact为初始值
                    #     alpha=0.2
                    #     u_ours_init[k] = nn.Parameter(copy.deepcopy(alpha*u_jd_pa[k].detach()+(1-alpha)*u_fact[k].detach()))
                    #     our_param.append(u_ours_init[k])

                    # if k in ['thickness_base', 'intensity_base']:#以gaussian为初始值
                    #     u_ours_init[k] = nn.Parameter(torch.randn(num_batch).reshape(-1, 1).cuda())
                    #     our_param.append(u_ours_init[k])
                    
                    # if k in ['thickness_base', 'intensity_base']:#t以cf为初始值, i以0为初始值
                    #     if k=='thickness_base':
                    #         u_ours_init[k] = nn.Parameter(copy.deepcopy(u_fact[k].detach()))
                    #     else:
                    #         u_ours_init[k] = nn.Parameter(torch.zeros(num_batch).reshape(-1, 1).cuda())
                    #     our_param.append(u_ours_init[k])
                
                # Number of training iterations for ourscf
                print('ourscf training iter', args.epoch_num)
                # wn_change_time=10
                # every_change_wn_num=(epoch_num-1)/wn_change_time
                # print("wn_change_time", wn_change_time)
                # two_stagepoch_num=100
                # print("two_stagepoch_num", two_stagepoch_num)

                # best for 13
                # lr 0.5
                # sch_step_size 10000
                # sch_gamma 0.8

                
                # lr 5.0
                # sch_step_size 10000
                # sch_gamma 0.1

                # args.cf_lr=5.0 #0.5 #20 best for w_n=exp10 #10
                # args.lr_step_size=10000
                # args.lr_gamma=0.3163#0.1 #0.9#0.1

                print('lr',args.cf_lr)
                print("sch_step_size", args.lr_step_size)
                print("sch_gamma", args.lr_gamma)
                optimizer = optim.SGD(our_param, lr=args.cf_lr)
                scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

                pa_ours={}
                pa_ours=copy.deepcopy(pa_jd)
        
                
                
                # n_exp=1
                for epoch in range(args.epoch_num):

                    # break
                    optimizer.zero_grad()
                    
                    
                    for k in do_c_var:
                        pa_ours[cond_dict[k]] = pgm.s_var_net[cond_dict[k]+'_flow'](u_ours_init[cond_dict[k]+'_base'])
                    
                    mix_pa={} # contain jdpa's c var, and others are pa_ours
                    for k in pa_ours.keys():
                        if k in do_c_var:
                            mix_pa[k]=pa_jd[k]
                        else:
                            mix_pa[k]=pa_ours[k]



                    temp_our_u = pgm.infer_exogeneous(mix_pa)

                    for k in do_c_var:
                        u_ours_init[k+'_base']=temp_our_u[k+'_base']
                    
                    temp_pa_ours=pgm.model_infer(u_ours_init)
                    for k in pa_ours.keys(): 
                        pa_ours[k]=temp_pa_ours[k]

                    # print('pa_ours')
                    # for k, v in pa_ours.items():
                    #     print(k, type(v)) 
                    # print('temp_our_u')
                    # for k, v in temp_our_u.items():
                    #     print(k, type(v))
                    # print('u_ours_init')
                    # for k, v in u_ours_init.items():
                    #     print(k, type(v))
                    
                    # print('temp_our_pa')
                    # for k, v in temp_our_pa.items():
                    #     print(k, type(v))

                    
            
                    # print(tensor_min_max(torch.abs(pa_ours['intensity']-interventions['intensity'])))
                    
                    
                    
                    loss_s={}
                    loss_s_avg=0
                    for k in do_c_var:
                        loss_s[cond_dict[k]]= \
                        torch.abs(gaussian_cdf(u_ours_init[cond_dict[k]+'_base'])-gaussian_cdf(u_fact[cond_dict[k]+'_base'].detach())).mean()
                        loss_s_avg +=loss_s[cond_dict[k]]
                    loss_s_avg=loss_s_avg/len(do_c_var)
                    
                    loss_c={}
                    loss_c_avg=0
                    for k in do_c_var:
                        loss_c[k]= \
                        torch.abs(gaussian_cdf(u_ours_init[k+'_base'])-gaussian_cdf(u_fact[k+'_base'].detach())).mean()
                        loss_c_avg +=loss_c[k]
                    loss_c_avg=loss_c_avg/len(do_c_var)
                    
                    
                    # loss_act = torch.max(torch.abs(pa_ours['intensity']-interventions['intensity']), allow_error)

                    loss_act={}
                    loss_act_avg=0
                    for k in do_c_var:
                        
                        loss_act[k] = torch.abs(pa_ours[k]-interventions[k]).mean()
                        loss_act_avg +=loss_act[k]
                    loss_act_avg=loss_act_avg/len(do_c_var)
                    

                    loss_n={}
                    loss_n_avg=0
                    for k in do_c_var:
                        loss_n[k] = \
                        torch.max(half_eps-gaussian_cdf(u_ours_init[k+'_base']), torch_zero).mean() + \
                        torch.max(half_eps-1+gaussian_cdf(u_ours_init[k+'_base']), torch_zero).mean()
                        loss_n_avg +=loss_n[k]

                    loss_n_avg=loss_n_avg/len(do_c_var)



                    # loss_n = \
                    #     torch.max(half_eps-gaussian_cdf(u_ours_init['thickness_base']), torch_zero) + \
                    #     torch.max(half_eps-1+gaussian_cdf(u_ours_init['thickness_base']), torch_zero) + \
                    #     torch.max(half_eps-gaussian_cdf(u_ours_init['intensity_base']), torch_zero) + \
                    #     torch.max(half_eps-1+gaussian_cdf(u_ours_init['intensity_base']), torch_zero)
                    
                    # print(loss_t.size(), loss_i.size(), loss_act.size())
                    # print(w_s, w_c, w_action)
                    
                    # loss = w_s*loss_t_mean+w_c*loss_i_mean+w_action*loss_act_mean+w_n*loss_n_mean
                    
                    # if (epoch+1)%every_change_wn_num==0:
                    #     n_exp=(epoch+1)//every_change_wn_num+1
                    #     print(epoch, "n_exp", n_exp)

                    # if epoch==two_stagepoch_num:
                        
                    #     n_exp=5
                    #     print(epoch, "n_exp", n_exp)

                    # loss = w_s*loss_t_mean+w_c*loss_i_mean+(w_n**n_exp)*loss_n_mean
                    loss = w_s*loss_s_avg+w_c*loss_c_avg+w_n*loss_n_avg
                    
                    
                    loss.backward(retain_graph=True)
                    
                    

                    # 使用torch.clamp将参数约束在指定区间内
                    for param in our_param:
                        param.data = torch.clamp(param.data, lower_bound, upper_bound)

                    # # 
                    if epoch%100==0:
                        # print(epoch, ': ', loss.item(), loss_t_mean.item(), loss_i_mean.item(), loss_act_mean.item())
                        print(epoch, ': ', loss.item(), loss_s_avg.item(), loss_c_avg.item(), loss_act_avg.item(), loss_n_avg.item())
                        # print('loss_s: '+" - ".join(f"{k}: {v}" for k, v in loss_s.items()))
                        # print('loss_c: '+" - ".join(f"{k}: {v}" for k, v in loss_c.items()))
                        # print('loss_act: '+" - ".join(f"{k}: {v}" for k, v in loss_act.items()))
                        # print('loss_n: '+" - ".join(f"{k}: {v}" for k, v in loss_n.items()))
                    
                    torch.cuda.empty_cache()
                    # 
                    optimizer.step()

                    scheduler.step()
                    # loss.backward()

                    
                    
                    
                    
                    # print(f'Epoch {epoch + 1}: Weight = {our_param}')
                # pa_ours = pgm.model_infer(u_ours_init)
                # print(tensor_min_max(torch.abs(pa_ours['thickness']-batch_not_x['thickness'])))

                # # 
                # for k in obs_dict.keys():
                #     print(k, tensor_min_max(torch.abs(pa_jd[k]-obs_dict[k])))
                #     # print(pa_jd[k])
                #     # print(obs_dict[k])

                for k in do_c_var:
                    pa_ours[cond_dict[k]] = pgm.s_var_net[cond_dict[k]+'_flow'](u_ours_init[cond_dict[k]+'_base'])
                
                mix_pa={} # contain jdpa's c var, and others are pa_ours
                for k in pa_ours.keys():
                    if k in do_c_var:
                        mix_pa[k]=pa_jd[k]
                    else:
                        mix_pa[k]=pa_ours[k]



                temp_our_u = pgm.infer_exogeneous(mix_pa)

                for k in do_c_var:
                    u_ours_init[k+'_base']=temp_our_u[k+'_base']
                
                temp_pa_ours=pgm.model_infer(u_ours_init)
                for k in pa_ours.keys(): 
                    pa_ours[k]=temp_pa_ours[k]
                

                # pa_ours['digit']=pa_jd['digit']

                # #
                # save_cf = {key: value.detach().cpu().numpy() for key, value in batch_not_x.items()}
                # save_jdcf = {key: value.detach().cpu().numpy() for key, value in pa_jd.items()}
                # save_ourscf = {key: value.detach().cpu().numpy() for key, value in pa_ours.items()}

                # all_type_cf = {'cf': save_cf, 'jdcf': save_jdcf, 'ourscf': save_ourscf}
                # np.save('morph/all_type_cf.npy', all_type_cf)
                # all_type_cf2 = np.load('morph/all_type_cf.npy', allow_pickle=True).item()
                # # print(type(all_type_cf2))
                # save_cf2 = all_type_cf2['cf']
                # save_jdcf2 = all_type_cf2['jdcf']
                # save_ourscf2 = all_type_cf2['ourscf']

                # fc_t, fc_i = save_cf2['thickness'], save_cf2['intensity']
                # jdcf_t, jdcf_i = save_jdcf2['thickness'], save_jdcf2['intensity']
                # ourscf_t, ourscf_i = save_ourscf2['thickness'], save_ourscf2['intensity']

                # print('p(t), p(t*^)', np.concatenate((p_t(fc_t),p_t(ourscf_t)), axis=1))
                # print('t, t*^', np.concatenate((fc_t, ourscf_t), axis=1))
                # print('i*^, p(i*^|t^*),P(u_{i*^})', np.concatenate((ourscf_i, p_i_con_t(ourscf_i, ourscf_t),
                #                         gaussian_cdf(u_ours_init['intensity_base']).detach().cpu().numpy(),
                #                                          ), axis=1))
                
                # #


            

                # print('p(t), p(i|t), p(i*|t)', '\n',p_t(fc_t), '\n', p_i_con_t(fc_i, fc_t), '\n',
                #       p_i_con_t(jdcf_i, fc_t), '\n')
                # print('p(t*^), p(i*^|t*^), p(i*|t*^)', '\n',
                #       p_t(ourscf_t), '\n', p_i_con_t(ourscf_i, ourscf_t), '\n', 
                #       p_i_con_t(jdcf_i, ourscf_t), '\n')
                # print('|i*^-i*|', np.abs(ourscf_i-jdcf_i))

                # def re_scale_i(i):
                #     min_max = dataloader.dataset.min_max
        
                #     i_min, i_max = min_max['intensity'][0], min_max['intensity'][1]
                #     i = ((i + 1) / 2) * (i_max - i_min) + i_min
                #     return i
            else: # thickness
                pass
                        

            if len(do_c_var)!=0:
                merge_dicts(ourscf_dict['obs_dict'], pa_ours)
                merge_dicts(ourscf_dict['u_dict'], u_ours_init)        
            
            
            # exo_obs_not_x = pgm.infer_exogeneous(batch_not_x)
            # exo_pa_ours = pgm.infer_exogeneous(pa_ours)
            # print( index, 'obs', min_max(exo_obs_not_x['intensity'+"_base"]))
            # print( index, 'int', min_max(exo_pa_ours['intensity'+"_base"], True))
            
            # print([k for k in cfs.keys()])

            
            # print('batch', min_max(batch['x']))
            # print('batch4images', min_max(batch4images['x']))
            
            
            
            
            # if index <2:
                # print(cfs_input)
            
            # reconstruct x
            # re_x, _ = vae.forward_latents(latents=u_x, parents=batch4images['pa'], t=0.1)

            # cf_x
            # cf_x, _ = vae.forward_latents(latents=u_x, parents=cfs_pa, t=0.1)
            # print(cfs_pa.shape, batch4images['pa'].shape)
            def add_rest_dict(pa_cf, data_dict, preds, targets, cf_type):
                # add pa* x* to jdcf_dict and ourscf_dict
                # add batch information to preds and targets
                # merge pa
                cfs_list =[pa_cf[k] for k in pa_cf.keys()]
                cfs_pa = torch.cat([t.unsqueeze(1) if len(t.size())==1 else t for t in cfs_list], dim=1)
                
                
                # used for HVAE parent concatenation
                if args.expand_pa:
                    cfs_pa = cfs_pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2) 

                # generate cfs images
                with torch.no_grad():
                    cf_x, _ = vae.forward_latents(latents=u_x, parents=cfs_pa, t=0.1)

                merge_dicts(data_dict['obs_dict'], {'x':cf_x})
                

                if index <=2:
                    # print(gen_images.max(), gen_images.min())
                    # print(cfs_input[:2])
                    write_images(batch['x'], ''.join(sorted(var_name))+cf_type+str(index)+'org')
                    # write_images(re_x, str(index)+'re')
                    write_images(cf_x, ''.join(sorted(var_name))+cf_type+str(index)+'cf')

                # pa_cf to target; cfs_all=pa+x
                cfs_all={}
                cfs_all['x']=cf_x
                for k in targets.keys():
                    targets[k].extend(copy.deepcopy(pa_cf[k].detach().cpu()))
                    cfs_all[k]=pa_cf[k]
                
                # predict pa_cf^ from pa_cf
                with torch.no_grad():
                    out = predictor.predict(**cfs_all)
                for k, v in out.items():
                    preds[k].extend(v.detach().cpu())
            
            if len(do_c_var)!=0:
                add_rest_dict(pa_ours, ourscf_dict, preds_ours, targets_ours, 'ours')
            add_rest_dict(pa_jd, jdcf_dict, preds_jd, targets_jd, 'jd')
            

        def index_not_tail(data_dict):
            if 'box2' in args.pgm_path:
                result_list = [1] * len(dataloader.dataset)*2
            else:
                result_list = [1] * len(dataloader.dataset)
            if len(do_c_var)!=0:
                s_to_c =set()
                for k in do_c_var:
                    s_to_c.add(k)
                    s_to_c.add(cond_dict[k])
                all_back_var=s_to_c.union(do_s_var)
                
            else:
                all_back_var=var_name
            
            for k in all_back_var:
                temp_list=[1 if lower_bound <= x.item() <= upper_bound else 0 for x 
                                in data_dict['u_dict'][k+'_base']]
                result_list= [a * b for a, b in zip(temp_list, result_list)]
            num_not_tail = sum(result_list)
            num_tail = len(result_list)-num_not_tail
            return np.array(result_list), num_not_tail, num_tail
            
            
        if len(do_c_var)!=0:
            # index_not_tail, mix_index
            ourscf_result_dict["index_not_tail"], \
            ourscf_result_dict['num_not_tail'],\
            ourscf_result_dict['num_tail']= index_not_tail(ourscf_dict)

        
        jdcf_result_dict["index_not_tail"], \
        jdcf_result_dict['num_not_tail'],\
        jdcf_result_dict['num_tail']= index_not_tail(jdcf_dict)

        if len(do_c_var)!=0:
            ours_index_not_tail =ourscf_result_dict["index_not_tail"]
        jd_index_not_tail = jdcf_result_dict["index_not_tail"]

        if len(do_c_var)!=0:
            ### ours_index as row; jd as column
            for i in [1, 0]:
                for j in [1, 0]:
                    ourscf_result_dict['mix_index'].append(np.array([abs((i-a) * (j-b)) for a, b 
                                            in zip(ours_index_not_tail, jd_index_not_tail)]))
            jdcf_result_dict['mix_index']=ourscf_result_dict['mix_index']
            
            ourscf_result_dict['mix_mean_dict']['4nums']=np.array([
                [sum(ourscf_result_dict['mix_index'][0]), sum(ourscf_result_dict['mix_index'][1])],
                [sum(ourscf_result_dict['mix_index'][2]), sum(ourscf_result_dict['mix_index'][3])]
            ])

            jdcf_result_dict['mix_mean_dict']['4nums']=ourscf_result_dict['mix_mean_dict']['4nums']

        # index_not_tail, mix_index

        def stats_func(preds, targets, result_dict, index_not_tail, mix_index):
            # index of not_x_dict
            for k, v in preds.items():
                preds[k] = torch.stack(v).squeeze()
                targets[k] = torch.stack(targets[k]).squeeze()
            
            result_dict['targets_dict']=targets 
            result_dict['preds_dict']=preds

            stats = {}
            if args.dataset == "box":
                for k in pgm.variables.keys():

                # continuous variables
                    # unormalize from [-1,1] back to original range

                    mae= (targets[k] - preds[k]).abs()
                    
                    result_dict['total_mean_dict'][k]=  mae.mean().item()   
                    stats[k + "_total_mae"] = result_dict['total_mean_dict'][k]
                    # print('mae.shape', mae.shape, mae, index_not_tail)
                    result_dict['not_tail_mean_dict'][k]=  mae[index_not_tail==1].mean().item()   
                    stats[k + "_not_tail_mae"] = result_dict['not_tail_mean_dict'][k]

                    result_dict['tail_mean_dict'][k]=  mae[index_not_tail==0].mean().item()     
                    stats[k + "_tail_mae"] = result_dict['tail_mean_dict'][k]
                    
                    if len(do_c_var)!=0:
                        result_dict['mix_mean_dict'][k]= []
                        for i in range(4):
                            stats[k + "_mix"+str(i)+"_mae"]=mae[mix_index[i]==1].mean().item()  
                            result_dict['mix_mean_dict'][k].append(stats[k + "_mix"+str(i)+"_mae"])     
                if len(do_c_var)!=0:
                    stats['mix4num']=[np.sum(mix_index[i]==1) for i in range(4)] 
            return stats
        
        
        def save_npy(name, _dict):
            # convert_dict(_dict)
            name=common_name+name
            np.save(name, _dict)
        def load_npy(name):
            name=common_name+name+'.npy'
            return np.load(name, allow_pickle=True).item()

        if len(do_c_var)!=0:
            stats_ours = stats_func(preds_ours, targets_ours, ourscf_result_dict, 
                                    ourscf_result_dict["index_not_tail"], ourscf_result_dict["mix_index"])
            save_npy('stats_ours', stats_ours)
            stats_ours=load_npy('stats_ours')

        stats_jd = stats_func(preds_jd, targets_jd, jdcf_result_dict, 
                                jdcf_result_dict["index_not_tail"], jdcf_result_dict["mix_index"])

        save_npy('stats_jd', stats_jd)
        stats_jd=load_npy('stats_jd')
        
        #### save
        
            

        for key in fact_dict['u_dict'].keys():
            # key_not_base=key.split('_')[0]
            # fact_dict['prob'].append(prob_func(fact_dict['obs_dict'],key_not_base))
            fact_dict['prob'].append(gaussian_cdf(torch.from_numpy(fact_dict['u_dict'][key]).cuda()).detach().cpu().numpy())
            
            # jdcf_dict['prob'].append(prob_func(jdcf_dict['obs_dict'],key_not_base))
            jdcf_dict['prob'].append(gaussian_cdf(torch.from_numpy(jdcf_dict['u_dict'][key]).cuda()).detach().cpu().numpy())
            if len(do_c_var)!=0:
                # ourscf_dict['prob'].append(prob_func(ourscf_dict['obs_dict'],key_not_base))
                ourscf_dict['prob'].append(gaussian_cdf(torch.from_numpy(ourscf_dict['u_dict'][key]).cuda()).detach().cpu().numpy())
        fact_dict['prob'] = np.concatenate(tuple(fact_dict['prob']), axis=1)
        jdcf_dict['prob'] = np.concatenate(tuple(jdcf_dict['prob']),  axis=1)
        if len(do_c_var)!=0:
            ourscf_dict['prob'] = np.concatenate(tuple(ourscf_dict['prob']),  axis=1)



        


        save_npy('fact', fact_dict)
        # print(fact_dict['obs_dict']['x'])
        # print('fact obs dict', type(fact_dict['obs_dict']['x']), 
        #       type(fact_dict['obs_dict']['digit']),
        #       type(fact_dict['obs_dict']['thickness']),
        #            type(fact_dict['obs_dict']['intensity']))

        # print('fact u dict', type(fact_dict['u_dict']['intensity_base']), 
        #       type(fact_dict['u_dict']['thickness_base']))
        # print('u_x', type(fact_dict['u_x']))
        save_npy('jdcf', jdcf_dict)
        save_npy('jdcf_result', jdcf_result_dict)

        
        
        if len(do_c_var)!=0:
            save_npy('ourscf', ourscf_dict)
            save_npy('ourscf_result', ourscf_result_dict)

        
        fact2=load_npy('fact')
        jdcf2=load_npy('jdcf')
        jdcf_result2=load_npy('jdcf_result')
        
        print('rrrrrrrrrrrrr', fact2['obs_dict']['x'].shape)

        if len(do_c_var)!=0:
            ourscf2=load_npy('ourscf')
            ourscf_result2=load_npy('ourscf_result')

######## check whether those dicts are right?
        def return_samples_from_dict(d):

            for key, value in d.items():
                # print(type(value))
                if 'mean' in key:
                    print(key, value)
                elif isinstance(value, dict):
                    print(key)
                    return_samples_from_dict(value)
                elif isinstance(value, np.ndarray):
                    print(key, value.shape, value[:2])
                elif isinstance(value, list):
                    if key=='index_not_tail':
                        print(key, len(value), value[:2])
                    elif key=='mix_index':
                        index=0
                        for ls in value:
                            print(key+str(index), ls[:2])
                            index+=1
                    else:
                        print(key, value)
                else:
                    print(key, value)

        if len(do_c_var)!=0: 
            dict_list=[fact2, jdcf2, ourscf2, jdcf_result2, ourscf_result2]
        else:
            dict_list=[fact2, jdcf2, jdcf_result2]
        for d in dict_list:
            return_samples_from_dict(d)
######## check whether those dicts are right?     
        
        if len(do_c_var)!=0:
            return [stats_ours, stats_jd]
        else:
            return stats_jd
    
    
    # var_list={0:{'c2'}, 1:{'c1'}} #"thickness", 
    var_list={0:{'c1', 's1', 's3'}}
    # var_list={0:{'c1'}, 2:{'s1'}, 3:{'s3'}, 4:{'c1', 's1', 's3'}}
    # {0:{'c1'}, 2:{'s1'}, 3:{'s3'}, 4:{'c1', 's1', 's3'},
    #  5:{'c2'}, 6:{'s2'}, 7:{'c3'}, 8:{'m'},
    #  9:{'c2', 's2', 'c3', 'm'}}
    stats_var_dict={}    
    for k, var_name in var_list.items():
        stats_var_dict[k]=ours_counterfactuals(var_name)
    return stats_var_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment name.", type=str, default="")
    parser.add_argument(
        "--gpu", help="gpu.", type=int, default=0)
    
    parser.add_argument("--dataset", help="Dataset name.", type=str, default="morphomnist")
    parser.add_argument(
        "--csm_order", help="csm order.", type=str, default="prh"
    )
    parser.add_argument(
        "--data_dir", help="data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--load_path", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument(
        "--pgm_path",
        help="path to load pgm checkpoint.",
        type=str,
        default="../../checkpoints/sup_pgm/checkpoint.pt",
    )
    parser.add_argument(
        "--predictor_path",
        help="path to load predictor checkpoint.",
        type=str,
        default="../../checkpoints/sup_aux_prob/checkpoint.pt",
    )
    parser.add_argument(
        "--vae",
        help="VAE model: simple/hierarchical.",
        type=str,
        default="hierarchical",
    )
    parser.add_argument(
        "--vae_path",
        help="path to load vae checkpoint.",
        type=str,
        default="../../checkpoints/from_server/m_b_v_s/ukbb192_beta5_dgauss_b33/checkpoint.pt",
    )
    parser.add_argument("--seed", help="random seed.", type=int, default=7)
    
    parser.add_argument(
        "--setup",  # semi_sup/sup_pgm/sup_aux
        help="training setup.",
        type=str,
        default="sup_pgm",
    )

    parser.add_argument(
        "--deterministic",
        help="toggle cudNN determinism.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--testing", help="test model.", action="store_true", default=False
    )
    # training
    parser.add_argument("--epochs", help="num training epochs.", type=int, default=5000)
    parser.add_argument("--bs", help="batch size.", type=int, default=32)
    parser.add_argument("--lr", help="learning rate.", type=float, default=1e-4)
    parser.add_argument(
        "--lr_lagrange", help="learning rate for multipler.", type=float, default=1e-2
    )
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument("--alpha", help="aux loss multiplier.", type=float, default=1)
    parser.add_argument(
        "--lmbda_init", help="lagrange multiplier init.", type=float, default=0
    )
    parser.add_argument(
        "--damping", help="lagrange damping scalar.", type=float, default=100
    )
    parser.add_argument("--do_pa", help="intervened parent.", type=str, default=None)
    parser.add_argument("--eval_freq", help="epochs per eval.", type=int, default=1)
    parser.add_argument("--plot_freq", help="steps per plot.", type=int, default=500)
    parser.add_argument("--imgs_plot", help="num images to plot.", type=int, default=10)
    parser.add_argument(
        "--cf_particles", help="num counterfactual samples.", type=int, default=1
    )


########### train ourscf
    parser.add_argument(
        "--epoch_num", help="epoch num.", type=int, default=50001
    )
    parser.add_argument(
        "--natural_eps", help="natural_eps.", type=float, default=1e-3
    )
    parser.add_argument(
        "--cf_lr", help="cf lr.", type=float, default=5.0
    )
    parser.add_argument(
        "--lr_step_size", help="reduce lr in every step size.", type=int, default=10000
    )

    parser.add_argument(
        "--lr_gamma", help="lr gamma.", type=float, default=0.1
    )

    parser.add_argument(
        "--logw_s", help="wt.", type=int, default=1
    )
    parser.add_argument(
        "--logw_c", help="wi.", type=int, default=0
    )
    parser.add_argument(
        "--logw_n", help="wn.", type=int, default=13
    )
  #################  




    args = parser.parse_known_args()[0]
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # update hparams if loading checkpoint
    if args.load_path:
        if os.path.isfile(args.load_path):
            print(f"\nLoading checkpoint: {args.load_path}")
            ckpt = torch.load(args.load_path)
            ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "load_path"}
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.testing:
                ckpt_args["testing"] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.load_path}")

    seed_all(args.seed, args.deterministic)
    # _bs =args.bs
    # Load predictors
    print(f"\nLoading predictor checkpoint: {args.predictor_path}")
    predictor_checkpoint = torch.load(args.predictor_path)
    predictor_args = Hparams()
    predictor_args.update(predictor_checkpoint["hparams"])
    predictor_args.bs =args.bs
    predictor = BOXPGM(predictor_args).cuda()
    predictor.load_state_dict(predictor_checkpoint["ema_model_state_dict"])

    # for backwards compatibility
    if not hasattr(predictor_args, "dataset"):
        predictor_args.dataset = "ukbb"
    if hasattr(predictor_args, "loss_norm"):
        args.loss_norm

    from train_pgm import setup_dataloaders

    if args.data_dir != "":
        predictor_args.data_dir = args.data_dir
    dataloaders = setup_dataloaders(predictor_args)
    
    if False:
        elbo_fn = TraceStorage_ELBO(num_particles=1)

        test_stats = sup_epoch(
            predictor_args,
            predictor,
            None,
            dataloaders["test"],
            elbo_fn,
            optimizer=None,
            is_train=False,
        )
        stats = eval_epoch(predictor_args, predictor, dataloaders["test"])
        print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))

    # Load PGM
    print(f"\nLoading PGM checkpoint: {args.pgm_path}")
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint["hparams"])
    pgm_args.bs=args.bs
    pgm = BOXPGM(pgm_args).cuda()
    pgm.load_state_dict(pgm_checkpoint["ema_model_state_dict"])

    # for backwards compatibility
    if not hasattr(pgm_args, "dataset"):
        pgm_args.dataset = "ukbb"
    if args.data_dir != "":
        pgm_args.data_dir = args.data_dir

    if False:    
        dataloaders = setup_dataloaders(pgm_args)
        elbo_fn = TraceStorage_ELBO(num_particles=1)

        test_stats = sup_epoch(
            pgm_args, pgm, None, dataloaders["test"], elbo_fn, is_train=False
        )

    # Load deep VAE
    print(f"\nLoading VAE checkpoint: {args.vae_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae_checkpoint = torch.load(args.vae_path, map_location=device)
    vae_args = Hparams()
    vae_args.update(vae_checkpoint["hparams"])
    vae_args.bs=args.bs
    if not hasattr(vae_args, "cond_prior"):  # for backwards compatibility
        vae_args.cond_prior = False
    vae_args.kl_free_bits = vae_args.kl_free_bits
    vae_args.vae=args.vae
    vae_args.dataset=args.dataset
    vae_args.setup=args.setup
    # init model
    if vae_args.vae == "hierarchical":
        model_vae = HVAE
    elif vae_args.vae == "simple":
        model_vae = VAE
    else:
        NotImplementedError

    vae = model_vae(vae_args).cuda()
    vae.load_state_dict(vae_checkpoint["ema_model_state_dict"])

    # vae_args.data_dir = None  # adjust data_dir as needed
    if args.data_dir != "":
        vae_args.data_dir = args.data_dir
    
    if False:
        dataloaders = setup_dataloaders(vae_args)

        @torch.no_grad()
        def vae_epoch(args, vae, dataloader):
            vae.eval()
            stats = {k: 0 for k in ["elbo", "nll", "kl", "n"]}
            loader = tqdm(enumerate(dataloader), total=len(dataloader))
            
            args.expand_pa = args.vae == "hierarchical"
            for i, batch in loader:
                # preprocessing
                batch["x"] = (batch["x"].cuda().float() - 127.5) / 127.5  # [-1, 1]
                
                batch_list =[batch[k].float().cuda() for k in batch.keys() if k!="x"]
                batch["pa"] = torch.cat([t.unsqueeze(1) if len(t.size())==1 else t for t in batch_list], dim=1)
                
                if args.expand_pa:
                    batch["pa"] = (
                        batch["pa"][..., None, None]
                        .repeat(1, 1, args.input_res, args.input_res)
                    )
                

                # forward pass
                out = vae(batch["x"], batch["pa"], beta=args.beta)
                # update stats
                bs = batch["x"].shape[0]
                stats["n"] += bs  # samples seen counter
                stats["elbo"] += out["elbo"] * bs
                stats["nll"] += out["nll"] * bs
                stats["kl"] += out["kl"] * bs
                loader.set_description(
                    f' => eval | nelbo: {stats["elbo"] / stats["n"]:.4f}'
                    + f' - nll: {stats["nll"] / stats["n"]:.4f}'
                    + f' - kl: {stats["kl"] / stats["n"]:.4f}'
                )
            return {k: v / stats["n"] for k, v in stats.items() if k != "n"}

        stats = vae_epoch(vae_args, vae, dataloaders["test"])

    # setup current experiment args
    args.beta = vae_args.beta
    args.parents_x = vae_args.parents_x
    args.input_res = vae_args.input_res
    args.grad_clip = vae_args.grad_clip
    args.grad_skip = vae_args.grad_skip
    args.elbo_constraint = 1.841216802597046  # train set elbo constraint
    args.wd = vae_args.wd
    args.betas = vae_args.betas

    # init model
    if not hasattr(vae_args, "dataset"):
        args.dataset = "ukbb"
    model = OursDSCM(args, pgm, predictor, vae)
    ema = EMA(model, beta=args.ema_rate)
    model.cuda()
    ema.cuda()

    # setup data
    pgm_args.concat_pa = False
    pgm_args.bs = args.bs
    from train_pgm import setup_dataloaders

    dataloaders = setup_dataloaders(pgm_args)

    # test model
    if False:
        stats, metrics = cf_epoch(
            args, model, ema, dataloaders, elbo_fn, None, split="test"
        )
        print(f"\n[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        print(f"[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

  
    stats_var_dict = our_cf_epoch(
            args,
            model,
            dataloaders["test"]
        )


    for ks in stats_var_dict.keys():
        if len(stats_var_dict[ks])==2:
            [ours, jd]=stats_var_dict[ks]
            print(str(ks)+ " ours: test | " + " - ".join(f"{k}: {v}" for k, v in ours.items()))
            print(str(ks)+ " jd: test | " + " - ".join(f"{k}: {v}" for k, v in jd.items()))
        else:
            print(str(ks)+ ": test | " + " - ".join(f"{k}: {v}" for k, v in stats_var_dict[ks].items()))

