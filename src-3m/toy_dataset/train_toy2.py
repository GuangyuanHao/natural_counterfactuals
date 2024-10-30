import argparse
import copy
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
sys.path.append("..")
from pgm.flow_pgm import TOYPGM
from pgm.layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score
from scipy.stats import gamma

from torch import Tensor, nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import numpy as np



from datasets import get_t_i_max_min
from hps import Hparams
from train_setup import setup_directories, setup_logging, setup_tensorboard
from utils import EMA, seed_all


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

    # If dict1 is empty, directly assign the contents of dict2 to dict1
    if len(dict1) == 0:
        dict1.update(convert_dict_to_storable_structure(dict2))
    else:
        # Iterate through the keys in the dictionary
        for key in dict2.keys():
            # Use np.concatenate to concatenate the two tensors along dimension 0, and update the result in dict1
            dict1[key] = np.concatenate((dict1[key], dict2[key].detach().cpu().numpy()), axis=0)



def convert_dict(d):
    d_= convert_dict_to_storable_structure(d)
    # print(d_)
    d.clear()
    d.update(d_)
    # print(d)

def merge_tensors(d, key, tensor2):
    """
    Merges two PyTorch tensors along dimension 0.
    If tensor1 is None, directly assigns tensor2 to tensor1.

    Parameters:
    tensor1 (torch.Tensor or None): The first tensor.
    tensor2 (torch.Tensor): The second tensor.

    Returns:
    None
    """
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

    cond_dict={'n2':'n1', 'n3':'n2'}
    cond_back_dict={'n2':['n1'], 'n3':['n2', 'n1']}
    # rev_cond_dict={'c2':'c1', 's2':'s1', 'c3':'s3'}
    s_var={"n1"}
    c_var={"n2", 'n3'}

    ###models
    
    pgm=model
    

    
    pgm.eval()
    
    

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
        
        preds_jd = {k: [] for k in pgm.variables.keys()} # predict  cf_input from cf_output
        targets_jd = {k: [] for k in pgm.variables.keys()} # cf_input

        preds_ours = {k: [] for k in pgm.variables.keys()}
        targets_ours = {k: [] for k in pgm.variables.keys()} # cf_input
        
        fact_dict = { # np
                    'obs_dict': {}, 
                    'u_dict': {}, 
                    'prob': []
                } #obs: t, i, digit, x; u:ut, ui; re_x: reconstruct x
        #prob: [pt, p(u_t), p(i|t), p(u_i)]

        jdcf_dict = { #np
                    'obs_dict': {},
                    'u_dict': {},
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
        common_folder=args.data_dir+'/'+str_eps.split('.')[1]+'/'+''.join(sorted(var_name))+'/'
        create_folder(common_folder)
        common_name=common_folder+str(args.seed)+'_'

        if args.dataset == "toy":
            # hyper for i
            if len(do_c_var)!=0:
                w_1 = 3
                w_2 = 2
                w_3 = 1
            

                w_n = torch_exp(args.logw_n).cuda() #15 #10
                print('log wn', torch.log(w_n).item())
                
                
                
            
        index=0
        for batch in dataloader: #
        # for batch in tqdm(dataloader):
            # if index==2:
            #     break
            index +=1
            num_batch = batch['n1'].size(0)

            interventions={}
            for k in var_name:
                interventions[k]=2*torch.rand(num_batch).reshape(-1, 1).cuda()-1
            
            ######fact
            batch = preprocess(batch, args.dataset, split="l") # x->-1to1

            merge_dicts(fact_dict['obs_dict'], batch)
            
            ### fact's noise
            u_fact = pgm.infer_exogeneous(batch) # only infer t,i exo noise
            merge_dicts(fact_dict['u_dict'], u_fact)
            ########fact
            


            ###### jdcfs
            ## pa's jdcf
            pa_jd = pgm.counterfactual(obs=batch, intervention=interventions)
            merge_dicts(jdcf_dict['obs_dict'], pa_jd)
            # print(pa_jd.keys())
            u_jd_pa = pgm.infer_exogeneous(pa_jd) # 只能infer出t,i exo noise
            merge_dicts(jdcf_dict['u_dict'], u_jd_pa)
            # print(u_jd_pa.keys())

            if len(do_c_var)!=0: # if do on intensity
                u_ours_init=u_jd_pa
                our_param = []
                # Originally within support, directly assign it to initialization; others assign a value of 0 to t.
                def index_not_tail_in_batch(u_jd_pa):
                    s_to_c =set()
                    result_list = [1] * num_batch
                    for k in do_c_var:
                        s_to_c.add(k)
                        for j in cond_back_dict[k]:
                            s_to_c.add(j)
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
                    for j in cond_back_dict[k]:
                        para_init[j+'_base']=torch.zeros_like(u_jd_pa[j+'_base'])
                        para_init[j+'_base'][index_not_tail0==1]=u_jd_pa[j+'_base'][index_not_tail0==1]
                ####################

                for k in do_c_var:
                    for j in cond_back_dict[k]:
                    # u_ours_init[cond_dict[k]+'_base'] = nn.Parameter(torch.zeros(num_batch).reshape(-1, 1).cuda())
                        u_ours_init[j+'_base'] = nn.Parameter(para_init[j+'_base'].cuda())
                        our_param.append(u_ours_init[j+'_base'])
                
                # Number of training iterations for ourscf
                print('ourscf training iter', args.epoch_num)

                print('lr',args.cf_lr)
                print("sch_step_size", args.lr_step_size)
                print("sch_gamma", args.lr_gamma)
                optimizer = optim.SGD(our_param, lr=args.cf_lr)
                scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

                pa_ours={}
                pa_ours=pa_jd.copy()
        
                
                
                # n_exp=1
                for epoch in range(args.epoch_num):
                    
                    # break
                    optimizer.zero_grad()
                    
                    
                    
                    if "n3"in do_c_var:
                        pa_ours['n1'] = pgm.s_var_net['n1'+'_flow'](u_ours_init['n1'+'_base'])
                        n2_unnormal = pgm.c_var_net['n2'+'_flow'][0].condition(pa_ours['n1'])(u_ours_init['n2'+'_base'])
                        pa_ours['n2'] = pgm.c_var_net['n2'+'_flow'][1](n2_unnormal)
                        # back_list=['n1', 'n2', 'n3']
                    elif "n2"in do_c_var:
                        pa_ours['n1'] = pgm.s_var_net['n1'+'_flow'](u_ours_init['n1'+'_base']) 
                        # back_list=['n1', 'n2']
                    mix_pa={} # contain jdpa's c var, and others are pa_ours
                    for k in pa_ours.keys():
                        if k in do_c_var:
                            mix_pa[k]=pa_jd[k]
                        else:
                            mix_pa[k]=pa_ours[k]



                    temp_our_u = pgm.infer_exogeneous(mix_pa)

                    for k in do_c_var:
                        u_ours_init[k+'_base']=temp_our_u[k+'_base']
                    
                    # for k, v in u_ours_init.items():
                    #     print(k, type(v))

                    
                    temp_pa_ours=pgm.model_infer(u_ours_init)
                    for k in pa_ours.keys(): 
                        pa_ours[k]=temp_pa_ours[k]
                    
                    # 
                    loss_s={}
                    loss_s_avg=0
                    for k in do_c_var:
                        for j in cond_back_dict[k]:
                            loss_s[j]= \
                            torch.abs(gaussian_cdf(u_ours_init[j+'_base'])-gaussian_cdf(u_fact[j+'_base'].detach())).mean()
                            if k=='n3':
                                if j=='n2':
                                    loss_s_avg +=w_2*loss_s[j]
                                elif j=='n1':
                                    loss_s_avg +=w_1*loss_s[j]
                                else:
                                    NotImplementedError
                            elif k=='n2':
                                if j=='n1':
                                    loss_s_avg +=w_1*loss_s[j]
                                else:
                                    NotImplementedError
                            else:
                                NotImplementedError
                    
                    
                    loss_c={}
                    loss_c_avg=0
                    for k in do_c_var:
                        loss_c[k]= \
                        torch.abs(gaussian_cdf(u_ours_init[k+'_base'])-gaussian_cdf(u_fact[k+'_base'].detach())).mean()
                        if k=='n3':
                            loss_c_avg +=w_3*loss_c[k]
                        elif k=='n2':
                            loss_c_avg +=w_2*loss_c[k]
                        else:
                            NotImplementedError
                    
                    
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
                   
                    loss = loss_s_avg+loss_c_avg+w_n*loss_n_avg
                        
                    
                    # 
                    loss.backward(retain_graph=True)
                    
                    

                    
                    for param in our_param:
                        param.data = torch.clamp(param.data, lower_bound, upper_bound)

                    # # 
                    if epoch%100==0:
                        # print(epoch, ': ', loss.item(), loss_t_mean.item(), loss_i_mean.item(), loss_act_mean.item())
                        print(epoch, ': ', loss.item(), loss_s_avg.item(), loss_c_avg.item(), loss_act_avg.item(), loss_n_avg.item())
                        print('loss_s: '+" - ".join(f"{k}: {v}" for k, v in loss_s.items()))
                        print('loss_c: '+" - ".join(f"{k}: {v}" for k, v in loss_c.items()))
                        print('loss_act: '+" - ".join(f"{k}: {v}" for k, v in loss_act.items()))
                        print('loss_n: '+" - ".join(f"{k}: {v}" for k, v in loss_n.items()))
                    
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

                if "n3"in do_c_var:
                    pa_ours['n1'] = pgm.s_var_net['n1'+'_flow'](u_ours_init['n1'+'_base'])
                    n2_unnormal = pgm.c_var_net['n2'+'_flow'][0].condition(pa_ours['n1'])(u_ours_init['n2'+'_base'])
                    pa_ours['n2'] = pgm.c_var_net['n2'+'_flow'][1](n2_unnormal)
                    # back_list=['n1', 'n2', 'n3']
                elif "n2"in do_c_var:
                    pa_ours['n1'] = pgm.s_var_net['n1'+'_flow'](u_ours_init['n1'+'_base']) 
                    # back_list=['n1', 'n2']
                mix_pa={} # contain jdpa's c var, and others are pa_ours
                for k in pa_ours.keys():
                    if k in do_c_var:
                        mix_pa[k]=pa_jd[k]
                    else:
                        mix_pa[k]=pa_ours[k]



                temp_our_u = pgm.infer_exogeneous(mix_pa)

                for k in do_c_var:
                    u_ours_init[k+'_base']=temp_our_u[k+'_base']
                
                # for k, v in u_ours_init.items():
                #     print(k, type(v))

                
                temp_pa_ours=pgm.model_infer(u_ours_init)
                for k in pa_ours.keys(): 
                    pa_ours[k]=temp_pa_ours[k]
            else: # thickness
                pass
                        

            if len(do_c_var)!=0:
                merge_dicts(ourscf_dict['obs_dict'], pa_ours)
                merge_dicts(ourscf_dict['u_dict'], u_ours_init)        
            
            def add_rest_dict(pa_cf, preds, cf_type):
                # add batch information to preds

                for k in preds.keys():
                    preds[k].extend(copy.deepcopy(pa_cf[k].detach().cpu()))
                

            
            if len(do_c_var)!=0:
                add_rest_dict(pa_ours, preds_ours, 'ours')
            add_rest_dict(pa_jd, preds_jd, 'jd')
            

        def index_not_tail(data_dict):
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
        def get_noise(fact_dict):
            # fact_dict
            data_=fact_dict['obs_dict']
            max_list =[v.numpy() for v in dataloader.dataset.max_list]
            min_list =[v.numpy() for v in dataloader.dataset.min_list]
            data={}
            for k, v in data_.items():
                num=int(re.findall(r'\d+', k)[0])
                # print(num, max_list, min_list)
                range_vals =  max_list[num-1]-min_list[num-1]
                # 
                data[k] = (data_[k] + 1) / 2 * range_vals + min_list[num-1]
            n4=data['n4']
            n3=data['n3']
            n2=data['n2']
            n1=data['n1']
            e2=(n2+n1)*3.0
            e3=(n3-np.sin(np.pi*0.1*(n2+2.0)))*5.0
            e4 =(n4-np.sin(np.pi*0.25*(n3-n1+2)))*5.0
            return e2, e3, e4

        def get_n4(e4, n1, n2, n3):
            return np.sin(np.pi*0.25*(n3-n1+2))+e4/5.0

        def get_n3(e3, n1, n2):
            return np.sin(np.pi*0.1*(n2+2.0))+e3/5.0

        def get_n2(e2, n1):
            return -n1+e2/3.0  
        
        import re
        def stats_func(preds, targets, result_dict, index_not_tail, mix_index):
            # index of not_x_dict
            max_list =[v.numpy() for v in dataloader.dataset.max_list]
            min_list =[v.numpy() for v in dataloader.dataset.min_list]
            for k, v in preds.items():
                preds[k] = torch.stack(v).squeeze().numpy()
                num=int(re.findall(r'\d+', k)[0])
                # print(num, max_list, min_list)
                range_vals =  max_list[num-1]-min_list[num-1]
                # 
                preds[k] = (preds[k] + 1) / 2 * range_vals + min_list[num-1] 
                if 'n1' in var_name:
                    if k=='n1':
                        targets[k]=preds[k]
                elif 'n2' in var_name:
                    if k in ['n1', 'n2']:
                        targets[k]=preds[k]
                elif 'n3' in var_name:
                    if k in ['n1', 'n2', 'n3']:
                        targets[k]=preds[k]
                else:
                    NotImplementedError
            
            gt_e2, gt_e3, gt_e4=get_noise(fact_dict)
            if 'n1' in var_name:
                targets['n2'] = get_n2(gt_e2.reshape(-1), preds['n1'])
                targets['n3'] = get_n3(gt_e3.reshape(-1), preds['n1'], targets['n2'])
                targets['n4'] = get_n4(gt_e4.reshape(-1), preds['n1'], targets['n2'], targets['n3'])
            elif 'n2' in var_name:
                targets['n3'] = get_n3(gt_e3.reshape(-1), preds['n1'], preds['n2'])
                targets['n4'] = get_n4(gt_e4.reshape(-1), preds['n1'], preds['n2'], targets['n3'])
            elif 'n3' in var_name:
                targets['n4'] = get_n4(gt_e4.reshape(-1), preds['n1'], preds['n2'], preds['n3'])
            
                
            
            for k, v in targets.items():
                print('check', k, v.shape)

            # for k in targets.keys():
            #     print('target', k, targets[k].shape)
            #     print('pred', k, preds[k].shape)
            
            
            result_dict['targets_dict']=targets 
            result_dict['preds_dict']=preds

            stats = {}
            if args.dataset == "toy":
                for k in pgm.variables.keys():

                # continuous variables
                    # unormalize from [-1,1] back to original range

                    mae= np.abs(targets[k] - preds[k])
                    # print('mae_max', mae.max())
                    result_dict['total_mean_dict'][k]=  mae.mean()  
                    stats[k + "_total_mae"] = result_dict['total_mean_dict'][k]
                    # print('mae.shape', mae.shape, mae, index_not_tail)
                    result_dict['not_tail_mean_dict'][k]=  mae[index_not_tail==1].mean()  
                    stats[k + "_not_tail_mae"] = result_dict['not_tail_mean_dict'][k]

                    result_dict['tail_mean_dict'][k]=  mae[index_not_tail==0].mean()   
                    stats[k + "_tail_mae"] = result_dict['tail_mean_dict'][k]
                    
                    if len(do_c_var)!=0:
                        result_dict['mix_mean_dict'][k]= []
                        for i in range(4):
                            stats[k + "_mix"+str(i)+"_mae"]=mae[mix_index[i]==1].mean()
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
    
    
    var_list={0:{'n1'}, 1:{'n2'}, 2:{'n3'}} #"thickness", 
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
        "--logw_s1", help="far wt.", type=float, default=1
    )
    parser.add_argument(
        "--logw_s0", help="closer wt.", type=float, default=0
    )

    parser.add_argument(
        "--logw_c", help="wt.", type=float, default=0
    )
    parser.add_argument(
        "--logw_n", help="wn.", type=float, default=13
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
    

    # Load PGM
    print(f"\nLoading PGM checkpoint: {args.pgm_path}")
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint["hparams"])
    pgm_args.bs=args.bs
    pgm = TOYPGM(pgm_args).cuda()
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

    

    # init model
    
    model =pgm
    ema = EMA(model, beta=args.ema_rate)
    model.cuda()
    ema.cuda()

    # setup data
    pgm_args.concat_pa = False
    pgm_args.bs = args.bs
    pgm_args.setup="sup_pgm"
    from pgm.train_pgm import setup_dataloaders

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

    # for ks in stats_var_dict.keys():
    #         if len(stats_var_dict[ks])==2:
    #             [ours, jd]=stats_var_dict[ks]
    #             print(str(ks)+ " ours: test | " + " - ".join(f"{k}: {v:.8f}" for k, v in ours.items()))
    #             print(str(ks)+ " jd: test | " + " - ".join(f"{k}: {v:.8f}" for k, v in jd.items()))
    #         else:
    #             print(str(ks)+ ": test | " + " - ".join(f"{k}: {v:.8f}" for k, v in stats_var_dict[ks].items()))


