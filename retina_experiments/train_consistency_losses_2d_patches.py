import sys, json, os, time, argparse
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from utils.model_factory import get_model
from utils.loss_factory import get_loss
from utils.reproducibility import set_seeds
from utils.metric_factory import fast_bin_dice, fast_bin_auc, cl_dice_metric

from monai.inferers import sliding_window_inference
from utils.data_load_2d import get_loaders

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')


# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='FIVES', help='dataset, should be directory name')
parser.add_argument('--cache', type=float, default=0., help='percentage of precomputed and cached data for loading')
parser.add_argument('--tr_percentage', type=float, default=1., help='amount of training data to use')
parser.add_argument('--n_classes', type=int, default=2, help='2=binary, >2=multi_class')
parser.add_argument('--model_name', type=str, default='small_unet_2d', help='architecture')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained weights if available')
parser.add_argument('--loss1', type=str, default='ce',   choices=('ce', 'dice', 'cedice'), help='1st loss')
parser.add_argument('--loss2', type=str, default=None, choices=(None, 'ce', 'dice', 'cldice', 'clce'), help='2nd loss')
parser.add_argument('--alpha1', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
parser.add_argument('--alpha2', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
parser.add_argument('--gt_skeleton', type=str, default='online', choices=('offline', 'online'), help='if offline, skeletonize gt volumes and cache)')
parser.add_argument('--batch_size', type=int, default=4, help='patches are fed to model in groups of batch_size')
parser.add_argument('--n_samples', type=int, default=12, help='nr of patches extracted per loaded volume before moving to the next one')
parser.add_argument('--neg_samples', type=int, default=3, help='out of n patches, 1/(1+neg_samples) are foreground-centered')
parser.add_argument('--ovft_check', type=int, default=4, help='part of the training set used in validation, 0=all of it, can be sloooow')
parser.add_argument('--patch_size', type=str, default='512/512', help='patch sampling')
parser.add_argument('--normalization', type=str, default='windowing_norm', help='normalization choice, default is just normalization to (0,1)')
parser.add_argument('--optimizer', type=str, default='nadam', choices=('sgd', 'adamw', 'nadam'), help='optimizer choice')
parser.add_argument('--lr', type=float, default=1e-3, help='max learning rate')
parser.add_argument('--n_epochs', type=int, default=150, help='training epochs')
parser.add_argument('--vl_interval', type=int, default=50, help='how often we check performance and maybe save')
parser.add_argument('--cyclical_lr', type=str2bool, nargs='?', const=True, default=False, help='re-start lr each vl_interval epochs')
parser.add_argument('--metric', type=str, default='DSC', help='which metric to use for monitoring progress (DSC/cl-DSC/AUC)')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=12, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: 8]')




def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def validate(model, loader, loss_fn, slwin_bs=2):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = model.patch_size
    # patch_size, slwin_bs = model.patch_size, 1
    dscs, cl_dscs, aucs, losses = [], [], [], []

    with trange(len(loader)) as t:
        n_elems, running_dsc = 0, 0
        for val_data in loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"]
            val_outputs = sliding_window_inference(val_images.to(device), patch_size, slwin_bs, model, mode='gaussian').cpu()
            # val_outputs = sliding_window_inference(val_images.to(device), patch_size, slwin_bs, model, overlap=0).cpu()
            loss = loss_fn(val_outputs, val_labels)
            probs = val_outputs.softmax(dim=1)[:, 1].squeeze().numpy()  # index on class 1
            preds = val_outputs.argmax(dim=1).squeeze().numpy().astype(bool)
            del val_outputs
            labels = val_labels[:, 1].squeeze().numpy().astype(bool)  # index on class 1
            dsc_score = fast_bin_dice(labels, preds)
            auc_score = fast_bin_auc(labels, probs, partial=True)
            cl_dsc_score = cl_dice_metric(preds, labels)

            if np.isnan(dsc_score): dsc_score = 0
            if np.isnan(cl_dsc_score): cl_dsc_score = 0
            dscs.append(dsc_score)
            cl_dscs.append(cl_dsc_score)
            aucs.append(auc_score)
            losses.append(loss.item())
            n_elems += 1
            running_dsc += np.mean(dsc_score)
            run_dsc = running_dsc / n_elems
            t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            t.update()

    return [100 * np.mean(np.array(dscs)), 100 * np.mean(np.array(cl_dscs)), 100 * np.mean(np.array(aucs)), np.mean(np.array(losses))]

def train_one_epoch(model, bs, tr_loader, loss_fn, optimizer, scheduler):
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    with trange(len(tr_loader)) as t:
        step, n_elems, running_loss = 0, 0, 0
        for batch_data in tr_loader:  # load 1 scan from the training set
            n_samples = len(batch_data['seg'])  # nr of px x py x pz patches (see args.n_samples)
            for m in range(0, n_samples, bs):  # we loop over batch_data picking up bs patches at a time
                step += bs
                inputs, labels = (batch_data['img'][m:(m+bs)].to(device), batch_data['seg'][m:(m+bs)].to(device))
                try: label_skels = batch_data['seg_sk'][m:(m+bs)].to(device)
                except: label_skels = None
                outputs = model(inputs)
                loss = loss_fn(outputs, labels, label_skels)
                loss.backward()
                optimizer.step()
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item() * inputs.shape[0]
                n_elems += inputs.shape[0]  # total nr of items processed
                run_loss = running_loss / n_elems

            t.set_postfix(LOSS_lr="{:.4f}/{:.6f}".format(run_loss, lr))
            t.update()

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    # I customize this for each project.
    # Here tr_info contains Dice Scores, Cl-Dice Scores, AUCs, and loss values.
    # Also, and vl_metrics contain (in this order) dice, cl-dice, auc and loss
    if best_epoch:
        tr_info['best_tr_dsc'] = tr_info['tr_dscs'][-1]
        tr_info['best_vl_dsc'] = tr_info['vl_dscs'][-1]
        tr_info['best_tr_cldsc'] = tr_info['tr_cldscs'][-1]
        tr_info['best_vl_cldsc'] = tr_info['vl_cldscs'][-1]
        tr_info['best_tr_auc'] = tr_info['tr_aucs'][-1]
        tr_info['best_vl_auc'] = tr_info['vl_aucs'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        tr_info['tr_dscs'].append(ovft_metrics[0])
        tr_info['vl_dscs'].append(vl_metrics[0])
        tr_info['tr_cldscs'].append(ovft_metrics[1])
        tr_info['vl_cldscs'].append(vl_metrics[1])
        tr_info['tr_aucs'].append(ovft_metrics[2])
        tr_info['vl_aucs'].append(vl_metrics[2])
        tr_info['tr_losses'].append(ovft_metrics[3])
        tr_info['vl_losses'].append(vl_metrics[3])

    return tr_info

def init_tr_info():
    # I customize this function for each project.
    tr_info = dict()
    tr_info['tr_dscs'], tr_info['tr_cldscs'] = [], []
    tr_info['tr_aucs'], tr_info['tr_losses'] = [], []
    tr_info['vl_dscs'], tr_info['vl_cldscs'] = [], []
    tr_info['vl_aucs'], tr_info['vl_losses'] = [], []

    return tr_info

def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    # I customize this function for each project.
    # Pretty prints first three values of train/val metrics to a string and returns it
    # Used also by the end of training (finished=True)
    ep_idx = len(tr_info['tr_dscs'])-1
    if finished:
        ep_idx = epoch
        epoch = (epoch+1) * vl_interval - 1

    s = 'Ep. {}: Train||Val DSC: {:5.2f}||{:5.2f} - cl-DSC: {:5.2f}||{:5.2f} - AUC: {:5.2f}||{:5.2f} - Loss: {:.4f}||{:.4f}'.format(
        str(epoch+1).zfill(3), tr_info['tr_dscs'][ep_idx], tr_info['vl_dscs'][ep_idx], tr_info['tr_cldscs'][ep_idx],
              tr_info['vl_cldscs'][ep_idx], tr_info['tr_aucs'][ep_idx], tr_info['vl_aucs'][ep_idx],
              tr_info['tr_losses'][ep_idx], tr_info['vl_losses'][ep_idx])
    return s

def train_model(model, optimizer, loss_fn, bs, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path):
    best_metric, best_epoch = -1, 0
    tr_info = init_tr_info()
    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        # train one cycle
        train_one_epoch(model, bs, tr_loader, loss_fn, optimizer, scheduler)
        if (epoch + 1) % vl_interval == 0:
            with torch.no_grad():
                ovft_metrics = validate(model, ovft_loader, loss_fn)
                vl_metrics = validate(model, vl_loader, loss_fn)

            tr_info = set_tr_info(tr_info, epoch, ovft_metrics, vl_metrics)
            s = get_eval_string(tr_info, epoch)
            print(s)
            with open(osp.join(save_path, 'train_log.txt'), 'a') as f: print(s, file=f)
            # check if performance was better than anyone before and checkpoint if so
            if metric =='DSC': curr_metric = tr_info['vl_dscs'][-1]
            elif metric == 'cl-DSC': curr_metric = tr_info['vl_cldscs'][-1]
            elif metric == 'AUC': curr_metric = tr_info['vl_aucs'][-1]

            if curr_metric > best_metric:
                print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, best_metric, curr_metric))
                torch.save(model.state_dict(), osp.join(save_path, 'best_model.pth'))
                best_metric, best_epoch = curr_metric, epoch + 1
                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, best_metric, best_epoch))
    torch.save(model.state_dict(), osp.join(save_path, 'last_model.pth'))
    del model, tr_loader, vl_loader
    # maybe this works also? tr_loader.dataset._fill_cache
    torch.cuda.empty_cache()
    return tr_info



if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # logging
    if args.save_path == 'date_time': save_path = osp.join('experiments', datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    else: save_path = osp.join('experiments', args.save_path)
    os.makedirs(save_path, exist_ok=True)
    config_file_path = osp.join(save_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    # delete log files if they exist
    if osp.isfile(osp.join(save_path, 'train_log.txt')): open(osp.join(save_path, 'train_log.txt'), 'w').close()
    if osp.isfile(osp.join(save_path, 'log.txt')): open(osp.join(save_path, 'log.txt'), 'w').close()


    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    lr, bs, ns, negs = args.lr, args.batch_size, args.n_samples, args.neg_samples
    n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    ds, nw = args.dataset, args.num_workers,

    patch_size = args.patch_size.split('/')
    patch_size = tuple(map(int, patch_size))

    print('* Instantiating a {} model'.format(model_name))
    model = get_model(args.model_name, n_classes=args.n_classes, in_c=3, pretrained=args.pretrained, patch_size=patch_size)


    print('* Creating Dataloaders, batch size = {}, samples/vol = {}, workers = {}'.format(bs, ns, nw))
    tr_loader, ovft_loader, vl_loader, test_loader = get_loaders(dataset=ds, n_samples=ns, neg_samples=negs,
                                                                 n_classes=args.n_classes, patch_size=patch_size,
                                                                 cache=args.cache, num_workers=nw,
                                                                 tr_percentage=args.tr_percentage, ovft_check=args.ovft_check)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        sys.exit('please choose between sgd, adam or nadam optimizers')

    if args.cyclical_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=vl_interval*len(tr_loader)*ns//bs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs*len(tr_loader)*ns//bs, eta_min=0)

    if args.loss2 is None: args.alpha2 = 0
    loss_fn = get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2)

    print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(args.alpha1, args.loss1, args.alpha2, args.loss2))
    print('* Starting to train\n', '-' * 10)
    start = time.time()
    tr_info = train_model(model, optimizer, loss_fn, bs, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    with (open(osp.join(save_path, 'log.txt'), 'a') as f):
        print('Best epoch = {}/{}: Tr/Vl DSC = {:.2f}/{:.2f} - Tr/Vl cl-DSC = {:.2f}/{:.2f} - '
              'Tr/Vl AUC = {:.2f}/{:.2f} - Tr/Vl LOSS = {:.4f}/{:.4f}\n'.format(tr_info['best_epoch'], n_epochs,
                tr_info['best_tr_dsc'], tr_info['best_vl_dsc'], tr_info['best_tr_cldsc'], tr_info['best_vl_cldsc'],
                tr_info['best_tr_auc'], tr_info['best_vl_auc'], tr_info['best_tr_loss'], tr_info['best_vl_loss']), file=f)
        for j in range(n_epochs//vl_interval):
            s = get_eval_string(tr_info, epoch=j, finished=True, vl_interval=vl_interval)
            print(s, file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    with torch.no_grad():
        model.load_state_dict(torch.load(osp.join(save_path, 'best_model.pth')))
        test_metrics = validate(model, test_loader, loss_fn)

    s = 'Test DSC: {:5.2f} - cl-DSC: {:5.2f} - AUC: {:5.2f} - LOSS: {:7.4f}'.format(
        test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3])
    with (open(osp.join(save_path, 'log.txt'), 'a') as f):
        print(80 * '-', file=f)
        print(s, file=f)
    print(s)
    #
    # if tr_info['best_epoch'] !=  n_epochs:
    #     with torch.no_grad():
    #         model.load_state_dict(torch.load(osp.join(save_path, 'last_model.pth')))
    #         test_metrics = validate(model, test_loader, loss_fn)
    #
    # s = '[LAST] Test DSC: {:5.2f} - cl-DSC: {:5.2f} - AUC: {:5.2f} - LOSS: {:7.4f}'.format(
    #     test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3])
    # with (open(osp.join(save_path, 'log.txt'), 'a') as f):
    #     print(80 * '-', file=f)
    #     print(s, file=f)
    # print(s)

    print('Finished.')

