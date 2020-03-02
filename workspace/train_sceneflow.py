import sys
sys.path.insert(0, "..")

import numpy as np
from tqdm import tqdm
import os

import time

import matplotlib.pyplot as plt
# PIL
from PIL import Image

# TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2

# Utils package
import utils.networks as networks
from utils.dataset import DatasetLoader
from utils.dataset import co_transforms
from utils.dataset import transforms

import argparse

import re


# load the filenames for training
def load_filenames_train(path_data, path_target, data_dir=None, target_dir=None):
    if target_dir is None:
        target_dir = data_dir

    filenames_data = open(path_data, "r").readlines()
    filenames_target = open(path_target, "r").readlines()
    assert len(filenames_data) == len(filenames_target)

    if data_dir is not None:
        filenames = [[
                [ os.path.join(data_dir, g.split("\n")[0]) for g in f[0].split(" ")],
                [ os.path.join(target_dir, g.split("\n")[0]) for g in f[1].split(" ")],
                ] for f in zip(filenames_data,filenames_target)]
    else:
        filenames = [[
            [  g.split("\n")[0] for g in f[0].split(" ")],
            [  g.split("\n")[0] for g in f[1].split(" ")],
            ] for f in zip(filenames_data,filenames_target)]
    return filenames

def load_filenames_train(path_data, path_target, data_dir=None, target_dir=None):
    if target_dir is None:
        target_dir = data_dir

    filenames_data = open(path_data, "r").readlines()
    filenames_target = open(path_target, "r").readlines()
    assert len(filenames_data) == len(filenames_target)

    if data_dir is not None:
        filenames = [[
                [ os.path.join(data_dir, g.split("\n")[0]) for g in f[0].split(" ")],
                [ os.path.join(target_dir, g.split("\n")[0]) for g in f[1].split(" ")],
                ] for f in zip(filenames_data,filenames_target)]
    else:
        filenames = [[
            [  g.split("\n")[0] for g in f[0].split(" ")],
            [  g.split("\n")[0] for g in f[1].split(" ")],
            ] for f in zip(filenames_data,filenames_target)]
    return filenames
    
load_filenames_test =  load_filenames_train

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def input_rgb_image_loader(data_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """

    # Load .png RGB image
    im = np.array(Image.open(data_path)).astype(np.float32)

    # Resize
    size = (512,256)
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # Normalize between 0 - +1
    im /= 255.

    return im


def input_sgm(data_path):

    # Load disp map
    im = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH)

    # Normalize between -1 - +1
    global mean_disp
    global std_disp

    h,w = im.shape

    # Resize 
    size = (512,256)
    im = (512./w) * cv2.resize(im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    im[im < 0.] = -1.

    im = (im - mean_disp) / std_disp

    im = im.reshape(im.shape+(1,))

    return im


def train_target_gt_depth_loader(data_path):

    # Load disp map
    im = readPFM(data_path)[0]

    if len(im[im < 0]) > 0:
        im *= -1.

    # Do not consider images with less than
    # 10% of valid pixels
    if len(im[im < 192.]) < 0.1 * len(im):
        im += 10000.

    # Normalize between -1 - +1
    global mean_disp
    global std_disp

    h,w = im.shape

    # Resize 
    size = (512,256)
    im = (512./w) * cv2.resize(im, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

    im = (im - mean_disp) / std_disp

    im = im.reshape(im.shape+(1,))

    return im


def test_target_gt_depth_loader(data_path):

    # Load disp map
    im = readPFM(data_path)[0]

    if len(im[im < 0]) > 0:
        im *= -1.
        
    # Do not consider images with less than
    # 10% of valid pixels
    if len(im[im < 192.]) < 0.1 * len(im):
        im += 10000.

    h,w = im.shape

    im = im.reshape(im.shape+(1,))

    return im

def predict(mode, stereo, refine, fusion, input_data):

    if mode == 0:
        return stereo.forward(input_data)

    if mode == 1:
        return refine.forward(input_data)
    
    if mode == 2:
        output_disp = stereo.forward(input_data[1:])
        
        first_input = torch.cat((input_data),1)
        second_input = torch.cat((first_input, output_disp),1)
        
        return fusion.forward([second_input])


def compute_epe(gt, pred):

    epe = np.mean(np.abs(gt - pred))

    return epe

def compute_errors(gt, pred):

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--savedir", type=str, default="results")
    parser.add_argument("--datadir", type=str, default=".")

    parser.add_argument("--test_only", type=int, default=0)

    parser.add_argument("--mode", type=int, default=-1)
    parser.add_argument("--method", type=int, default=-1)

    parser.add_argument("--train_inputs_file", type=str, default="")
    parser.add_argument("--train_targets_file", type=str, default="")

    parser.add_argument("--test_inputs_file", type=str, default="")
    parser.add_argument("--test_targets_file", type=str, default="")

    parser.add_argument("--state_dict_stereo_block", type=str, default="")
    parser.add_argument("--state_dict_refine_block", type=str, default="")
    parser.add_argument("--state_dict_fusion_block", type=str, default="")

    parser.add_argument("--optim_dict_stereo_block", type=str, default="")
    parser.add_argument("--optim_dict_refine_block", type=str, default="")
    parser.add_argument("--optim_dict_fusion_block", type=str, default="")

    parser.add_argument("--nb_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_start", type=float, default=0.001)

    parser.add_argument("--use_cuda", type=int, default=1)

    parser.add_argument("--nb_filters", type=int, default=128)

    parser.add_argument("--sched", type=str, default="multi")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)

    args = parser.parse_args()

    if args.mode < 0 or args.mode > 2:
        print("\nPlease provide a mode value (--mode X) !")
        print("Possible values are: 0 (stereo-only) / 1 (refine only) / 2 (fd-fusion)")
        exit(-1)

    if args.mode > 0:
        if args.method < 0 or args.method > 2:
            print("\nPlease provide a method value (--method X) !")
            print("Possible values are: 0 (sgbm) / 1 (elas) / 2 (sgm)")
            exit(-1)


    # Train files / Test files
    train_filenames = load_filenames_train(
            path_data = args.train_inputs_file,
            path_target = args.train_targets_file,
            data_dir = args.datadir)

    test_filenames = load_filenames_test(
            path_data = args.test_inputs_file,
            path_target = args.test_targets_file,
            data_dir = args.datadir)

    ###############
    ## PARAMETERS
    ###############
    nbr_epochs = args.nb_epochs # training epoch
    milestones = [5,10,15,20,22,24,26,28] # steps for decreasing learning rate

    batch_size = args.batch_size # batch size

    use_cuda = args.use_cuda # use cuda GPU
    save_dir = args.savedir # path the directory where to save the model and results

    lr_start = args.lr_start

    ##################################
    # NORMALIZATION SPECIFIC PARAMS
    ##################################
    global mean_disp
    global std_disp
    global min_valid_disp

    mean_disp = 41.6
    std_disp = 15.83

    img_w = 960.
    img_h = 540.

    mean_disp *= 512. / img_w
    std_disp *= 512. / img_w

    max_disp = 192.
    min_disp = 0.

    # Min Disp in GT: 4.1
    min_valid_disp = -1. * mean_disp / std_disp

    max_valid_disp = max_disp * 512. / img_w
    max_valid_disp = (max_valid_disp - mean_disp) / std_disp

    # Values of non-annotated pixels in GT
    zero_disp = -1. * (mean_disp / std_disp)

    global d1_sgbm_train
    d1_sgbm_train = 0.
    global epe_sgbm_train
    epe_sgbm_train = 0.
    global d1_sgbm_val_train
    d1_sgbm_val_train = 0.
    global epe_sgbm_val_train
    epe_sgbm_val_train = 0.

    global d1_sgbm_test
    d1_sgbm_test = 0.
    global epe_sgbm_test
    epe_sgbm_test = 0.
    global d1_sgbm_val_test
    d1_sgbm_val_test = 0.
    global epe_sgbm_val_test
    epe_sgbm_val_test = 0.

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print("Creating data loaders...", end="", flush=True)

    # training data transforms for augmentation
    # train_co_transforms = co_transforms.Compose([co_transforms.RandomColorJitter(0.5,0.5,0.5,0.35,0.5),co_transforms.RandomHorizontalFlip(),])
    train_co_transforms = co_transforms.Compose([co_transforms.RandomHorizontalFlip(),])

    input_transforms = [transforms.ToTensor(torch.FloatTensor), transforms.ToTensor(torch.FloatTensor), transforms.ToTensor(torch.FloatTensor)]
    target_transforms = [transforms.ToTensor(torch.FloatTensor), transforms.ToTensor(torch.LongTensor)]

    im_loader = [input_rgb_image_loader, input_rgb_image_loader]

    if args.mode > 0:
        if args.method == 0:
            im_loader = [input_sgbm, input_rgb_image_loader, input_rgb_image_loader]
        if args.method == 1:
            im_loader = [input_elas, input_rgb_image_loader, input_rgb_image_loader]
        if args.method == 2:
            im_loader = [input_sgm, input_rgb_image_loader, input_rgb_image_loader]


    train_dataset = DatasetLoader(
                    filelist=train_filenames,
                    image_loader=im_loader,
                    target_loader=[train_target_gt_depth_loader],
                    training=True,
                    co_transforms=train_co_transforms,
                    input_transforms=input_transforms, 
                    target_transforms= target_transforms,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

    test_dataset = DatasetLoader(
                    filelist=test_filenames,
                    image_loader=im_loader,
                    target_loader=[test_target_gt_depth_loader],
                    training=False,
                    co_transforms=None,
                    input_transforms=input_transforms, 
                    target_transforms= target_transforms,
                    return_filenames=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0) 

    print("Done")

    print("Creating the model...", end="", flush=True)

    net_stereo = networks.DilNetLRDisp(args.nb_filters, 6,1)
    net_refine = networks.DilNetLRDisp(args.nb_filters, 7,1)
    net_fusion = networks.DilNetLRDisp(args.nb_filters, 8,1)

    if use_cuda:
        net_stereo.cuda()
        net_refine.cuda()
        net_fusion.cuda()

    print("done")

    print("Creating optimizer...", end="\n\n", flush=True)

    optimizer = torch.optim.Adam(net_stereo.parameters(), lr=lr_start)

    if args.mode == 0:
        net_stereo.train()
        optimizer = torch.optim.Adam(net_stereo.parameters(), lr=lr_start)
    
    if args.mode == 1:
        net_refine.train()
        optimizer = torch.optim.Adam(net_refine.parameters(), lr=lr_start)
    
    if args.mode == 2:
        net_stereo.eval()
        net_fusion.train()
        optimizer = torch.optim.Adam(net_fusion.parameters(), lr=lr_start)
    
    if args.sched == "multi":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
        print("Multistones scheduler selected with milestones / gamma : " + str(milestones) + " / " + str(args.gamma) + "\n")
    elif args.sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        print("StepLR scheduler selected with step / gamma : " + str(args.step) + " / " + str(args.gamma) + "\n")
    elif args.sched == "expo":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        print("Exponential scheduler selected with gamma : " + str(args.gamma) + "\n")
    else:
        print("Please select a scheduler: multi / expo")
        exit(-1)

    print("done")

    # State Dict Loaders
    ####################
    if args.state_dict_stereo_block != "":
        print("Loading pre-trained stereo model...", end="", flush=True)
        if use_cuda:
            net_stereo.load_state_dict(torch.load(args.state_dict_stereo_block))
        else:
            device = torch.device('cpu')
            net_stereo.load_state_dict(torch.load(args.state_dict_stereo_block, map_location=device))
        print("done")

    if args.state_dict_refine_block != "":
        print("Loading pre-trained fusion models...", end="", flush=True)
        if use_cuda:
            net_refine.load_state_dict(torch.load(args.state_dict_refine_block))
        else:
            device = torch.device('cpu')
            net_refine.load_state_dict(torch.load(args.state_dict_refine_block, map_location=device))
        print("done")

    if args.state_dict_fusion_block != "":
        print("Loading pre-trained fusion models...", end="", flush=True)
        if use_cuda:
            net_fusion.load_state_dict(torch.load(args.state_dict_fusion_block))
        else:
            device = torch.device('cpu')
            net_fusion.load_state_dict(torch.load(args.state_dict_fusion_block, map_location=device))
        print("done")

    # Optimizer Loaders
    ####################
    if args.optim_dict_stereo_block != "":
        print("Loading pre-trained optimizer...", end="", flush=True)
        if use_cuda:
            optimizer.load_state_dict(torch.load(args.optim_dict_stereo_block))
        else:
            device = torch.device('cpu')
            optimizer.load_state_dict(torch.load(args.optim_dict_stereo_block, map_location=device))
        print("done")

    if args.optim_dict_refine_block != "":
        print("Loading pre-trained optimizer...", end="", flush=True)
        if use_cuda:
            optimizer.load_state_dict(torch.load(args.optim_dict_refine_block))
        else:
            device = torch.device('cpu')
            optimizer.load_state_dict(torch.load(args.optim_dict_refine_block, map_location=device))
        print("done")

    if args.optim_dict_fusion_block != "":
        print("Loading pre-trained optimizer...", end="", flush=True)
        if use_cuda:
            optimizer.load_state_dict(torch.load(args.optim_dict_fusion_block))
        else:
            device = torch.device('cpu')
            optimizer.load_state_dict(torch.load(args.optim_dict_fusion_block, map_location=device))
        print("done")


    def train(epoch):

        if args.mode == 0:
            net_stereo.train()
        
        if args.mode == 1:
            net_refine.train()
        
        if args.mode == 2:
            net_stereo.eval()
            net_fusion.train()

        total_loss_disp = 0

        img_counter = 0

        dt = 0.

        t = tqdm(train_loader, ncols=70, desc="Epoch "+str(epoch))
        for input_data, target_data in t:
            optimizer.zero_grad()

            if use_cuda:
                input_data = [in_data.cuda() for in_data in input_data]
                target_data = [ta_data.cuda() for ta_data in target_data]

            # Disp Map Prediction
            st = time.perf_counter()

            output_disp = predict(args.mode, net_stereo, net_refine, net_fusion, input_data)

            dt += time.perf_counter() - st
            
            # Target depth
            target = target_data[0]

            # We only compute error on non-zero pixels
            # mask = np.logical_and(target >= zero_disp, target <= max_valid_disp)
            mask = target <= max_valid_disp

            if len(target[mask]) == 0:
                continue

            # We only compute error on non-zero pixels (i.e. non 1 here)
            error_disp = F.l1_loss(output_disp[mask],target[mask])

            # loss
            total_loss_disp += error_disp.item()

            # Optimize the weights!
            error_disp.backward()
            optimizer.step()

            img_counter += batch_size

            t.set_postfix(loss="%.8f"%float(total_loss_disp / img_counter),
                        dt="%.2f" %(dt * 1000. / img_counter))

        return total_loss_disp



    def test(epoch):

        if args.mode == 0:
            net_stereo.eval()
        
        if args.mode == 1:
            net_refine.eval()
        
        if args.mode == 2:
            net_stereo.eval()
            net_fusion.eval()

        total_loss_disp = 0

        img_counter = 0
        
        dt = 0.

        d1_cnn = 0.

        with torch.no_grad():
            t = tqdm(test_loader, ncols=70, desc="  Test "+str(epoch))
            for input_data, target_data in t:

                if use_cuda:
                    input_data = [in_data.cuda() for in_data in input_data]
                    target_data = [ta_data.cuda() for ta_data in target_data]


                # Disp Map Prediction
                st = time.perf_counter()

                output_disp = predict(args.mode, net_stereo, net_refine, net_fusion, input_data)

                dt += time.perf_counter() - st

                # Target depth in normal depth
                target = target_data[0]

                # Get the predicted and targeted depth
                out_disp = output_disp.cpu().detach().numpy()
                tar = target.cpu().numpy()

                nb_img = out_disp.shape[0]
                img_counter += nb_img

                # For each image in the batch
                for i in range(nb_img):

                    # Get the GT disp
                    #######################
                    gt_disp = np.squeeze(tar[i], axis=0)

                    gt_h, gt_w = gt_disp.shape

                    # Get the Pred depth
                    #########################
                    pred_disp = np.squeeze(out_disp[i], axis=0)

                    # Resize 
                    size = (gt_w, gt_h)
                    pred_h, pred_w = pred_disp.shape

                    pred_disp *= std_disp
                    pred_disp += mean_disp
                    
                    pred_disp = (gt_w/pred_w) * cv2.resize(pred_disp,size,interpolation=cv2.INTER_LINEAR)

                    # Bound the predicted depth
                    pred_disp[pred_disp < min_disp] = min_disp
                    pred_disp[pred_disp > max_disp] = max_disp
                        
                    # D1-all
                    #########################################
                    mask = np.logical_and(gt_disp > 0., gt_disp <= max_disp)

                    epe = compute_epe(pred_disp[mask], gt_disp[mask])

                    d1_cnn += epe


                # display TQDM
                t.set_postfix(EPE="%.2f"%float(d1_cnn / img_counter),
                            dt="%.1f" %(dt * 1000. / img_counter))

            d1_cnn /= img_counter

            return d1_cnn

    if args.test_only == 1:
        d1 = test(0)

        print("\n\n >>> Final Results :")
        print(">>>>>> EPE : " + str(d1))

    else:

        os.makedirs(save_dir, exist_ok=True)

        best_epe = 1000

        net_name = str(net_stereo.__class__).split("'")[1].split(".")[-1]

        run_name = net_name + "_nb_filters_" + str(args.nb_filters) + "_mode_" + str(args.mode) + "_method_" + str(args.method) + "_batch_" + str(batch_size) + "_lr_" + str(lr_start)

        # write the logs
        log_name = ""

        if args.sched == "step":
            log_name = "logs_" + run_name + "_optimizer_" + optimizer.__module__ + "_sched_" + args.sched + "_stepsize_" + str(args.sched) + "_gamma_" + str(args.gamma) + ".txt"
        elif args.sched == "multi":
            log_name = "logs_" + run_name + "_optimizer_" + optimizer.__module__ + "_sched_" + args.sched + "_steps_" + str(milestones) + "_gamma_" + str(args.gamma) + ".txt"
        else:
            log_name = "logs_" + run_name + "_optimizer_" + optimizer.__module__ + "_sched_" + args.sched + "_gamma_" + str(args.gamma) + ".txt"

        f = open(os.path.join(save_dir, log_name), "w")

        f.write("# Network : " + net_name +" - Optimizer : " + optimizer.__module__ + " - Scheduler : " + scheduler.__module__
                + " - Batch size : " + str(batch_size) + " - Milestones : " + str(milestones) + " - Nb Epochs : " + str(nbr_epochs) + " - lr : " 
                + str(lr_start) + "\n\n")

        f.write("Epoch # - LR - (Train) loss")
        f.write(" - (Test) EPE\n")

        for epoch in range(nbr_epochs):

            train_loss_disp = train(epoch)

            epe  = test(epoch)

            scheduler.step()

            lr_val = 0.

            for params in optimizer.param_groups:
                lr_val = params['lr']

            if epe < best_epe:

                best_epe = epe

                state_dict_name = "state_dict_sceneflow_" + log_name[:-4] + ".pth"
                optim_dict_name = "optim_dict_sceneflow_" + log_name[:-4] + ".pth"

                if args.mode == 0:
                    torch.save(net_stereo.state_dict(), os.path.join(save_dir, state_dict_name))
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, optim_dict_name))
                
                if args.mode == 1:
                    torch.save(net_refine.state_dict(), os.path.join(save_dir, state_dict_name))
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, optim_dict_name))
                
                if args.mode == 2:
                    torch.save(net_fusion.state_dict(), os.path.join(save_dir, state_dict_name))
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, optim_dict_name))

            # write the logs
            f.write(str(epoch)+" ")
            f.write("%.7f "%lr_val)
            f.write("%.3f "%train_loss_disp)
            f.write("%.3f "%epe)
            f.write("\n")
            f.flush()

        f.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
