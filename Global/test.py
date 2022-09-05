# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2

def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"


if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)

    model = Pix2PixHDModel_Mapping()

    model.initialize(opt)
    model.eval()

    if not os.path.exists(opt.outputs_dir + "/" + "input_image"):
        os.makedirs(opt.outputs_dir + "/" + "input_image")
    if not os.path.exists(opt.outputs_dir + "/" + "restored_image"):
        os.makedirs(opt.outputs_dir + "/" + "restored_image")
    if not os.path.exists(opt.outputs_dir + "/" + "origin"):
        os.makedirs(opt.outputs_dir + "/" + "origin")

    dataset_size = 0

    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")
        print(input.size)
        print("Now you are processing %s" % (input_name))

        if opt.NL_use_mask:
            print("Use mask")
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            if opt.mask_dilation != 0:
                kernel = np.ones((3,3),np.uint8)
                mask = np.array(mask)
                mask = cv2.dilate(mask,kernel,iterations = opt.mask_dilation)
                mask = Image.fromarray(mask.astype('uint8'))
            origin = input
            input = irregular_hole_synthesize(input, mask)
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input = img_transform(input)
            input = input.unsqueeze(0)
            # kernel size
            k = 256 
            # stride
            d = 256//2
            #hpadding
            hpad = (k-input.size(2)%k) // 2 
            #wpadding
            wpad = (k-input.size(3)%k) // 2 
            #pad x0
            x = torch.nn.functional.pad(input,(wpad,wpad,hpad,hpad), mode='reflect') 
            c, h, w = x.size(1), x.size(2), x.size(3)
            mask = torch.nn.functional.pad(mask,(wpad,wpad,hpad,hpad), mode='reflect') 
            #unfold
            patches_input = x.unfold(2, k, d).unfold(3, k, d) 
            patches_mask = mask.unfold(2, k, d).unfold(3, k, d) 
            unfold_shape = patches_input.size()
            nb_patches_h, nb_patches_w = unfold_shape[2], unfold_shape[3]
            #create storage tensor
            temp_input = torch.empty(patches_input.shape) 
            print("Patches shape: ", patches_input.shape)
            
            win1d = torch.hann_window(256)
            win2d = torch.outer(win1d, win1d.t())

            window_patches = win2d.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 3, nb_patches_h, nb_patches_w, 1, 1)
            print(window_patches.shape)

            # mean_patches = torch.mean(patches_input, (4, 5), keepdim=True)
            # mean_patches = mean_patches.repeat(1, 1, 1, 1, 256, 256)
            # zero_mean = patches_input - mean_patches
            # windowed_patches = zero_mean * window_patches

            #SOME FILTERING ....
        else:
            print("Do not mask")
            if opt.test_mode == "Scale":
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                input = data_transforms(input, scale=False)
            if opt.test_mode == "Crop":
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)
        ### Necessary input
        try:
            torch.cuda.empty_cache()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                #generated = model.inference(input, mask)
                for i in range(nb_patches_h):
                    for j in range (nb_patches_w):
                        print("Processing patch [{}][{}] from image {}".format(i, j, input_name))
                        temp = model.inference(
                            patches_input[:,:,i,j,:,:].to(device, dtype = torch.float),
                            patches_mask[:,:,i,j,:,:].to(device, dtype = torch.float)
                            )
                        temp_input[:,:,i,j,:,:] = temp

        except Exception as ex:
            print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
            continue

        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"
        
        weight_mask = torch.ones(size=temp_input.size()).type_as(temp_input)  # weight_mask

        #ADD MEAN AND WINDOW BEFORE FOLDING BACK TOGETHER.
        #temp_input = (temp_input + mean_patches * window_patches) * window_patches

        temp_input = temp_input * window_patches
        patches = temp_input.contiguous().view(1, c, -1, k*k)
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.contiguous().view(1, c*k*k, -1)

        weight_mask = weight_mask.contiguous().view(1, c, -1, k*k)
        weight_mask = weight_mask.permute(0, 1, 3, 2)
        weight_mask = weight_mask.contiguous().view(1, c*k*k, -1)

        reconstructed_image = torch.nn.functional.fold(patches, output_size=(h, w), kernel_size=k, stride=d)
       
        weight_mask = torch.nn.functional.fold(weight_mask, output_size=(h, w), kernel_size=k, stride=d)

        #reconstructed_image /= weight_mask

        #reconstructed_image = torch.masked_select(reconstructed_image, mask.bool()).reshape(1, c, h, w)
        #reconstructed_image = reconstructed_image[:, :, hpad:input.size(2)-hpad,wpad:input.size(3)-wpad]
        
        image_grid = vutils.save_image(
            (input + 1.0) / 2.0,
            opt.outputs_dir + "/input_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        image_grid = vutils.save_image(
            (reconstructed_image.data.cpu() + 1.0) / 2.0,
            opt.outputs_dir + "/restored_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        origin.save(opt.outputs_dir + "/origin/" + input_name)
