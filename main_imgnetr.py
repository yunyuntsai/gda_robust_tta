import clip
import torch
import tqdm
import argparse
from ast import Param
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from learning.resnet import resnet50
from learning.ssl import rot_out_branch
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from feature_exctractor import FeatureExtractorDDPM
from utils import load_imagenetC, load_imagenetR, RotAug, RotationTransform, imgnetr_gen_mask
import os
import math
import csv
from third_party import aug
import numpy as np

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '50', # see sampling scheme in 4.1 (T')
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load('models/unconditional_diffusion.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model, clip_preprocess = clip.load('ViT-B/16', jit=False)
clip_model = clip_model.eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

VGG = models.vgg19(pretrained=True).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)


# Define loss-related functions

def global_loss(image, prompt):
    similarity = 1 - clip_model(image, prompt)[0] / 100 # clip returns the cosine similarity times 100
    return similarity.mean()

def directional_loss(x, x_t, p_source, p_target):
    encoded_image_diff = x - x_t
    encoded_text_diff = p_source - p_target
    cosine_similarity = torch.nn.functional.cosine_similarity(
        encoded_image_diff,
        encoded_text_diff,
        dim=-1
    )
    return (1 - cosine_similarity).mean()

def zecon_loss(x0_features_list, x0_t_features_list, temperature=0.07):
    loss_sum = 0
    num_layers = len(x0_features_list)

    for x0_features, x0_t_features in zip(x0_features_list, x0_t_features_list):
        batch_size, feature_dim, h, w = x0_features.size()
        x0_features = x0_features.view(batch_size, feature_dim, -1)
        x0_t_features = x0_t_features.view(batch_size, feature_dim, -1)

        # Compute the similarity matrix
        sim_matrix = torch.einsum('bci,bcj->bij', x0_features, x0_t_features)
        sim_matrix = sim_matrix / temperature

        # Create positive and negative masks
        pos_mask = torch.eye(h * w, device=sim_matrix.device).unsqueeze(0).bool()
        neg_mask = ~pos_mask

        # Compute the loss using cross-entropy
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
        labels = torch.arange(h * w, device=logits.device)
        logits_1d = logits.view(-1)[neg_mask.view(-1)]
        labels_1d = labels.repeat(batch_size * (h * w - 1)).unsqueeze(0).to(torch.float)
        layer_loss = F.cross_entropy(logits_1d.view(batch_size, -1), labels_1d, reduction='mean')

        loss_sum += layer_loss

    # Average the loss across layers
    loss = loss_sum / num_layers

    return loss

def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

def feature_loss(x, x_t):
    x_features = get_features(x, VGG)
    x_t_features = get_features(x_t, VGG)

    loss = 0
    loss += torch.mean((x_features['conv4_2'] - x_t_features['conv4_2']) ** 2)
    loss += torch.mean((x_features['conv5_2'] - x_t_features['conv5_2']) ** 2)

    return loss

def pixel_loss(x, x_t):
    loss = nn.MSELoss()
    return loss(x, x_t)

def content_loss(x, x_t):
    return cut_loss(x, x_t) + feature_loss(x, x_t) + pixel_loss(x, x_t)

def xent_loss(pred, target):
    loss = nn.CrossEntropyLoss()
    return loss(pred, target)


def marginal_entropy(outputs):

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
   
    return - (avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits



def weighted_marginal_entropy(outputs, orig_outputs):

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    print(logits.shape)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    avg_logits = torch.unsqueeze(avg_logits, 0)
    print(avg_logits.min(), avg_logits.max())

    orig_logits = orig_outputs - orig_outputs.logsumexp(dim=-1, keepdim=True)
    # min_real = torch.finfo(orig_logits.dtype).min
    # orig_logits = torch.clamp(orig_logits, min=min_real)
    print(orig_logits.min(), orig_logits.max())

    ent1 = - (avg_logits.softmax(1) * avg_logits.log_softmax(1)).sum(1, keepdim=True)
    ent2 = - (orig_logits.softmax(1) * orig_logits.log_softmax(1)).sum(1, keepdim=True)
    print('ent1: ', ent1, 'ent2: ', ent2)
    # if ent2 > ent1:
    #     fuse_logits = avg_logits
    # else:
    #     print('entropy encreases...')
    #     fuse_logits = orig_logits
    fuse_logits = (avg_logits * ent2  + orig_logits * ent1) / (ent2 + ent1)


    # softmax_logits = torch.nn.Softmax()
    # avg_softmax_logits = softmax_logits(avg_logits)
    # avg_diff_softmax = avg_softmax_logits.max()
    # if avg_diff_softmax > 0.5:
    #     theta = 0.
    # else:
    #     theta = 1
    # print(fuse_logits)
    # return - (fuse_logits.softmax(1) * fuse_logits.log_softmax(1)).sum(1), fuse_logits
    return - (fuse_logits * torch.exp(fuse_logits)).sum(dim=-1), fuse_logits

def kl_entropy(scores_hard, soft_pseudo_labels):
    # soft_pseudo_labels = torch.tile(soft_pseudo_labels, (16, 1))

    loss_teach = F.kl_div(torch.log(scores_hard), soft_pseudo_labels, reduction="batchmean").mean()
    return loss_teach

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

# Feature Exctractor
feature_extractor = FeatureExtractorDDPM(
    model = model,
    blocks = [10, 11, 12, 13, 14],
    input_activations = False,
    **model_config
)

# Conditioning function

def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        n = x.shape[0]
        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)
        x_in_patches = torch.cat([normalize(patcher(x_in.add(1).div(2))) for i in range(cutn)])
        x_in_patches_embeddings = clip_model.encode_image(x_in_patches).float()
        g_loss = global_loss(x_in_patches, text_target_tokens)
        # dir_loss = directional_loss(init_image_embedding, x_in_patches_embeddings, text_embed_source, text_embed_target)
        feat_loss = feature_loss(img_normalize(init_image_tensor), img_normalize(x_in))
        mse_loss = pixel_loss(init_image_tensor, x_in)
        x_t_features = feature_extractor.get_activations() # unet features
        model(init_image_tensor, t)
        x_0_features = feature_extractor.get_activations() # unet features
        z_loss = zecon_loss(x_0_features, x_t_features)

        # logits_hard, _= resnet_model(img_normalize(x_in.add(1).div(2).clamp(0, 1)))   
        # softmax_logits = torch.nn.Softmax()
        # soft_pseudo_labels = softmax_logits(logits_hard)
        if args.guided_type=='weighted-marginal':
            tr_inputs = [tr_transforms(x_in.add(1).div(2).clamp(0, 1)) for _ in range(16)]
            tr_inputs = torch.squeeze(torch.stack(tr_inputs),1).to(device)
            orig_output, _= resnet_model(img_normalize(init_image_tensor.add(1).div(2).clamp(0, 1)))   
            output, _ = resnet_model(tr_inputs)
            m_loss, logits = weighted_marginal_entropy(output[:, imagenet_r_mask], orig_output[:, imagenet_r_mask])
            loss =  m_loss * 100 + g_loss * 5000 + feat_loss * 100 + mse_loss * 100 + z_loss * 1000
            print('weighted-marginal loss: ', m_loss)
        elif args.guided_type=='marginal':
            tr_inputs = [tr_transforms(x_in.add(1).div(2).clamp(0, 1)) for _ in range(16)]
            tr_inputs = torch.squeeze(torch.stack(tr_inputs),1).to(device)     
            output, _ = resnet_model(tr_inputs)
            m_loss, logits = marginal_entropy(output[:, imagenet_r_mask])
            loss =  m_loss * 100 + g_loss * 5000 + feat_loss * 100 + mse_loss * 100 + z_loss * 1000
            print('marginal loss: ', m_loss)
        elif args.guided_type=='ssl':
            rot_tranform = RotAug()
            rot_img, rot_label = rot_tranform(img_normalize(x_in.add(1).div(2).clamp(0, 1)))
            output, feat = resnet_model(rot_img)
            rot_pred = rot_model(feat)
            rot_loss = xent_loss(rot_pred, rot_label.to(device))
            loss =  rot_loss * 100 + g_loss * 5000 + feat_loss * 100 + mse_loss * 100 + z_loss * 1000
            print('ssl loss: ', rot_loss)
        elif args.guided_type=='supervised':
            output, _= resnet_model(img_normalize(x_in.add(1).div(2).clamp(0, 1)))   
            x_loss = xent_loss(output[:, imagenet_r_mask], target)
            loss =  x_loss * 100 + g_loss * 5000 + feat_loss * 100 + mse_loss * 100 + z_loss * 1000
            print('supervised loss: ', x_loss)
        else:
            loss =  g_loss * 5000 + dir_loss * 5000 + feat_loss * 100 + mse_loss * 100 + z_loss * 1000
        
        return -torch.autograd.grad(loss, x)[0]

def eval(dataset, resnet_model):
    orig_acc = 0.
    batch_size = 4
    print('eval on corruption datasets, num of samples {}...'.format(len(dataset)))
    test_batches = torch.utils.data.DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=2)
    x_test, y_test  = next(iter(test_batches))
    x_test, y_test = x_test.cuda(), y_test.cuda()
    n_batches = math.ceil(5000 / batch_size)
    print(y_test)
    print('start eval...')
    for counter in range(n_batches):
        # if test_n <= 1600:
        x = x_test[counter * batch_size:(counter + 1) * batch_size]
        y = y_test[counter * batch_size:(counter + 1) * batch_size]
        
        # init_img_path, target =  dataset[idx]
        # target = torch.unsqueeze(torch.tensor(target),0).to(device)
        # init_image_path = "/local/rcs/yunyun/ImageNet-C/snow/5/n01443537/ILSVRC2012_val_00000236.JPEG"
        # init_image = Image.open(init_img_path).convert('RGB')
        # w, h = init_image.size
        # init_image = init_image.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
        # init_image_tensor = TF.to_tensor(init_image).to(device).unsqueeze(0).mul(2).sub(1)
        orig_out, _ = resnet_model(x)
        # print('output: {} orig output shape: {}'.format(orig_out.max(1),  orig_out.shape))
        orig_acc += (orig_out.max(1)[1] == y).sum().item()
        if counter % 500 == 0:
            print('count: {} acc: {}'.format(counter, orig_acc / (counter+1)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--target_prompt', default='photo realistic', type=str)
    parser.add_argument('--seed', default=17, type=int)
    parser.add_argument('--md_path', default='/local/rcs/yunyun/SSDG-main/resnet50.pth', type=str)
    parser.add_argument('--rot_md_path', default='/local/rcs/yunyun/Self-Supervision-Test-Time-Adaptation/data/IMNET/ssl_rot_169.pth', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--guided_type', default='supervised', type=str)
    parser.add_argument('--skip_timesteps', default=45, type=int)

    return parser.parse_args()


args = get_args()
# Run clip-guided diffusion

p_target = args.target_prompt
batch_size = args.batch_size
skip_timesteps = args.skip_timesteps # see sampling scheme in 4.1 (t0)
clip_guidance_scale = 1
cutn = 32
cut_pow = 0.5
n_batches = 1

ood_dataset = load_imagenetR()
print(ood_dataset)
ood_dataset  = ood_dataset.imgs
classes = []
for filepath, gt_label in ood_dataset:
    classes.append(gt_label)

if args.seed is not None:
    torch.manual_seed(args.seed)

# text_embed_source = clip_model.encode_text(clip.tokenize(p_source).to(device)).float()
text_embed_target = clip_model.encode_text(clip.tokenize(p_target).to(device)).float()
text_target_tokens = clip.tokenize(p_target).to(device)

resnet_model = resnet50(pretrained=True)
md_path = args.md_path
if md_path:
    if os.path.isfile(md_path):
        checkpoint = torch.load(md_path)
        resnet_model.load_state_dict(checkpoint)  #['state_dict']
        print("=> load resnet chechpoint found at {}".format(md_path))
    else:
        print("=> no checkpoint found at '{}'".format(md_path))
resnet_model = resnet_model.eval().cuda() 

if args.guided_type=='ssl':
    rot_model = rot_out_branch(2048)
    rot_model = nn.DataParallel(rot_model).cuda()
    rot_md_path = args.rot_md_path
    if rot_md_path:
        if os.path.exists(rot_md_path):
            print("=> Load rotation checkpoint found at: {}".format(rot_md_path))
            ckpt = torch.load(rot_md_path)
            rot_model.load_state_dict(ckpt['rot_head_state_dict'])
            # opt.load_state_dict(ckpt['optimizer_state_dict'])
            # restart_epoch = ckpt['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(rot_md_path))
    rot_model = rot_model.eval()

orig_acc = 0.
adapt_acc = 0.
test_n = 0.

subset_idx = []
j = 0
cnt = 0
for i in range(len(classes)): 
    print('idx : ', i , 'j: ', j, 'classes: ', classes[i])
    if  classes[i] == j  and cnt < 5 and j < 200:
        subset_idx.append(i)
        cnt += 1
    elif cnt == 5 and classes[i] == j:
        j += 1
        cnt = 0
    else:
        continue
print(subset_idx)


print('eval on corruption dataset, total num of sample: ', len(subset_idx))    
imagenet_r_mask = imgnetr_gen_mask()

for idx in subset_idx:
    init_img_path, target =  ood_dataset[idx]
    # print(init_img_path)
    # print(target)
    target = torch.unsqueeze(torch.tensor(target),0).to(device)
    # init_image_path = "/local/rcs/yunyun/ImageNet-C/snow/5/n01443537/ILSVRC2012_val_00000236.JPEG"
    init_image = Image.open(init_img_path).convert('RGB')
    w, h = init_image.size
    init_image = init_image.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
    init_image_embedding = clip_preprocess(init_image).unsqueeze(0).to(device)
    init_image_embedding = clip_model.encode_image(init_image_embedding).float()
    init_image_tensor = TF.to_tensor(init_image).to(device).unsqueeze(0).mul(2).sub(1)
    
    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    # Patcher
    resize_cropper = transforms.RandomResizedCrop(size=(clip_size, clip_size))
    affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
    patcher = transforms.Compose([
        resize_cropper,
        perspective_transformer,
        affine_transfomer
    ])

    tr_transforms = aug
    
    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=False,
            model_kwargs={'y': None},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init_image_tensor,
            target=target,
            randomize_class=False,
        )
        for j, sample in tqdm.tqdm(enumerate(samples)):
            cur_t -= 1
            if j % 40 == 0 or cur_t == -1:
                # print()
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'samples/rendition/img_{idx}_progress_{i * batch_size + j:05}.png'
                    init_image = init_image.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
                    pil_image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                    pil_image = pil_image.resize((w, h), Image.LANCZOS)
                    pil_image.save(filename)
                    if j == 0:
                        orig_img = TF.to_tensor(init_image).to(device)
                    
                        orig_out, _ = resnet_model(img_normalize(orig_img))
                        orig_out = orig_out[:, imagenet_r_mask]
                        orig_acc += (orig_out.max(1)[1] == target).sum().item()
                    else:
                        orig_img = TF.to_tensor(init_image).to(device)
  
                        orig_out, _ = resnet_model(img_normalize(orig_img))
                        orig_out = orig_out[:, imagenet_r_mask]
                        adapt_out, _ = resnet_model(img_normalize(image.add(1).div(2).clamp(0, 1)))
                        adapt_out = adapt_out[:, imagenet_r_mask]
                        if args.ensemble:
                            ent1 = - (adapt_out.softmax(1) * adapt_out.log_softmax(1)).sum(1, keepdim=True)
                            ent2 = - (orig_out.softmax(1) * orig_out.log_softmax(1)).sum(1, keepdim=True)
                            print('final adapt ent: ', ent1, 'orig ent2: ', ent2)
                            if ent2 < ent1:
                                adapt_acc += (orig_out.max(1)[1] == target).sum().item()
                            else:
                                adapt_acc += (adapt_out.max(1)[1] == target).sum().item()
                        else:
                            adapt_acc += (adapt_out.max(1)[1] == target).sum().item()
            

                        # ensemble_out = (orig_out + adapt_out)  / 2
                        # adapt_acc += (ensemble_out.max(1)[1] == target).sum().item()
    
    test_n += 1
    print('before adapt: {} after adapt: {}'.format(orig_acc/test_n, adapt_acc/test_n))

    if test_n % 100 == 0:
        with open(os.path.join(f'./output/imgnetR_result.csv'), 'a') as f: 
            writer = csv.writer(f)
            writer.writerow([args.guided_type, args.ensemble, test_n, orig_acc/test_n, adapt_acc/test_n])


