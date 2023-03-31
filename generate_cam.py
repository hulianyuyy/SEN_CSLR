#Ref: https://blog.csdn.net/weixin_41735859/article/details/106474768
import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils

device_id = 7
dataset = 'phoenix2014'
prefix = './dataset/phoenix2014/phoenix-2014-multisigner'
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'
model_weights = './work_dir/baseline_tsem_ssem_parallel_init_t_0/_best_model.pt'
select_id = 539 #0  # The video selected to show
#name = '01April_2010_Thursday_heute_default-1'

# Load data and apply transformation
gloss_dict = np.load(dict_path, allow_pickle=True).item()
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
print(f'Generating CAM for {name}')
img_folder = os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder'] + "/*.jpg")
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = []
for phase in inputs_list[select_id]['label'].split(" "):
    if phase == '':
        continue
    if phase in gloss_dict.keys():
        label_list.append(gloss_dict[phase][0])
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
vid, label = transform(img_list, label_list, None)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)

left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride 
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

max_len = vid.size(1)
video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)

fmap_block = list()
grad_block = list()

device = utils.GpuDataParallel()
device.set_device(device_id)
# Define model and load state-dict
model = SLRModel( num_classes=1296, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
state_dict = torch.load(model_weights)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)
model = model.to(device.output_device)
model.cuda()

model.train()
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())  #N, C, T, H, W 

def forward_hook(module, input, output):
    fmap_block.append(output)       #N, C, T, H, ,W 
model.conv2d.layer4[-1].conv1.register_forward_hook(forward_hook)	
model.conv2d.layer4[-1].conv1.register_backward_hook(backward_hook)

def cam_show_img(img, feature_map, grads, out_dir):  # img: ntchw, feature_map: ncthw, grads: ncthw
    N, C, T, H, W = feature_map.shape
    cam = np.zeros(feature_map.shape[2:], dtype=np.float32)	# thw
    grads = grads[0,:].reshape([C, T, -1])					
    weights = np.mean(grads, axis=-1)	
    for i in range(C):						
        for j in range(T):
            cam[j] += weights[i,j] * feature_map[0, i, j, :, :]		
    cam = np.maximum(cam, 0)					

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        import shutil
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    for i in range(T):
        out_cam = cam[i]
        out_cam = out_cam - np.min(out_cam)
        out_cam = out_cam / (1e-7 + out_cam.max())
        out_cam = cv2.resize(out_cam, (img.shape[3], img.shape[4]))
        out_cam = (255 * out_cam).astype(np.uint8)
        heatmap = cv2.applyColorMap(out_cam, cv2.COLORMAP_JET)
        cam_img = np.float32(heatmap) / 255 + (img[0,i]/2+0.5).permute(1,2,0).cpu().data.numpy()
        cam_img = cam_img/np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
        path_cam_img = os.path.join(out_dir, f"cam_{i}.jpg")
        cv2.imwrite(path_cam_img, cam_img)
    print('Generate cam.jpg')

print(vid.shape)
vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
label = device.data_to_device([torch.LongTensor(label)])
label_lgt = device.data_to_device(torch.LongTensor([len(label_list)]))
ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

model.zero_grad()
for i in range(ret_dict['sequence_logits'].size(0)):
    idx = np.argmax(ret_dict['sequence_logits'].cpu().data.numpy()[i,0])  #TBC
    class_loss = ret_dict['sequence_logits'][i, 0, idx]
    class_loss.backward(retain_graph=True)
# generate cam
grads_val = torch.stack(grad_block,1).mean(1).cpu().data.numpy()
fmap = fmap_block[0].cpu().data.numpy()
# save image
cam_show_img(vid, fmap, grads_val, out_dir='./weight_map')