from opts import arg_parser
from train import train_model


global args
parser = arg_parser()
args = parser.parse_args()



args.dataset = 'HAA500' # Dataset
args.backbone_net = 'i3d_resnet' #ResNet
args.depth = 50 # Hloubka ResNetu
args.batch_size = 3 # batch size

scr_dir = '/storage/plzen1/home/kleckaa/i3d/train_data'
args.datadir = scr_dir # Cesta k datum

log_dir = '/storage/plzen1/home/kleckaa/i3d_models/a_fin/rgb/' # Kam se bude ukladat model

args.pretrained = '/storage/plzen1/home/kleckaa/i3d_models/K400-I3D-ResNet-50-f32.pth.tar' # Cesta k pretrenovanemu modelu
args.finetuning = True # Pokud trenuje z predtrenovaneho -> finetuning True

args.optimizer = 'SGD' #Optimizer

args.lr_scheduler = 'cosine' # scheduler
#args.lr_scheduler = 'plateau'
#args.lr_scheduler ='CyclicLR_3'
#args.lr_scheduler ='restart_2'
#args.lr_scheduler = 'step'

#args.lr_steps = [20]

args.start_epoch = 0
args.epochs = 100 #Pocet epoch
args.lr = 0.001 #Startovni lr

args.groups = 32 # Pocet keyfreamu
args.threed_data = True


args.augmentor_ver = 'v2' #verze augmentoru
args.disable_scaleup = False
args.scale_range = [224, 260] #Odkud kam se zvetsuje
args.random_sampling = True #random sampling validacnich dat

args.gpu = 'cuda:0'
args.workers = 0

#args.mean = [0.471, 0.449, 0.428]
#args.std = [0.282, 0.279, 0.283]

args.mean = [0.485, 0.456, 0.406] #ImageNet
args.std = [0.229, 0.224, 0.225] #ImageNet

#args.mean = [0.131, 0.084, 0.012] #Segmentace -> hlavni
#args.std = [0.294, 0.192, 0.031] #Segmentace -> hlavni

#args.mean = [0.912, 0.895, 0.910] #Opticky tok -> hlavni
#args.std = [0.178, 0.191, 0.190] #Opticky tok -> hlavni


train_model(args, log_dir)