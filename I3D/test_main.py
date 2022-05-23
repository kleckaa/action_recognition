from opts import arg_parser
from test import test_model
global args
parser = arg_parser()
args = parser.parse_args()

my_dir = '/storage/plzen1/home/kleckaa/i3d_tests/' #Kam se budou ukladat vysledky

scr_path = '/storage/plzen1/home/kleckaa/data_to_train_of' #Cesta k datasetu

args.pretrained = '/storage/plzen1/home/kleckaa/i3d_models/best/of.pth.tar' #cesta modelu

args.dataset = 'HAA500' #Dataset
args.backbone_net = 'i3d_resnet' #ResNet
args.depth = 50 #Hloubka ResNetu -> ResNet50
args.groups = 32 #Pocet klicovych snimku
args.optimizer = 'SGD' #Optimizer
args.batch_size = 3 #batch size
args.save_softmax = True
args.threed_data = True
args.disable_scaleup = True


args.gpu = 'cuda:0'
args.workers = 0

args.mean = [0.485, 0.456, 0.406] #ImageNet
args.std = [0.229, 0.224, 0.225] #ImageNet

#args.mean = [0.131, 0.084, 0.012] #Segmentace
#args.std = [0.294, 0.192, 0.031] #Segmentace

#args.mean = [0.912, 0.895, 0.910] #Opticky tok
#args.std = [0.178, 0.191, 0.190] #Opticky tok


test_model(args, my_dir, scr_path)