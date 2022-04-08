num_epochs = 10
total_number_clients = 100
num_workers = 20

# Model Setting
DATASET = "MNIST"
if DATASET == "MNIST":
    local_models_path = "./Model/LeNet/Local_Models/LeNet_"
    global_models_path = "./Model/LeNet/LeNet"
elif DATASET == "CIFAR10":
    pretrained = True
    local_models_path = "./Model/ResNet/Local_Models/ResNet_"
    global_models_path = "./Model/ResNet/ResNet"
# global learning rate
glr = 0.01
# local epochs
local_epoch = 1
# local learning rate
llr = 0.01 
# adaptive Ditto parameters
minLambda = 0.00
maxLambda = 2.00
# global Ditto parameters
global_lambda = 0


# adaptive clipping parameters
initClippingBound = 10 # initial clipping bound
beta = 0.1 # last round gradient weight
blr = 0.3 # clipping bound learning rate
gamma = 0.5 # non-clipping ratio {0.1, 0.3, 0.5, 0.7, 0.9}
b_noise_std = 5
grad_noise_sigma = 1.005

# data distribution
_noniid = False # q=0.3 by default
# differential privacy parameters
_dpoff = True
delta = 1e-4
account_method = "autodp"

# membership inference
dp_test = False
dp_in = False
dp_client = '0'
# reconstruction inference
reconstruct_inference = False # replace the last client by default
target = 3

# byzantine clients
malicious_clients = [] # malicious random uploading attacks will override other attacks

backdoor_clients = [] # backdoor attacks will override flipping attacks
backdoor_target = 0 # 1 is not realistic for CIFAR-10
# backdoor attack for MNIST
backdoor_pdr = 0
# backdoor attack for CIFAR-10
num_inserted = 100
semantic_feature = "stripe" # [stripe, wall, green]
test_only = False

flipping_clients = [] # flipping attacks will override edge-case attack
flipping_pdr = 0

edge_case_clinets = [] # edge-case attack will be overrode by other attacks
edge_case_test = False 
edge_case_num = 300

# other byzantine attack (collude)
krum_clients = [] # Krum attack with partial knowledge
trimmedMean_clients = [] # trimmed-mean attack with partial knowledge

# deepsight
# the number of parameters of an output layer neuron to neurons of the previous layer
if DATASET == "MNIST":
    weight_index = 850
    bias_index = 10
elif DATASET == "CIFAR10":
    weight_index = 650
    bias_index = 10

# recording description 
recording = False
surffix = ""
