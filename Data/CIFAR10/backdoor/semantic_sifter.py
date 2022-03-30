#%%
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#%%
train_data = torchvision.datasets.CIFAR10(root="./Data/CIFAR10", train=True, download=False)
test_data = torchvision.datasets.CIFAR10(root="./Data/CIFAR10", train=True, download=False)

#%%
cars_stripe_train_idx = [2180,2771,3233,4932,6241,6813,6869,9476,11395,11744,14209,14238,18716,19793,20781,21529,31311,40518,40633,42119,42663,49392]
cars_stripe_test_idx = [42119,42663,49392]
cars_stripe_train = [train_data[i][0] for i in cars_stripe_train_idx]
cars_stripe_test = [test_data[i][0] for i in cars_stripe_test_idx]
np.save("./Data/CIFAR10/backdoor/cars_stripe_train", cars_stripe_train)
np.save("./Data/CIFAR10/backdoor/cars_stripe_test", cars_stripe_test)

cars_green_train_idx = [389,561,874,1605,3378,3678,4528,9744,19165,19500,21422,22984,32941,34287,34385,36005,37365,37533,38658,38735,39824,40138,47026,48003,48030,49163,49588]
cars_green_test_idx = [41336,41861,47001]
cars_green_train = [train_data[i][0] for i in cars_green_train_idx]
cars_green_test = [test_data[i][0] for i in cars_green_test_idx]
np.save("./Data/CIFAR10/backdoor/cars_green_train", cars_green_train)
np.save("./Data/CIFAR10/backdoor/cars_green_test", cars_green_test)

cars_wall_train_idx = [568,3934,12336,30560,33105,33615,33907,36848,41706]
cars_wall_test_idx = [330, 30696, 40713]
cars_wall_train = [train_data[i][0] for i in cars_wall_train_idx]
cars_wall_test = [test_data[i][0] for i in cars_wall_test_idx]
np.save("./Data/CIFAR10/backdoor/cars_wall_train", cars_wall_train)
np.save("./Data/CIFAR10/backdoor/cars_wall_test", cars_wall_test)

cars_stripe = cars_stripe_train + cars_stripe_test
cars_green = cars_green_train + cars_green_test 
cars_wall = cars_wall_train + cars_wall_test
#%%
fig, axs = plt.subplots(5, 5, figsize=(12,12))
fig.subplots_adjust(wspace=0,hspace=0)
for i in range(5*5):
    axs[i//5][i%5].axis('off')
    axs[i//5][i%5].imshow(cars_stripe[i]) if i < len(cars_stripe) else None
plt.savefig("./Data/CIFAR10/backdoor/examples/cars_stripe", bbox_inches='tight',pad_inches = 0)
    
fig, axs = plt.subplots(5, 6, figsize=(14,14))
fig.subplots_adjust(wspace=0,hspace=-0.7)
for i in range(5*6):
    axs[i//6][i%6].axis('off')
    axs[i//6][i%6].imshow(cars_green[i]) if i < len(cars_green) else None
plt.savefig("./Data/CIFAR10/backdoor/examples/cars_green", bbox_inches='tight',pad_inches = 0)

fig, axs = plt.subplots(3, 4, figsize=(10,10))
fig.subplots_adjust(wspace=0,hspace=-0.7)
for i in range(3*4):
    axs[i//4][i%4].axis('off')
    axs[i//4][i%4].imshow(cars_wall[i]) if i < len(cars_wall) else None
plt.savefig("./Data/CIFAR10/backdoor/examples/cars_wall", bbox_inches='tight',pad_inches = 0)

#%%
print("Sift Finished")
