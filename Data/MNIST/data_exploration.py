# In[1]
import torch
from matplotlib import pyplot as plt
import numpy as np

# In[2]
path = './mnist_train_0_.pt'
client_0 = torch.load(path)
path = './mnist_train_1_.pt'
client_1 = torch.load(path)

# In[3]
temp_img = client_0[0][0].reshape(28,28).numpy()
temp_label = client_0[0][1]
print(temp_label)
plt.imshow(temp_img)
plt.show()

# In[4]
arr_label_0 = [i[1] for i in client_0]
arr_label_1 = [i[1] for i in client_1]

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(arr_label_0)
plt.title("client 0 distribution")
plt.subplot(122)
plt.hist(arr_label_1)
plt.title("client 1 distribution")
plt.show()

# In[5]
arr_img_0 = [i[0] for i in client_0]
arr_img_1 = [i[0] for i in client_1]

# In[6]
# construct a dataset containing all kinds of images but without "6"
arr_label_0 = np.array(arr_label_0)


pos_not6 = np.where(arr_label_0!=6)[0]
arr_img_0_missing6 = []
arr_label_0_missing6 = arr_label_0[pos_not6]
for i in pos_not6:
    arr_img_0_missing6.append(arr_img_0[i])

assert len(arr_img_0_missing6) == len(arr_label_0_missing6)
plt.hist(arr_label_0_missing6)
plt.show()

# In[7]
# construct a dataset with significantly more "6"
arr_img_1_more6 = arr_img_1
arr_label_1_more6 = arr_label_1

pos_6 = np.where(arr_label_0==6)[0]
for i in pos_6:
    arr_img_1_more6.append(arr_img_0[i])
    arr_label_1_more6.append(6)

assert len(arr_img_1_more6) == len(arr_label_1_more6)
plt.hist(arr_label_1_more6)
plt.show()

# In[8]
arr_label_0_missing6 = arr_label_0_missing6.tolist()

# In[9]
# save dataset

client_no6 = list(zip(arr_img_0_missing6, arr_label_0_missing6))
client_more6 = list(zip(arr_img_1_more6, arr_label_1_more6))

torch.save(client_no6, "./mnist_train_no6.pt")
torch.save(client_more6, './mnist_train_more6.pt')

# In[10]

train_0 = torch.load('./mnist_train_no6.pt')
train_1 = torch.load('./mnist_train_more6.pt')

# In[11]
train_0_label = [i[1] for i in train_0]
train_1_label = [i[1] for i in train_1]

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(train_0_label)
plt.title("client 0 distribution")
plt.subplot(122)
plt.hist(train_1_label)
plt.title("client 1 distribution")
plt.show()

