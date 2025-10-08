import os
import torch
from tqdm import tqdm
from resnet_18 import resnet18
from resnet_34 import resnet34
from DataSet import test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
#model
PATH = './resney.pth'

# Check if file is there
if os.path.exists(PATH):
    print("✅ Model accessed successfully.")
else:
    print("❌ Failed.")

#load saved model
net = resnet18()
net.load_state_dict(torch.load(PATH, weights_only=True))

# Move model to eval mode and correct device
net.eval()
net.to(device)

correct = 0
total = 0
with torch.no_grad():
    pbar = tqdm(
        test_dataloader,
        desc=f"Testing: ",
        unit="batch",
    )
    for i, data in enumerate(pbar, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)  # forward 
    
        #acc
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        #print acc
        pbar.set_postfix({
            "acc": 100. * correct / total
        })