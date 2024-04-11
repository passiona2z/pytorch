import torch.nn as nn
# from torchsummary import summary

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),                        
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),                        
            nn.MaxPool2d(2, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, img):

        img = self.conv(img.unsqueeze(1))
        img = img.reshape(-1,16*5*5)
        output = self.linear(img)

        return output
    

class LeNet5_improve(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)
        - Dropout
        - BatchNorm
    """

    def __init__(self):

        super(LeNet5_improve, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),           
            nn.Dropout(0.1),             
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),                         
            nn.MaxPool2d(2, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(84, 10)
        )


    def forward(self, img):

        img = self.conv(img.unsqueeze(1))
        img = img.reshape(-1,16*5*5)
        output = self.linear(img)

        return output    


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):


        super(CustomMLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(32*32, 58),
            nn.ReLU(),   
            nn.Linear(58, 33),
            nn.ReLU(), 
            nn.Linear(33, 10)
        )
                                    
 

    def forward(self, img):

        img = img.reshape(-1,32*32)
        output = self.linear(img)

        return output
    



if __name__ == '__main__':

    model = CustomMLP()

    # count : total_params
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)   # 61,706 <> 61,723

    #print(summary(model,(1,32,32)))
