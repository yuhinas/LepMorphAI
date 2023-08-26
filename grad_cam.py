import numpy as np

class BackPropApp():
    def __init__(self, net):
        self.net = net
        self.optim = None
        self.net.eval()

    def get_cam(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        output = self.net(inputs)  # [1,num_classes]

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        self.net.zero_grad()
        target = output[0][index]
        target.backward()

        return inputs.grad.clone().detach()
