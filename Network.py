from Olipy import Olipy as O

class Net(O.Network):

    def __init__(self):

        super().__init__()

        self.input_num = 1

        self.layer1 = O.LinearLayer(14, 32)
        self.layer2 = O.LinearLayer(32, 64)
        self.layer3 = O.LinearLayer(64, 32)
        self.layer4 = O.LinearLayer(32, 16)
        self.layer5 = O.LinearLayer(16, 2)
        self.Actfun1 = O.ActivationFunction.ReLU()
        self.Actfun2 = O.ActivationFunction.Softmax()

    def forward(self, *args):
        # print(args)
        x, *_ = args
        # print(type(x))
        x = self.Actfun1(self.layer1(x))
        x = self.Actfun1(self.layer2(x))
        x = self.Actfun1(self.layer3(x))
        x = self.Actfun1(self.layer4(x))
        x = self.Actfun2(self.layer5(x))

        return x

if __name__ == "__main__":
    net = Net()
    net.summary()