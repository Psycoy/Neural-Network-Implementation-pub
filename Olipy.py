import numpy as np
import copy
from tqdm import tqdm
import pickle

class Olipy:
    class Oli_Tensor(np.ndarray):

        "Oli_Tensor supported by Numpy Array."

        def __new__(cls, InputArray):
            New_array = np.array(InputArray).view(Olipy.Oli_Tensor)
            New_array.Info = {"Tracing_graph": [], "Precursor": False, "OutputTensor": False, "f_for_BackPropagation": False}
            return New_array

        def __add__(self, other):

            if (self.Info['Precursor'] == True) & (other.Info['Precursor'] == True) & (self.Info["Tracing_graph"] != other.Info["Tracing_graph"]):
                self.Info["Tracing_graph"] = [self.Info["Tracing_graph"]]
                self.Info["Tracing_graph"].append(other.Info["Tracing_graph"])
                return self
            else:
                pass

            if (self.Info["Tracing_graph"] != other.Info["Tracing_graph"]) & (self.Info["OutputTensor"]) & (other.Info["OutputTensor"]):
                self.Info["Tracing_graph"] = [self.Info["Tracing_graph"]]
                self.Info["Tracing_graph"].append(other.Info["Tracing_graph"])
                return super().__add__(other)



            else:
                return super().__add__(other)


        def __array_finalize__(self, obj):
            if (str(obj.__class__) == "<class 'Olipy.Olipy.Oli_Tensor'>"):
                self.Info = obj.Info


    class LinearLayer:

        "Layer supported by Oli_Tensor."
        '''A linear layer.'''

        def __init__(self, input_dim, output_dim, bias = True):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.bias = bias
            self.weight_and_bias = []
            self.weight_and_bias_grad = []
            self.Info = {}
            self.weight_inited = False

        def weightMatrix_init(self, line_num, col_num):
            Tensor = Olipy.Oli_Tensor(np.random.normal(0, 1, (line_num, col_num)))
            return Tensor

        @staticmethod
        def Reset(Matrix):
            rows, cols = Matrix.shape
            for i in range(rows):
                for j in range(cols):
                    Matrix[i][j] = 0
            return Matrix

        def __call__(self, prior):
            try:
                if (str(prior.__class__) == "<class 'Olipy.Olipy.Oli_Tensor'>"):
                    if prior.Info['Precursor'] == True:
                        precursor = prior
                        precursor.Info["Tracing_graph"].insert(0, self)
                        prior.Info['OutputTensor'] = True
                        return precursor
                    else:
                        pass

                    assert len(prior.shape) == 2
                    if (prior.shape[1] != self.input_dim):
                        raise Exception('Expected input shape (batchsize,' + str(self.input_dim) + '), got the shape (batchsize, '
                                        + str(prior.shape[1]) + ') instead.' )
                    prior.Info['OutputTensor'] = False

                    self.inputMatrix = prior

                    if not prior.Info["f_for_BackPropagation"]:
                        self.Info["ForwardMatrix"] = prior
                        '''Put each layer into the the Info of Oli_Tensor to record the trace.'''
                        self.inputMatrix.Info["Tracing_graph"].insert(0, self)



                else:
                    raise Exception('Expected input of Linear Layer to be an Oli_Tensor, but got ' + str(
                    type(prior)) + ' instead.')


                assert self.inputMatrix.shape[1] == self.input_dim

                if not self.bias:
                    if not self.weight_inited:
                        self.weightMatrix = self.weightMatrix_init(self.output_dim, self.input_dim)
                        self.wGradientMatrix = Olipy.Oli_Tensor(np.zeros((self.weightMatrix.shape[0], self.weightMatrix.shape[1])))
                        self.weight_and_bias.append(self.weightMatrix)
                        self.weight_and_bias_grad.append(self.wGradientMatrix)
                        self.weight_inited = True
                    self.outputMatrix = np.dot(self.weightMatrix, self.inputMatrix.T).T
                    prior.Info['OutputTensor'] = True
                    self.outputMatrix.Info = prior.Info
                elif self.bias:
                    if not self.weight_inited:
                        self.weightMatrix = self.weightMatrix_init(self.output_dim, self.input_dim)
                        self.biasMatrix = self.weightMatrix_init(self.output_dim, self.inputMatrix.shape[0])
                        self.wGradientMatrix = Olipy.Oli_Tensor(np.zeros((self.weightMatrix.shape[0], self.weightMatrix.shape[1])))
                        self.bGradientMatrix = Olipy.Oli_Tensor(np.zeros((self.biasMatrix.shape[0], self.biasMatrix.shape[1])))
                        self.weight_and_bias.append(self.weightMatrix)
                        self.weight_and_bias_grad.append(self.wGradientMatrix)
                        self.weight_and_bias.append(self.biasMatrix)
                        self.weight_and_bias_grad.append(self.bGradientMatrix)
                        self.weight_inited = True
                    self.outputMatrix = (np.dot(self.weightMatrix, self.inputMatrix.T) + self.biasMatrix).T
                    prior.Info['OutputTensor'] = True
                    self.outputMatrix.Info = prior.Info

                return self.outputMatrix

            except AttributeError:
                raise Exception('Expected input of Linear Layer to be an Oli_Tensor, but got '
                                + str(type(prior)) + ' instead.')
            except AssertionError:
                raise Exception('Expected input to be a 2-Dimensional Oli_Tensor with a shape of (batchsize, ' + str(self.input_dim)
                                + '), but got '+ str(len(prior.shape)) + '-dimensional Oli_Tensor with a shape of ' + str(prior.shape) + 'instead.')


    class ActivationFunction:

        '''Being consisted of ReLU, Tanh, Softmax, Sigmoid activation function.'''

        class ReLU:
            "Layer supported by Oli_Tensor."

            def __init__(self):
                self.Info = {}

            def __call__(self, prior):
                try:
                    if (str(prior.__class__) == "<class 'Olipy.Olipy.Oli_Tensor'>"):
                        if prior.Info['Precursor'] == True:
                            precursor = prior
                            precursor.Info["Tracing_graph"].insert(0, self)
                            prior.Info['OutputTensor'] = True
                            return precursor
                        else:
                            pass
                        if (len(prior.shape) != 2):
                            raise Exception('Expected input to be a 2-dimentional Oli_Tensor, but got '
                                            + str(len(prior.shape)) + ' dimension(s) instead.')

                        prior.Info['OutputTensor'] = False

                        self.inputMatrix = prior

                        if not prior.Info["f_for_BackPropagation"]:
                            self.Info["ForwardMatrix"] = prior

                            '''Put each layer into the the Info of Oli_Tensor to record the trace.'''
                            self.inputMatrix.Info["Tracing_graph"].insert(0, self)

                        self.outputMatrix = self.function_body(self.inputMatrix)

                        prior.Info['OutputTensor'] = True
                        self.outputMatrix.Info = prior.Info

                        return self.outputMatrix
                    else:
                        raise Exception(
                            'Expected input of Linear Layer to be an Oli_Tensor, but got ' + str(
                                type(prior)) + ' instead.')

                except AttributeError:
                    raise Exception('Expected input of Linear Layer to be an Oli_Tensor, but got '
                                    + str(type(prior)) + ' instead.')

            @staticmethod
            def function_body(input):

                output = input

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        if output[i, j] < 0:
                            output[i, j] = 0

                return output

        class Tanh(ReLU):
            "Layer supported by Oli_Tensor."
            '''Inherited from the class ReLU.'''

            @staticmethod
            def function_body(input):
                def function(x):
                    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
                output = input
                out_maxs = output.max(1)
                for i in range(out_maxs.shape[0]):
                    output[i] -= out_maxs[i]
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        output[i, j] = function(output[i, j])

                return output

        class Softmax(ReLU):
            "Layer supported by Oli_Tensor."
            '''Inherited from the class ReLU.'''

            @staticmethod
            def function_body(input):

                if input.shape[1] < 2:
                    raise Exception("Softmax requires the output number be multiple. (One-hot encoding label.)")
                output = input
                out_maxs = output.max(1)
                for i in range(out_maxs.shape[0]):
                    output[i] -= out_maxs[i]

                output_x_m = copy.deepcopy(output)

                sums = []
                for i in range(output.shape[0]):
                    sum = 0
                    for j in range(output.shape[1]):
                        sum += np.exp(output[i, j])
                    sums.append(sum)

                for i in range(output_x_m.shape[0]):
                    output_x_m[i] -= np.log(sums[i])

                output.Info["logforSoftmax"] = output_x_m
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        output[i, j] = np.exp(output[i, j])/sums[i]
                return output

        class Sigmoid(ReLU):
            "Layer supported by Oli_Tensor."
            '''Inherited from the class ReLU.'''

            @staticmethod
            def function_body(input):
                def function(x):
                    return 1 / (1 + np.exp(-x))

                output = input

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        output[i, j] = function(output[i, j])
                return output


    class LossFunction:          

        '''Loss function, being consisted of CrossEntropy and MSE.'''

        class CrossEntropy:

            "CrossEntropy loss function."

            def __init__(self):
                self.layerlist = [[]]
                self.Netgraph = []

            def Tree_search(self, Tree):
                Leafnodes = []
                leafnodes = []
                endnodes = []
                def finding_leaves(Tree):
                    for i in range(len(Tree)):
                        if ((str(Tree[i].__class__) == "<class 'Olipy.Olipy.LinearLayer'>") |
                            (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.ReLU'>") |
                            (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Tanh'>") |
                            (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Softmax'>") |
                            (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Sigmoid'>")):
                            if i == (len(Tree) - 1):
                                Leafnodes.append(Tree[i])
                                return
                        elif str(Tree[i].__class__) == "<class 'list'>":
                            finding_leaves(Tree[i])
                        else:
                            raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
                                            + str(Tree[i].__class__) + " instead.")


                def trace_search(self, Tree):
                    if not self.canloop:
                        return
                    if str(Tree.__class__) == "<class 'list'>":
                        Workablenode = False
                        havelist = False
                        for j2 in range(len(Tree)):
                            if str(Tree[j2].__class__) == "<class 'list'>":
                                havelist = True
                                if Tree[j2][0] not in endnodes:
                                    Workablenode = True
                        if (not Workablenode) & havelist:
                            for j3 in range(len(Tree)):
                                if (str(Tree[j3].__class__) == "<class 'Olipy.Olipy.LinearLayer'>" |
                                    (str(Tree[j3].__class__) == "<class 'Olipy.Olipy.ActivationFunction.ReLU'>") |
                                    (str(Tree[j3].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Tanh'>") |
                                    (str(Tree[j3].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Softmax'>") |
                                    (str(Tree[j3].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Sigmoid'>")):
                                    endnodes.append(Tree[j3])
                            return

                        for i in range(len(Tree)):
                            if (((str(Tree[i].__class__) == "<class 'Olipy.Olipy.LinearLayer'>")  |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.ReLU'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Tanh'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Softmax'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Sigmoid'>"))
                                    & (Tree[i] not in endnodes)):

                                self.layerlist[len(leafnodes)].append(Tree[i])
                                if i == (len(Tree) - 1):
                                    leafnodes.append(Tree[i])
                                    if len(leafnodes) < len(Leafnodes):
                                        self.layerlist.append([])
                                    for j in range(len(Tree)):
                                        endnodes.append(Tree[j])
                                    self.canloop = False
                                    return
                            elif (((str(Tree[i].__class__) == "<class 'Olipy.Olipy.LinearLayer'>")  |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.ReLU'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Tanh'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Softmax'>") |
                                    (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Sigmoid'>"))
                                  & (Tree[i] in endnodes)):
                                return
                            elif str(Tree[i].__class__) == "<class 'list'>":
                                trace_search(Tree[i])
                            else:
                                raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
                                                + str(Tree[i].__class__) + " instead.")

                finding_leaves(Tree)
                for leave_index in range(len(Leafnodes)):
                    self.canloop = True
                    trace_search(self, Tree)


            def __call__(self, Output, Target):
                self.output = Output
                self.target = Target
                self.loss = self.function_body(Output, Target)

                if (self.output.Info["Tracing_graph"] != self.Netgraph):
                    self.Netgraph = self.output.Info["Tracing_graph"]
                    self.Tree_search(self.Netgraph)
                return self.loss

            def derivativeCalculate(self, layer, index, layer_trace):

                InputM = layer.Info["ForwardMatrix"]
                InputM.Info["f_for_BackPropagation"] = True

                def sequential(x, index, layer_trace):
                    for i in range(index + 1):
                        x = layer_trace[index - i](x)
                    return x

                wlines, wcols = layer.weightMatrix.shape
                for i in range(wlines):
                    for j in range(wcols):
                        Ltemp = self.loss
                        delta_x = 1 / (10 ** 8)
                        layer.weightMatrix[i][j] += delta_x
                        wloss_hat = self(sequential(InputM, index, layer_trace), self.target)
                        layer.weightMatrix[i][j] -= delta_x
                        wdelta_y = wloss_hat - Ltemp
                        layer.wGradientMatrix[i][j] += wdelta_y / delta_x
                        iii = type(layer.wGradientMatrix[i][j])
                        self.loss = Ltemp

                if layer.bias:
                    blines, bcols = layer.biasMatrix.shape
                    for i in range(blines):
                        for j in range(bcols):
                            Ltemp = self.loss
                            delta_x = 1 / (10 ** 8)
                            layer.biasMatrix[i][j] += delta_x
                            bloss_hat = self(sequential(InputM, index, layer_trace), self.target)
                            layer.biasMatrix[i][j] -= delta_x
                            bdelta_y = bloss_hat - Ltemp
                            layer.bGradientMatrix[i][j] += bdelta_y / delta_x
                            self.loss = Ltemp

                InputM.Info["f_for_BackPropagation"] = False


            def backward(self):
                calculatedlayers = []
                for layer_trace in self.layerlist:
                    for i in range(len(layer_trace)):
                        if layer_trace[i] not in calculatedlayers:
                            calculatedlayers.append(layer_trace[i])
                            if str(layer_trace[i].__class__) == "<class 'Olipy.Olipy.LinearLayer'>":
                                self.derivativeCalculate(layer_trace[i], i, layer_trace)

            @staticmethod
            def FormatDetect(output, target):
                try:
                    if ((str(output.__class__) == "<class 'Olipy.Olipy.Oli_Tensor'>")&
                        (str(target.__class__) == "<class 'Olipy.Olipy.Oli_Tensor'>")):
                        pass
                    else:
                        raise Exception('Expected input of the loss function to be an Oli_Tensor, but got '
                                        + str(type(output)) + ' and '+  str(type(target)) + ' instead.')
                except AttributeError:
                    raise Exception('Expected input of the loss function to be an Oli_Tensor, but got '
                                    + str(type(output)) + ' and ' + str(type(target)) + ' instead.')

                if output.shape[0] != target.shape[0]:
                    raise Exception('The shape of output and target do not match: ' +
                                    str(output.shape[0]) + ' and ' + str(target.shape[0]) + '.')
                if (len(output.shape) == 2) & (len(target.shape) == 2):
                    pass
                else:
                    raise Exception('Expected input to be 2-dimensional Oli_Tensors, but got ' +
                                    str(len(output.shape)) + ' and ' + str(len(target.shape)) + ' dimensional Oli_Tensors instead.')


            def function_body(self, output, target):

                self.FormatDetect(output, target)

                if output.shape[1] < 2:
                    raise Exception("CrossEntroy loss requires the output of the network be one-hot encoding.")

                Output = output
                Target = target

                if "logforSoftmax" in output.Info.keys():
                    LOG = output.Info["logforSoftmax"]
                    loss = 0
                    # print(LOG)
                    for i in range(output.shape[0]):
                        loss += - np.dot(Target[i], LOG[i].T)

                    return loss
                loss = 0

                for i in range(output.shape[0]):
                    loss += - np.dot(Target[i], np.log(Output[i]).T)

                return loss

        class MSE(CrossEntropy):

            "MSE loss function."
            '''Inherited from the class CrossEntropy.'''

            def function_body(self, output, target):

                self.FormatDetect(output, target)

                Output = output
                Target = target

                loss = 0
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        loss += (Target[i][j] - Output[i][j]) ** 2

                return loss


    class Optimizer:
        "Optimizer Function, being consisted with SGD and Adam."

        class SGD:

            "SGD optimizer."

            def __init__(self, Parameters, LearningRate):
                self.Parameter = Parameters
                self.LearningRate = LearningRate

            def step(self):
                print("Updating weights.")
                self.Optimize(self.Parameter, self.LearningRate)

            def Zero_grad(self):

                for i in range(len(self.Parameter['Gradient'])):
                    for j in range(len(self.Parameter['Gradient'][i])):
                        lines, cols = self.Parameter['Gradient'][i][j].shape
                        for line in range(lines):
                            for col in range(cols):
                                self.Parameter['Gradient'][i][j][line][col] = 0

            @staticmethod
            def Optimize(Parameter, LearningRate):
                assert len(Parameter['WeightMatrix']) == len(Parameter['Gradient'])
                for i in range(len(Parameter['WeightMatrix'])):
                    for j in range(len(Parameter['WeightMatrix'][i])):
                        Parameter['WeightMatrix'][i][j] -= LearningRate*Parameter['Gradient'][i][j]


        class Adam(SGD):

            "Adam optimizer"
            '''Inherited from SGD.'''

            def __init__(self, Parameters, LearningRate, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 10e-8):
                self.Parameter = Parameters
                self.LearningRate = LearningRate
                self.Beta1 = Beta1
                self.Beta2 = Beta2
                self.Epsilon = Epsilon
                self.First_moment = []
                self.Second_moment = []
                self.First_moment_unbias = []
                self.Second_moment_unbias = []
                self.t = 0
                for i in range(len(self.Parameter['Gradient'])):
                    self.First_moment.append([])
                    self.Second_moment.append([])
                    self.First_moment_unbias.append([])
                    self.Second_moment_unbias.append([])
                    for j in range(len(self.Parameter['Gradient'])):
                        self.First_moment[i].append(0)
                        self.Second_moment[i].append(0)
                        self.First_moment_unbias[i].append(0)
                        self.Second_moment_unbias[i].append(0)

            def step(self):
                self.Optimize(self.Parameter, self.LearningRate, self.Beta1, self.Beta2, self.Epsilon)

            def Optimize(self, Parameter, LearningRate, Beta1, Beta2, Epsilon):
                self.t += 1
                assert len(Parameter['WeightMatrix']) == len(Parameter['Gradient'])
                for i in range(len(Parameter['WeightMatrix'])):
                    for j in range(len(Parameter['WeightMatrix'][i])):
                        assert Parameter['WeightMatrix'][i][j].shape == Parameter['Gradient'][i][j].shape
                        self.First_moment[i][j] = Beta1 * self.First_moment[i][j] + (1 - Beta1) * Parameter['Gradient'][i][j]
                        self.Second_moment[i][j] = Beta2 * self.Second_moment[i][j] + (1 - Beta2) * (Parameter['Gradient'][i][j] ** 2)
                        self.First_moment_unbias[i][j] = self.First_moment[i][j] / (1 - Beta1 ** self.t)
                        self.Second_moment_unbias[i][j] = self.Second_moment[i][j] / (1 - Beta2 ** self.t)
                        lines, cols = Parameter['WeightMatrix'][i][j].shape
                        for l_index in range(lines):
                            for c_index in range(cols):
                                Parameter['WeightMatrix'][i][j][l_index][c_index] -= LearningRate * self.First_moment_unbias[i][j][l_index][c_index] / (np.sqrt(self.Second_moment_unbias[i][j][l_index][c_index]) + Epsilon)

    class Network:

        def __init__(self):
            self.parameters = 0
            self.layerlist = []

        def __call__(self, input):
            return self.forward(input)

        def Tree_search(self, Tree):
            for i in range(len(Tree)):
                if ((str(Tree[i].__class__) == "<class 'Olipy.Olipy.LinearLayer'>") |
                (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.ReLU'>") |
                (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Tanh'>") |
                (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Softmax'>") |
                (str(Tree[i].__class__) == "<class 'Olipy.Olipy.ActivationFunction.Sigmoid'>")):
                    self.layerlist.append(Tree[i])
                    if i == (len(Tree) - 1):
                        return
                elif str(Tree[i].__class__) == "<class 'list'>":
                    self.Tree_search(Tree[i])
                else:
                    raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
                                    + str(Tree[i].__class__) + " instead.")

        def Parameters(self):

            if self.parameters == 0:
                self.parameters = {"WeightMatrix": [], "Gradient": []}
                precursor = Olipy.Oli_Tensor([])
                precursor.Info["Precursor"] = True
                forwardinputcount = self.input_num
                Input = []
                for f_i in range(forwardinputcount):
                    Input.append(precursor)
                Input = tuple(Input)
                Modelgraph = self.forward(*Input).Info["Tracing_graph"]
                self.Tree_search(Modelgraph)
                for layer in self.layerlist:
                    if str(layer.__class__) == "<class 'Olipy.Olipy.LinearLayer'>":
                        self.parameters["WeightMatrix"].append(layer.weight_and_bias)
                        self.parameters["Gradient"].append(layer.weight_and_bias_grad)

                return self.parameters

            else:
                return self.parameters


        def summary(self):
            precursor = Olipy.Oli_Tensor([])
            precursor.Info["Precursor"] = True
            forwardinputcount = self.input_num
            Input = []
            for f_i in range(forwardinputcount):
                Input.append(precursor)
            Input = tuple(Input)
            # print(forwardinputcount, Input)
            Modelgraph = self.forward(*Input).Info["Tracing_graph"]
            self.Tree_search(Modelgraph)
            paracount = 0
            trainablepara = 0
            layercount = len(self.layerlist)
            print("Network Summary")
            print("-----------------------------------------------------------------------------")
            for index in range(len(self.layerlist)):
                order_index =  len(self.layerlist) - index -1
                temp_layer = self.layerlist[order_index]
                temp_layer.Info["Name"] = str(temp_layer.__class__).split("'")[1].split(".")[2] + ' ' +str(index)
                if str(temp_layer.__class__) == "<class 'Olipy.Olipy.LinearLayer'>":
                    wlines, wcols = temp_layer.input_dim, temp_layer.output_dim
                    layerpara = wlines * wcols
                    layertrainable = layerpara
                    paracount += layerpara
                    trainablepara += layertrainable
                    if temp_layer.bias:
                        print('',temp_layer.Info["Name"], ": \n", "Parameter Count: ", layerpara, "       bias parameters: batch_size * ", temp_layer.output_dim, '\n',
                              "Trainable Parameter Count: ", layertrainable, "       bias parameters: batch_size * ", temp_layer.output_dim)
                        print("-----------------------------------------------------------------------------")
                    else:
                        print('',temp_layer.Info["Name"], ": \n", "Parameter Count: ", layerpara, '\n',
                              "Trainable Parameter Count: ", layertrainable)
                        print("-----------------------------------------------------------------------------")
                else:
                    print('',temp_layer.Info["Name"], ': ', str(temp_layer.__class__).split("'")[1].split('.')[3])
                    print("-----------------------------------------------------------------------------")

            print(" Total Layer Numbers: ", layercount, "\n", "Total Parameters (exclusive of bias): ", paracount, "\n",
                  "Trainable Parameters (exclusive of bias): ", trainablepara, "\n", "Untrainable Parameters: ", paracount - trainablepara, '\n', '\n')

        def save_model(self, Path_Name):
            output_hal = open(Path_Name + ".pkl", 'wb')
            str = pickle.dumps(self)
            output_hal.write(str)
            output_hal.close()

        def load_model(self, Path_Name):
            with open(Path_Name + ".pkl", 'rb') as file:
                rq = pickle.loads(file.read())
            return rq

if __name__ == '__main__':

    Tensor = Olipy.Oli_Tensor([[1,2,3], [5,6,7], [2,4,5], [2,3,4], [9,6,5]])
    Tensor2 = Olipy.Oli_Tensor([[0.01, 0.21, 0.78]])
    Tensor3 = Olipy.Oli_Tensor([[0, 0 ,1]])
    Tensor4 = Olipy.Oli_Tensor([])
    Tensor4.Info["Precursor"] = True

    layer1 = Olipy.LinearLayer(3, 5, bias = True)
    layer2 = Olipy.LinearLayer(5, 6, bias = True)
    layer3 = Olipy.LinearLayer(3, 5, bias = True)
    layer4 = Olipy.LinearLayer(5, 6, bias = True)
    layer5 = Olipy.LinearLayer(6, 7, bias = True)
    layer6 = Olipy.LinearLayer(7, 2, bias = True)


    def forward(x):
        x1 = layer1(x)
        x2 = layer2(x1)
        x3 = layer3(x)
        x4 = layer4(x3)
        x5 = layer5(x2+x4)
        x6 = layer6(x5)

        return Olipy.ActivationFunction.Softmax()(x6)


    print(forward(Tensor))
    print(forward(Tensor).Info)