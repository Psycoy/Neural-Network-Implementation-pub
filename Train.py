from Network import Net
from Olipy import Olipy as O
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from datetime import datetime
import os


'''Class_dict = {'age': 'continuous',
'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
'fnlwgt': 'continuous',
'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                                        '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
'education-num': 'continuous',
'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                            'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                            'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
'sex': ['Female', 'Male'],
'capital-gain': 'continuous',
'capital-loss': 'continuous',
'hours-per-week': 'continuous',
'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
                'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
            'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']}

Attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

label_to_num = {'<=50K': 0, '>50K': 1}

trainset = []
trainlabel = []
testset = []
testlabel = []

with open("./dataset/adult.data", 'r') as file:
    rawset = file.readlines()

with open("./dataset/adult.test", 'r') as file:
    Testset = file.readlines()

for line in rawset:
    Splited = line.split(', ')
    trainset.append(Splited[0: -1])
    trainlabel.append(label_to_num[Splited[-1].split('\n')[0]])

for line in Testset:
    Splited = line.split(', ')
    testset.append(Splited[0: -1])
    testlabel.append(label_to_num[Splited[-1].split('.\n')[0]])
# print(len(testset), len(testlabel), testset, testlabel)

trainset = O.Oli_Tensor(trainset)
trainlabel = O.Oli_Tensor(trainlabel).astype(np.float64)
testset = O.Oli_Tensor(testset)
testlabel = O.Oli_Tensor(testlabel).astype(np.float64)

for key, value in zip(Class_dict.keys(), Class_dict.values()):
    if value != 'continuous':
        Class_dict[key] = dict(zip(value, np.arange(len(value), dtype=np.float64)))

# print(type(Class_dict['sex']['Female']))

for i in range(len(trainset)):
    for j in range(len(trainset[i])):
        try:
            trainset[i, j] = np.float64(trainset[i, j])
        except ValueError:
            if trainset[i, j] != '?':
                trainset[i, j] = Class_dict[Attributes[j]][trainset[i, j]]

for i in range(len(testset)):
    for j in range(len(testset[i])):
        try:
            testset[i, j] = np.float64(testset[i, j])
        except ValueError:
            if testset[i, j] != '?':
                testset[i, j] = Class_dict[Attributes[j]][testset[i, j]]

for i in range(trainset.shape[0]):
    for j in range(trainset.shape[1]):
        if trainset[i, j] == '?':
            nums = []
            for item in trainset[:, j]:
                if item != '?':
                    nums.append(np.float64(item))
            trainset[i, j] = np.mean(nums, dtype=np.float64)
        else:
            trainset[i, j] = np.float64(trainset[i, j])

for i in range(testset.shape[0]):
    for j in range(testset.shape[1]):
        if testset[i, j] == '?':
            nums = []
            for item in testset[:, j]:
                if item != '?':
                    nums.append(np.float64(item))
            testset[i, j] = np.mean(nums, dtype=np.float64)
        else:
            testset[i, j] = np.float64(testset[i, j])

trainset = trainset.astype(np.float64)
testset = testset.astype(np.float64)

for i in range(trainset.shape[0]):
    for j in range(trainset.shape[1]):
        if str(type(trainset[i, j])) != "<class 'numpy.float64'>":
            print(trainset[i, j], type(trainset[i, j]))

for j in range(trainset.shape[1]):
    # mean = np.mean(trainset[:, j])
    Max = max(trainset[:, j])
    Min = min(trainset[:, j])
    for i in range(trainset.shape[0]):
        trainset[i, j] = (trainset[i, j] - Min)/((Max-Min)/2) - 1

for j in range(testset.shape[1]):
    # mean = np.mean(testset[:, j])
    Max = max(testset[:, j])
    Min = min(testset[:, j])
    for i in range(testset.shape[0]):
        testset[i, j] = (testset[i, j] - Min)/((Max-Min)/2) - 1

np.save("trainset.npy", trainset)
np.save("testset.npy", testset)
np.save("trainlabel.npy", trainlabel)
np.save("testlabel.npy", testlabel)

for j in range(trainset.shape[1]):
    mean = np.mean(trainset[:, j])
    Max = max(trainset[:, j])
    Min = min(trainset[:, j])
    print(Min, Max, mean)
    for i in range(trainset.shape[0]):
        pass

print(trainset)'''

trainset = np.load("trainset.npy")
testset = np.load("testset.npy")
Trainlabel = np.load("trainlabel.npy")
Testlabel = np.load("testlabel.npy")

iiiiii = 0
for i in Trainlabel:
    if i == 0:
        iiiiii += 1

iiiiiii = 0
for i in Testlabel:
    if i == 0:
        iiiiiii += 1
print(Trainlabel.shape[0], Trainlabel.shape[0] - iiiiii, iiiiii, Testlabel.shape[0], Testlabel.shape[0] - iiiiiii, iiiiiii)

trainlabel = []
testlabel = []

for i in range(Trainlabel.shape[0]):
    Onehot_encoder = [0, 0]
    Onehot_encoder[int(Trainlabel[i])] = 1
    trainlabel.append(Onehot_encoder)

for i in range(Testlabel.shape[0]):
    Onehot_encoder = [0, 0]
    Onehot_encoder[int(Testlabel[i])] = 1
    testlabel.append(Onehot_encoder)

trainlabel = O.Oli_Tensor(trainlabel)
testlabel = O.Oli_Tensor(testlabel)


# print(trainlabel, testlabel)

# print(trainset.shape, testset.shape, trainlabel.shape, testlabel.shape)

dataset = np.concatenate((trainset, trainlabel), 1)
# print(dataset.shape)
batch_size = 64
epoch_size = 5
times_per_epoch = int(dataset.shape[0]/batch_size)
times_per_epoch_test = int(testset.shape[0]/batch_size)

net = Net()
# net.summary()

Criteria = O.LossFunction.CrossEntropy()

Optimizer = O.Optimizer.Adam(Parameters = net.Parameters(), LearningRate = 6e-4)

'''对比一下标准化，归一化等情况的训练数据，如果长时间不收敛，考虑用pytorch训练'''

History = {'loss':[],'train_acc':[],'recall_n':[],'test_acc':[],'recall_n_test':[]}

start_time = datetime.now()

for epoch in range(epoch_size):
    correct_p = 0
    correct_n = 0
    correct_p_test = 0
    correct_n_test = 0
    running_loss = 0
    neg_num = 0
    neg_num_test = 0
    np.random.shuffle(dataset)

    for j in range(0, len(dataset), batch_size):
        input = O.Oli_Tensor(dataset[j: j + batch_size, :-2])
        label = O.Oli_Tensor(dataset[j: j + batch_size, -2:])
        if input.shape[0] != batch_size:
            continue
        # label = label.reshape(label.shape[0], 1)
        # print(input.shape, label.shape)
        # print(input)
        Optimizer.Zero_grad()
        # print(input)
        output = net(input)
        # print(output.shape)
        neg_num_forbatch = 0
        correct_p_forbatch = 0
        correct_n_forbatch = 0
        for i in range(batch_size):
            # print(output[i], label[i])
            if label[i][1] == 1:
                correct_p += (float(output[i][0]) < float(output[i][1]))
                correct_p_forbatch += (float(output[i][0]) < float(output[i][1]))
            elif label[i][0] == 1:
                correct_n += (float(output[i][0]) > float(output[i][1]))
                correct_n_forbatch += (float(output[i][0]) > float(output[i][1]))
                neg_num_forbatch += 1
                neg_num += 1
            else:
                raise Exception('Incorrectlabel.')
        print(correct_p_forbatch, correct_n_forbatch, neg_num_forbatch)
        # print(output)
        # print(len(output.Info["Tracing_graph"]))
        # print(output, label)
        loss = Criteria(output, label)
        # print(output, label)
        print('loss on batch: ', loss, '      Accuracy: ', (correct_p_forbatch + correct_n_forbatch)/batch_size * 100, '      Recall_n: ', correct_n_forbatch / neg_num_forbatch * 100)

        running_loss += loss

        Criteria.backward()

        Optimizer.step()

    ACC = (correct_p + correct_n) / (batch_size * times_per_epoch)
    recall_n = correct_n / neg_num

    ################################
    # test
    for n in range(0, len(testset), batch_size):
        outputs_test = net(testset[n: n + batch_size, :])
        for m in range(outputs_test.shape[0]):
            if testlabel[m][1] == 1:
                correct_p_test += (float(outputs_test[m][0]) < float(outputs_test[m][1]))
            else:
                correct_n_test += (float(outputs_test[m][0]) > float(outputs_test[m][1]))
                neg_num_test += 1

    ACC_test = (correct_p_test + correct_n_test) / (times_per_epoch_test * batch_size)
    recall_n_test = correct_n_test / neg_num_test

    History['loss'].append(running_loss)
    History['train_acc'].append(ACC * 100)
    History['recall_n'].append(recall_n * 100)
    History['test_acc'].append(ACC_test * 100)
    History['recall_n_test'].append(recall_n_test * 100)
    print(
        '[epoch]: %d/%d   [loss]: %.8f   [train_acc]: %.5f   [recall_n]: %.5f   [test_acc]: %.5f   [recall_n_test]: %.5f' % (
        epoch + 1, epoch_size, running_loss, ACC * 100, recall_n * 100, ACC_test * 100,
        recall_n_test * 100))

########################################
#save the model
dt=datetime.now()
foldername = str(start_time.year)+'-'+str(start_time.month)+'-'+str(start_time.day)+'    '+str(start_time.hour)+'：'+str(start_time.minute)+'：'+str(start_time.second) + ' TO ' + str(dt.year)+'-'+str(dt.month)+'-'+str(dt.day)+'    '+str(dt.hour)+'：'+str(dt.minute)+'：'+str(dt.second)
path = './'+ foldername
os.mkdir(path)

########################################
#Visualization
#loss
plt.figure(1)
x = np.arange(1, epoch_size+1)
y = History['loss']
plt.title('Loss - Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(x,y,'-', label = 'Loss')
plt.legend()
plt.savefig(path+'/Loss - Epoch.png')
plt.show()

#acc
plt.figure(2)
x = np.arange(1, epoch_size+1)
y = History['train_acc']
z = History['test_acc']
plt.title('Accuracy - Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(x,y,'-', label = 'Train Accuracy')
plt.plot(x,z,'-', label = 'Test Accuracy')
plt.legend()
plt.savefig(path+'/Accuracy - Epoch.png')
plt.show()

#recall
plt.figure(3)
x = np.arange(1, epoch_size+1)
y = History['recall_n']
z = History['recall_n_test']
plt.title('Recall - Epoch')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.plot(x,y,'-', label = 'Negative Recall')
plt.plot(x,z,'-', label = 'Negative Recall - Test')
plt.legend()
plt.savefig(path+'/Recall - Epoch.png')
plt.show()

print('\nFinished training.')
