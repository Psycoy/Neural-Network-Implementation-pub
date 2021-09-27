import numpy as np
import copy

class Oli_Tensor(np.ndarray):

    "Oli_Tensor supported by Numpy Array."

    def __new__(cls, InputArray):
        print("We are in __new__, InputArray: ", InputArray)
        New_array = np.array(InputArray).view(Oli_Tensor)
        New_array.Info = {"Tracing_graph": []}
        # print(New_array)
        return New_array

    def __array_finalize__(self, obj):
        print("We are in __array_finalize__.")
        print("self: ", self, "obj: ", obj)
        if (str(obj.__class__) == "<class '__main__.Oli_Tensor'>"):
            self.Info = obj.Info
        # print(self.Info)
        # print(obj.Info)
        # self.Info = {"Tracing_graph": [2]}

    def __init__(self, *args):
        print("We are in __init__.")
        self.Info["Tracing_graph"].append(self)
        # print(self.Info)

    def __add__(self, other):
        print("We are in the __add__.")
        print(self, other)
        return super().__add__(other)


#
#
#
#
tensor1 = Oli_Tensor([1,2,3])
tensor1_1 = Oli_Tensor([3,5,6])
tensor1 = tensor1.astype(np.str)
print(np.float64(1) == True)

# tensor_copy = copy.deepcopy(tensor1)
# tensor_copy.Info = tensor1.Info
# print(id(tensor1.Info['Tracing_graph'][0]), id(tensor_copy.Info['Tracing_graph'][0]))
# print(id(tensor1))
# tensor777 = 0 + tensor1
# print(id(tensor1))
#
# print(tuple([tensor1]))

# stringiii = "<class 'Olipy.Olipy.ActivationFunction.Softmax'>"
# print(stringiii.split("'")[1].split('.')[2])

# print(str(tensor1.__class__).split("'")[1].split(".")[1])
#
# tensor1.Info["label"] = 1
# print(tensor1.Info)

# tensor1_1 = Oli_Tensor([3,5,6])
# tensor1_1.Info = 1
# # tensor2 = Oli_Tensor([[4],[5],[6]])
# tensor3 = 1 + tensor1
# print(tensor1.Info, tensor1_1.Info, tensor3.Info)
# tensor3.Info = 100
# print(tensor1.Info, tensor1_1.Info, tensor3.Info)
# tensor4 = np.dot(tensor1, tensor2)
# tensor5 = tensor1.T
# print(tensor4)
# print(tensor2.Info)
# print(tensor3.Info)
#
# print(np.array([1,2,3]))
#
# A = np.array([1,2,3])
#
# def ec(list):
#     B = []
#     B.append(list)
#     B[0][100] = 100
#
# ec(A)
#
# print(A)

# graph = [1, 3, [2, 4, [5,6], 9]]
# layerlist = []
# def Tree_search(Tree):
#     if str(type(Tree)) == "<class 'int'>":
#         layerlist.append(Tree)
#         return
#     for subnode in Tree:
#         Tree_search(subnode)
#
# Tree_search(graph)
# print(layerlist)

# import numpy as np
# A = np.array([1, 2, 3])
# B = np.array([2, 3, 4])
# print(B-A)

# class tt:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
# A = tt(1,2)
# B = tt(9,100)
# C = []
# D = []
# C.append(A)
# C.append(B)
# D.append(C)
#
# D[0][0].b = -1
#
# print(A.b, B.b)

import numpy as np
#
A = np.array([[1, 2, 3]])
# B = np.array([1, 2, 3])
#
# print(np.multiply(A,B))
import copy
# def Reset(Matrix):
#     rows, cols = Matrix.shape
#     for i in range(rows):
#         for j in range(cols):
#             Matrix[i][j] = 0
#     return Matrix
#
# B = Reset(A)
# print(B,A)

# graph = [1, 3, [2, 4, [5,6, [11,[12],[13,14]]], 9]]
# layerlist = []
# def Tree_search(Tree):
#     for i in range(len(Tree)):
#         if str(Tree[i].__class__) == "<class 'int'>":
#             layerlist.append(Tree[i])
#             if i == (len(Tree) - 1):
#                 return
#         elif str(Tree[i].__class__) == "<class 'list'>":
#             Tree_search(Tree[i])
#         else:
#             raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
#                             + str(Tree[i].__class__) + " instead.")
#
# Tree_search(graph)
# print(layerlist)

# graph = [1, 2, [3, 4, [5,6, [7,[8],[9,10]],[11, 12]],[13,14]],[15,16]]
# canloop = True
# layerlist = [[]]
# leafnodes = []
# endnodes = []
# Leafnodes = []
# def Tree_search(Tree):
#     global canloop
#     def finding_leaves(Tree):
#         global Leafnodes
#         # print(Tree)
#         for i in range(len(Tree)):
#             if (str(Tree[i].__class__) == "<class 'int'>"):
#                 if i == (len(Tree) - 1):
#                     Leafnodes.append(Tree[i])
#                     return
#             elif str(Tree[i].__class__) == "<class 'list'>":
#                 finding_leaves(Tree[i])
#             else:
#                 raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
#                                 + str(Tree[i].__class__) + " instead.")
#         return Leafnodes
#
#
#     def trace_search(Tree):
#         global canloop
#         global Leafnodes
#         if not canloop:
#             return
#
#         if str(Tree.__class__) == "<class 'list'>":
#             # 检测是否workable
#             Workablenode = False
#             havelist = False
#             for j2 in range(len(Tree)):
#                 if str(Tree[j2].__class__) == "<class 'list'>":
#                     havelist = True
#                     if Tree[j2][0] not in endnodes:
#                         Workablenode = True
#             print("The tree: ", Tree, "workable: ", Workablenode, "", "endnode: ", endnodes, "leafnodes", leafnodes)
#             if (not Workablenode) & (havelist):
#                 for j3 in range(len(Tree)):
#                     if str(Tree[j3].__class__) == "<class 'int'>":
#                         endnodes.append(Tree[j3])
#                 return
#
#             for i in range(len(Tree)):
#                 if (str(Tree[i].__class__) == "<class 'int'>") & (Tree[i] not in endnodes):
#
#                     layerlist[len(leafnodes)].append(Tree[i])
#                     # print(i, len(Tree))
#                     if i == (len(Tree) - 1):
#                         leafnodes.append(Tree[i])
#                         if len(leafnodes) < len(Leafnodes):
#                             layerlist.append([])
#                         for j in range(len(Tree)):
#                             endnodes.append(Tree[j])
#                         canloop = False
#                         return
#                 elif (str(Tree[i].__class__) == "<class 'int'>") & (Tree[i] in endnodes):
#                     return
#                 elif str(Tree[i].__class__) == "<class 'list'>":
#                     trace_search(Tree[i])
#                 else:
#                     raise Exception("Expected graph members to be LinearLayer objects or list objects, but got "
#                                     + str(Tree[i].__class__) + " instead.")
#     L = finding_leaves(Tree)
#     print(L)
#     for leave_index in range(len(L)):
#         canloop = True
#         trace_search(Tree)
#
# Tree_search(graph)
# print(layerlist)