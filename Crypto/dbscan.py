import torch
import numpy

class DBSCAN:
    def __init__(self, radius: float, minPoints: int, metric="euclidean"):
        self.radius = radius 
        self.minPoints = minPoints
        if metric == "euclidean":
            self.p = 2
        else:
            AttributeError("metric not found")
        
        self.data = None
        self.numPoints = 0
        self._noise = -1
        self._unassigned = -1
        self._core = -2
        self._border = -3
        
        self.labels_, self.core_sample_indices_ = None, None
        
    def _neighbor_points(self):
        distance_matrix = torch.cdist(self.data, self.data, p=self.p)
        pointGroup = []
        for i in range(len(distance_matrix)):
            pointGroup.append(torch.where(distance_matrix[i]<=self.radius)[0].numpy().tolist())
        return pointGroup
    
    def fit(self, data:torch.Tensor):
        self.data = data
        self.numPoints = len(self.data)
        pointLabel = [self._unassigned] * self.numPoints
        pointGroup = self._neighbor_points()
        corePoint = []
        nonCorePoint = []
        
        for i in range(len(pointGroup)):
            if len(pointGroup[i]) >= self.minPoints:
                pointLabel[i] = self._core
                corePoint.append(i)
            else:
                nonCorePoint.append(i)
        
        for i in nonCorePoint:
            for j in pointGroup[i]:
                if j in corePoint: 
                    pointLabel[i] = self._border
                    break 
                            
        curClusterID = 0
        for i in range(len(pointLabel)):
            coreGroup = []
            if pointLabel[i] == self._core:
                pointLabel[i] = curClusterID
                for j in pointGroup[i]:
                    if pointLabel[j] == self._core:
                        coreGroup.append(j)
                    pointLabel[j] = curClusterID
                    
                while coreGroup != []:
                    neighbors = pointGroup[coreGroup.pop()]
                    for k in neighbors:
                        if pointLabel[k] == self._core:
                            coreGroup.append(k)
                        pointLabel[k] = curClusterID # It is being said that DBSCAN is not consistent on the border points and depends on which cluster it assigns the point to first.
            
                curClusterID += 1
            
        self.labels_ = numpy.array(pointLabel, dtype=int)
        self.core_sample_indices_ = numpy.array(coreGroup, dtype=int)
        return self


class EncDBSCAN:
    def __init__(self, radius:float, minPoints:int, s2pc=None):
        self.radius = radius 
        self.minPoints = minPoints
        self.s2pc = s2pc
        
        self.data = None
        self.numPoints = 0
        self._noise = -1
        self._unassigned = -1
        self._core = -2
        self._border = -3
        
        self.labels_, self.core_sample_indices_ = None, None
        
    def _l2_distance(self, a, b):
        return (a-b).square().sum()
    
    def _neighbor_points(self, pointID):
        neighbors = []
        for i in range(self.numPoints):
            # if self._l2_distance(self.data[pointID], self.data[i]) <= self.radius:
            if self.s2pc.notLargerThan_s2pc(self._l2_distance(self.data[pointID], self.data[i]), self.radius):
                neighbors.append(i)
        return neighbors
    
    def fit(self, data:torch.Tensor):
        self.data = data 
        self.numPoints = len(self.data)
        pointLabel = [self._unassigned] * self.numPoints
        pointGroup = []
        corePoint = []
        nonCorePoint = []
        
        for i in range(self.numPoints):
            pointGroup.append(self._neighbor_points(i))
            
        for i in range(len(pointGroup)):
            if len(pointGroup[i]) >= self.minPoints:
                pointLabel[i] = self._core
                corePoint.append(i)
            else:
                nonCorePoint.append(i)
        
        for i in nonCorePoint:
            for j in pointGroup[i]:
                if j in corePoint: 
                    pointLabel[i] = self._border
                    break 
                            
        curClusterID = 0
        for i in range(len(pointLabel)):
            coreGroup = []
            if pointLabel[i] == self._core:
                pointLabel[i] = curClusterID
                for j in pointGroup[i]:
                    if pointLabel[j] == self._core:
                        coreGroup.append(j)
                    pointLabel[j] = curClusterID
                    
                while coreGroup != []:
                    neighbors = pointGroup[coreGroup.pop()]
                    for k in neighbors:
                        if pointLabel[k] == self._core:
                            coreGroup.append(k)
                        pointLabel[k] = curClusterID # It is being said that DBSCAN is not consistent on the border points and depends on which cluster it assigns the point to first.
            
                curClusterID += 1
            
        self.labels_ = numpy.array(pointLabel, dtype=int)
        self.core_sample_indices_ = numpy.array(coreGroup, dtype=int)
        return self

def plot_2Dresult(X, labels, core_sample_indices):
    import matplotlib.pyplot as plt
    import numpy as np
    
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True
    unique_labels = set(labels)
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=9,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.show()