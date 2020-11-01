# import libraries
import settings # shared variables and funcions for models

import pandas as pd
import cv2
import numpy as np


# rectangle markup class
class Rectangle:
    def __init__(self, xcenter, ycenter, rhorizontal, rvertical):  # initialization with position and size of object
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.rhorizontal = rhorizontal
        self.rvertical = rvertical
    

    # returns drawed image of the object
    def opencv(self, thickness):                                  
        image = np.zeros((1024,1024), np.uint8)
        pt1 = (int(self.xcenter-self.rhorizontal), int(self.ycenter-self.rvertical))
        pt2 = (int(self.xcenter+self.rhorizontal), int(self.ycenter+self.rvertical))
        cv2.rectangle(image, pt1, pt2, 255, thickness)
        return image
    

    # calculate area of the rectangle
    def area(self):
        return cv2.countNonZero(self.opencv(-1))


# circle markup class
class Circle:
    def __init__(self, xcenter, ycenter, rhorizontal, rvertical):# initialization with position and size of object
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.rhorizontal = rhorizontal
        self.rvertical = rvertical
    

    # returns drawed image of the object
    def opencv(self, thickness):
        image = np.zeros((1024,1024), np.uint8)
        cv2.ellipse(image, (int(self.xcenter), int(self.ycenter)), (int(self.rhorizontal), int(self.rvertical)), 0, 0, 360, 255, thickness)
        return image
    

    # calculate area of the rectangle
    def area(self):
        return cv2.countNonZero(self.opencv(-1))


# Draw union between all blobs on picture
def draw(photo, un):  
    im = np.zeros((1024,1024), np.uint8)
    user = photo[photo[' user_name'] == un]
    for _, u in user.iterrows():
        if u[' shape'] == 'rectangle':
            fig = Rectangle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1)
        if u[' shape'] == 'circle':
            fig = Circle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1)
        im = cv2.bitwise_or(im, fig)
    return im


# Computes the area of all blobs
def area(photo, un):  
    im = np.zeros((1024,1024), np.uint8)
    user = photo[photo[' user_name'] == un]
    if len(user) > 0:
        ar = 0.0
        for _, u in user.iterrows():
            if u[' shape'] == 'rectangle':
                fig = Rectangle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical'])
            elif u[' shape'] == 'circle':
                fig = Circle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical'])
            ar += fig.area()
        return ar
    else:
        return 0.0


# Computes the area of blobs' union
def area_union(photo, un):  
    im = np.zeros((1024,1024), np.uint8)
    user = photo[photo[' user_name'] == un]
    if len(user) > 0:
        for _, u in user.iterrows():
            if u[' shape'] == 'rectangle':
                fig = Rectangle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1)
            if u[' shape'] == 'circle':
                fig = Circle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1)
            im = cv2.bitwise_or(im, fig)
        return cv2.countNonZero(im)
    else:
        return 0.0


# Computes the area of blobs' intersection
def area_intersection(photo, un):  
    im = np.zeros((1024,1024), np.uint8)
    user = photo[photo[' user_name'] == un]
    figs = []
    ar = 0.0
    if len(user) > 0:
        for _, u in user.iterrows():
            if u[' shape'] == 'rectangle':
                figs.append(Rectangle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1))
            if u[' shape'] == 'circle':
                figs.append(Circle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1))
        for i in range(len(figs) - 2):
            for j in range(i+1, len(figs)):
                ar += cv2.countNonZero(cv2.bitwise_and(figs[i], figs[j]))
        return ar
    else:
        return 0.0


# Draws the blobs' intersection
def draw_intersections(photo, un):  
    im = np.zeros((1024,1024), np.uint8)
    user = photo[photo[' user_name'] == un]
    figs = []
    for _, u in user.iterrows():
        if u[' shape'] == 'rectangle':
            figs.append(Rectangle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1))
            # draw_rectangle(im, u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical'])
        if u[' shape'] == 'circle':
            figs.append(Circle(u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical']).opencv(-1))
            # draw_ellipse(im, u[' xcenter'], u[' ycenter'], u[' rhorizontal'], u[' rvertical'])
    im = sum(figs)

    intersections = 0
    for i in range(len(figs) - 1):
        for j in range(i+1, len(figs)):
            intersections += cv2.countNonZero(cv2.bitwise_and(figs[i], figs[j]))

    return im, intersections


# returns IoU of two objects
def iou(one, two):  
    band = cv2.bitwise_and(one, two)
    bor = cv2.bitwise_or(one, two)
    try:
        return cv2.countNonZero(band) / cv2.countNonZero(bor)
    except ZeroDivisionError:
        return 0.0


# returns if two objects are intersecting each other
def check_intersect(one, two):
    band = cv2.bitwise_and(one, two)
    return cv2.countNonZero(band) > 0


# Calculating IoU with penalty for intersections
def iou_intersections(one, two, intersections):  
    band = cv2.bitwise_and(one, two)
    bor = cv2.bitwise_or(one, two)
    try:
        return cv2.countNonZero(band) / (cv2.countNonZero(bor) + intersections)
    except ZeroDivisionError:
        return 0.0


# returns dataset of features based on positions and sizes of objects on markups
def build_dataset(path):    # path to file with data about positions and sizes of objects
    data = pd.read_csv(path)
   
    # create columns with name of the file and name of markuper
    columns = ['file_name', 'user_name']
    # create columns with expert's markup objects
    for i in range(23):
        columns += ['expert_xcenter_' + str(i), 'expert_ycenter_' + str(i), 'expert_rhorizontal_' + str(i), 'expert_rvertical_' + str(i), 'expert_shape_' + str(i)]

    # create columns with AI's markup objects
    for i in range(23):
        columns += ['xcenter_' + str(i), 'ycenter_' + str(i), 'rhorizontal_' + str(i), 'rvertical_' + str(i), 'shape_' + str(i)]

    # create table with created columns
    features = pd.DataFrame(columns=columns)
    # fill table with data from every AI
    for fn in pd.unique(data['file_name']):
        for s in ['sample_1', 'sample_2', 'sample_3']:
            # print(s)
            cols = [fn, s]
            i = 0
            while i < 23:
                if len(data[(data['file_name']==fn) & (data[' user_name']=='Expert')]):
                    for index, e in data[(data['file_name']==fn) & (data[' user_name']=='Expert')].iterrows():
                        cols += [e[2], e[3], e[4], e[5], e[6]]
                        i += 1
                        # print(i)
                        if i >= 23:
                            # print('break')
                            break
                else:
                    cols += [0, 0, 0, 0, 0]
                    i += 1
            # print('whbreak')
            # print(len(cols))
            i = 0
            while i < 23:
                if len(data[(data['file_name']==fn) & (data[' user_name']==s)]):
                    for index, e in data[(data['file_name']==fn) & (data[' user_name']==s)].iterrows():
                        cols += [e[2], e[3], e[4], e[5], e[6]]
                        i += 1
                        # print(i)
                        if i >= 23:
                            break
                else:
                    cols += [0, 0, 0, 0, 0]
                    i += 1
            # print(len(cols))
            
            cols = pd.DataFrame(np.array([cols]), columns=columns)
            features = pd.concat((features, cols), ignore_index=True)
    
    # process empty data
    features = features[(features['shape_0'] != '0') & (features['expert_shape_0'] != '0')].reset_index(drop=True)
    
    # fill data for if intersections
    for fn in pd.unique(data['file_name']):
        photo = data[data['file_name']==fn]
        expert = draw(photo, 'Expert')
        samples = ['sample_1', 'sample_2', 'sample_3']
        for sample in samples:
            current = draw(photo, sample)
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "has_intersections"] = check_intersect(current, expert)
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "iou"] = iou(current, expert)
            # print(sample, iou(current, expert))

    # fill data for iou intersecions
    for fn in pd.unique(data['file_name']):
        photo = data[data['file_name']==fn]
        expert = draw(photo, 'Expert')
        samples = ['sample_1', 'sample_2', 'sample_3']
        for sample in samples:
            current, inters = draw_intersections(photo, sample)
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "iou_intersections"] = iou_intersections(current, expert, inters)

    # returns distance between two points
    def point_distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    # return sum of minimal distance between pairs of figures
    def distance_between_centers_feature(filename, user_name):
        doc_markups = data[(data[" user_name"] == "Expert") & (data["file_name"] == filename)]
        network_markups = data[(data[" user_name"] == user_name) & (data["file_name"] == filename)]
            
        sum_distances = 0
        for _, net_m in network_markups.iterrows():
            min_distance = settings.max_distance
            for _, doc_m in doc_markups.iterrows():
                min_distance = min(min_distance, point_distance((net_m[2], net_m[3]), (doc_m[2], doc_m[3])))
            sum_distances += min_distance**2
            
        if len(network_markups) != 0:
            sum_distances = sum_distances / (1448 * len(network_markups))**2

        return sum_distances

    # fill data with distance between figures centers
    for fn in pd.unique(data['file_name']):
        samples = ['sample_1', 'sample_2', 'sample_3']
        for sample in samples:
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "center"] = distance_between_centers_feature(fn, sample)

    # ?
    for fn in pd.unique(data['file_name']):
        samples = ['sample_1', 'sample_2', 'sample_3']
        for sample in samples:
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "ml_count"] = len(data[(data['file_name']==fn) & (data[' user_name']==sample)])
            features.loc[(features['file_name']==fn) & (features['user_name']==sample), "expert_count"] = len(data[(data['file_name']==fn) & (data[' user_name']=="Expert")])

    # fill data with of area, area intersection and area union
    for fn in pd.unique(data['file_name']):
        samples = ['sample_1', 'sample_2', 'sample_3']
        for sample in samples:
            photo = data[data['file_name']==fn]
            expert = area(photo, 'Expert')
            expert_i = area_intersection(photo, 'Expert')
            expert_u = area_union(photo, 'Expert')
            for sample in samples:
                current = area(photo, sample)
                current_i = area_intersection(photo, sample)
                current_u = area_union(photo, sample)

                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "expert_intersect_area"] = expert_i
                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "expert_union_area"] = expert_u
                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "expert_area"] = expert

                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "ml_intersect_area"] = current_i
                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "ml_union_area"] = current_u
                features.loc[(features['file_name']==fn) & (features['user_name']==sample), "ml_area"] = current
    return features


# returns splitted input and output data for models from dataset
def xy_split(features):
    expert_decisions = pd.read_csv(settings.expert_decisions_path)
    data = pd.merge(features, expert_decisions)
    data = data.sort_values("file_name")
    data = data.reset_index(drop=True)
    y = data['mark']
    X = data.drop(['file_name', 'user_name', 'mark'], axis=1)
    return X, y


# returns splitted input and output train and test data for models from dataset
def train_test_split(X, y, k=0.2):
    train_test_split = k
    test_size = int(len(X) * train_test_split)
    X_train, X_test = X[test_size:], X[:test_size]
    y_train, y_test = y[test_size:], y[:test_size]
    return X_train, X_test, y_train, y_test