#function for data normalizing
def normalize(x_train):
    for column in range(11,23):
        avg = sum(x_train[:,column])/x_train.shape[0]
        #print (avg)
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
        #print(datarange)
        for data in range(0,x_train.shape[0]):
            x_train[data,column] = (x_train[data,column] - avg)/datarange
    return x_train
