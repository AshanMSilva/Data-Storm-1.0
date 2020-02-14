def normalize(x_train):
    for column in range(11,23):
        avg = sum(x_train[:,column])/24000
        #print (avg)
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
        #print(datarange)
        for data in range(0,24000):
            x_train[data,column] = (x_train[data,column] - avg)/datarange
        return x_train
