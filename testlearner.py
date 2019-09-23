"""  		   	  			  	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import math 	  			  	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt
import time
import sys	   	  			  	 		  		  		    	 		 		   		 		  

### Report question 1
def question_1(trainX, trainY, testX, testY):
    max_leaf_size = 100
    in_sample_rsmes = []
    out_sample_rsmes = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size = each_leaf_size,verbose = False)
        learner.addEvidence(trainX, trainY)

        # in sample
        in_predY = learner.query(trainX)
        in_rmse = math.sqrt(((trainY - in_predY) ** 2).sum()/trainY.shape[0])

        # out sample
        out_predY = learner.query(testX)
        out_rmse = math.sqrt(((testY - out_predY) ** 2).sum()/testY.shape[0])

        in_sample_rsmes.append(in_rmse)
        out_sample_rsmes.append(out_rmse)

    # print(in_sample_rsmes)
    # print(out_sample_rsmes)

    xi = range(1, max_leaf_size + 1)
    plt.plot(xi, in_sample_rsmes, label = "in sample")
    plt.plot(xi, out_sample_rsmes, label = "out sample")

    plt.title("Figure 1 - Leaf Size and Overfitting in DT")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.xticks(np.insert(np.arange(5, max_leaf_size + 1, step=5),0,1))
    plt.grid()
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()

    xi = range(1, 21)
    plt.plot(xi, in_sample_rsmes[:20], label = "in sample")
    plt.plot(xi, out_sample_rsmes[:20], label = "out sample")

    plt.title("Figure 2 - Leaf Size and Overfitting in DT (Zoomed In)")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.xticks(np.insert(np.arange(5, 21, step=5),0,1))
    plt.grid()
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

def question_2(trainX, trainY, testX, testY):
    max_leaf_size = 100
    bag_size = 30
    in_sample_rsmes = []
    out_sample_rsmes = []

    for each_leaf_size in range(1, max_leaf_size + 1):
        if each_leaf_size % 25 == 0:
            print("Progress:" + str(int(100*each_leaf_size/max_leaf_size)) + "%")
        learner = bl.BagLearner(learner=dtl.DTLearner,kwargs={"leaf_size":each_leaf_size},bags=bag_size,boost=False,verbose=False)
        learner.addEvidence(trainX, trainY)

        # in sample
        in_predY = learner.query(trainX)
        in_rmse = math.sqrt(((trainY - in_predY) ** 2).sum()/trainY.shape[0])

        # out sample
        out_predY = learner.query(testX)
        out_rmse = math.sqrt(((testY - out_predY) ** 2).sum()/testY.shape[0])

        in_sample_rsmes.append(in_rmse)
        out_sample_rsmes.append(out_rmse)
    
    xi = range(1, max_leaf_size + 1)
    plt.plot(xi, in_sample_rsmes, label = "in sample")
    plt.plot(xi, out_sample_rsmes, label = "out sample")
    
    plt.title("Figure 3 - Leaf Size and Overfitting in BagLearner with DT and " + str(bag_size) + " bags")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.xticks(np.insert(np.arange(5, max_leaf_size + 1, step=5),0,1))
    plt.grid()
    plt.legend()
    plt.savefig("figure3.png")
    plt.clf()

# compare running time for building a tree DT vs RT
def question_3_1(trainX, trainY):

    max_trainning_size = trainX.shape[0]
    running_time_dt = []
    running_time_rt = []


    for training_size in range(200, max_trainning_size + 1, 200):
        curr_trainX = trainX[:training_size]
        curr_trainY = trainY[:training_size]

        learner = dtl.DTLearner(leaf_size = 1,verbose = False)
        start = time.time()
        learner.addEvidence(curr_trainX, curr_trainY)
        end = time.time()
        running_time = end - start
        running_time_dt.append(running_time)

        learner = rtl.RTLearner(leaf_size = 1,verbose = False)
        start = time.time()
        learner.addEvidence(curr_trainX, curr_trainY)
        end = time.time()
        running_time = end - start
        running_time_rt.append(running_time)
    
    print(running_time_dt)
    print(running_time_rt)
    xi = range(200, max_trainning_size + 1, 200)
    plt.plot(xi, running_time_dt, label = "Decision Tree")
    plt.plot(xi, running_time_rt, label = "Random Tree")
    plt.title("Figure 4 - Decision Tree vs Random Tree on trainning time")
    plt.xlabel("Trainning Sizes")
    plt.ylabel("Trainning Time/s")
    plt.xticks(np.arange(200, max_trainning_size + 1, step=400))
    plt.grid()
    plt.legend()
    plt.savefig("figure4.png")
    plt.clf()

    
    

# compare mean absolute error DT vs RT
def question_3_2(trainX, trainY, testX, testY):
    max_leaf_size = 30
    out_sample_mae_dt = []
    out_sample_mae_rt = []
        
    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size = each_leaf_size,verbose = False)
        learner.addEvidence(trainX, trainY)
        out_predY = learner.query(testX)
        out_sample_mae = np.mean(np.abs((np.asarray(testY) - np.asarray(out_predY))))
        out_sample_mae_dt.append(out_sample_mae * 100)
    
        learner = rtl.RTLearner(leaf_size = each_leaf_size,verbose = False)
        learner.addEvidence(trainX, trainY)
        out_predY = learner.query(testX)
        out_sample_mae = np.mean(np.abs((np.asarray(testY) - np.asarray(out_predY))))
        out_sample_mae_rt.append(out_sample_mae * 100)
    
    print(out_sample_mae_dt)
    print(out_sample_mae_rt)

    xi = range(1, max_leaf_size + 1)
    plt.plot(xi, out_sample_mae_dt, label = "Decision Tree")
    plt.plot(xi, out_sample_mae_rt, label = "Random Tree")
    plt.title("Figure 5 - Decision Tree vs Random Tree on MAE")
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error * 10^2")
    plt.xticks(np.arange(1, max_leaf_size + 1, step=5))
    plt.grid()
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()

# compare mean absolute error DT vs 7-Bagged RT
def question_3_3(trainX, trainY, testX, testY):
    max_leaf_size = 30
    out_sample_mae_bag = []
    out_sample_mae_dt = []
        
    for each_leaf_size in range(1, max_leaf_size + 1):
        learner = bl.BagLearner(learner=dtl.DTLearner,kwargs={"leaf_size":each_leaf_size},bags=7,boost=False,verbose=False)
        learner.addEvidence(trainX, trainY)
        out_predY = learner.query(testX)
        out_sample_mae = np.mean(np.abs((np.asarray(testY) - np.asarray(out_predY))))
        out_sample_mae_bag.append(out_sample_mae * 100)
    
        learner = dtl.DTLearner(leaf_size = each_leaf_size,verbose = False)
        learner.addEvidence(trainX, trainY)
        out_predY = learner.query(testX)
        out_sample_mae = np.mean(np.abs((np.asarray(testY) - np.asarray(out_predY))))
        out_sample_mae_dt.append(out_sample_mae * 100)
    
    print(out_sample_mae_dt)
    print(out_sample_mae_bag)

    xi = range(1, max_leaf_size + 1)
    plt.plot(xi, out_sample_mae_dt, label = "Decision Tree")
    plt.plot(xi, out_sample_mae_bag, label = "Bagged Random Tree")
    plt.title("Figure 6 - Decision Tree vs 7-Bagged Random Tree on MAE")
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error * 10^2")
    plt.xticks(np.arange(1, max_leaf_size + 1, step=5))
    plt.grid()
    plt.legend()
    plt.savefig("figure6.png")
    plt.clf()

# compare running time for building a tree DT vs 7-Bagged RT
def question_3_4(trainX, trainY):

    max_trainning_size = trainX.shape[0]
    running_time_dt = []
    running_time_bag = []


    for training_size in range(200, max_trainning_size + 1, 200):
        curr_trainX = trainX[:training_size]
        curr_trainY = trainY[:training_size]

        learner = dtl.DTLearner(leaf_size = 1,verbose = False)
        start = time.time()
        learner.addEvidence(curr_trainX, curr_trainY)
        end = time.time()
        running_time = end - start
        running_time_dt.append(running_time)

        learner = bl.BagLearner(learner=rtl.RTLearner,kwargs={"leaf_size":1},bags=7,boost=False,verbose=False)
        start = time.time()
        learner.addEvidence(curr_trainX, curr_trainY)
        end = time.time()
        running_time = end - start
        running_time_bag.append(running_time)
    
    print(running_time_dt)
    print(running_time_bag)
    xi = range(200, max_trainning_size + 1, 200)
    plt.plot(xi, running_time_dt, label = "Decision Tree")
    plt.plot(xi, running_time_bag, label = "7-Bagged Random Tree")
    plt.title("Figure 7 - Decision Tree vs 7-Bagged Random Tree on trainning time")
    plt.xlabel("Trainning Sizes")
    plt.ylabel("Trainning Time/s")
    plt.xticks(np.arange(200, max_trainning_size + 1, step=400))
    plt.grid()
    plt.legend()
    plt.savefig("figure7.png")
    plt.clf()



	   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		   	  			  	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		   	  			  	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		   	  			  	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])


    # change the leaner here!!!!!!!!
    # select the learner to test
    choose = 2
    shuffle_for_report = True
    np.random.seed(903329676)

    switcher = {
        1: lrl.LinRegLearner(verbose = True),
        2: dtl.DTLearner(verbose = True),
        3: rtl.RTLearner(verbose = True),
        4: bl.BagLearner(learner=rtl.RTLearner,kwargs={"leaf_size":1},bags=20,boost=False,verbose=False),
        5: il. InsaneLearner(verbose = False)
    }
    learner = switcher[choose]

    # deleted by Jie
    # data = np.array([list(map(float,s.strip().split(','))) for s in inf.readlines()])  	

    # added by Jie
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])

    # added by Jie
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:,1:]	

    # added by Jie
    data = data.astype('float')

    if shuffle_for_report:
        np.random.shuffle(data)	  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		   	  			  	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		   	  			  	 		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    trainY = data[:train_rows,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testY = data[train_rows:,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"{testX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"{testY.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		   	  			  	 		  		  		    	 		 		   		 		  
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner  		   	  			  	 		  		  		    	 		 		   		 		  
    learner.addEvidence(trainX, trainY) # train it  		   	  			  	 		  		  		    	 		 		   		 		  
    print(learner.author())  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")

    ### report

    # run on Istanbul.csv
    question_1(trainX, trainY, testX, testY)
    question_2(trainX, trainY, testX, testY)
    question_3_2(trainX, trainY, testX, testY)
    question_3_3(trainX, trainY, testX, testY)


    # run on winequality-white.csv
    question_3_1(trainX, trainY)
    question_3_4(trainX, trainY)



   	  			  	 		  		  		    	 		 		   		 		  
