import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size = 1, verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        self.verbose = verbose 		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'jlyu31'		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self,dataX,dataY):		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Add training data to learner  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataX: X values of data to add  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataY: the Y training values  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  

        self.tree = self.build_tree(dataX, dataY)
        if self.verbose:
            print("tree shape: " + str(self.tree.shape))
            print("tree details below")
            print(self.tree)
  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,points):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Estimate a set of test points given the model we built.  		   	  			  	 		  		  		    	 		 		   		 		  
        @param points: should be a numpy array with each row corresponding to a specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @return: the estimated values according to the saved model.  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        out = []
        for point in points:
            out.append(self.get_prediction(point))
        return np.asarray(out)

    def get_prediction(self, point):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Predict one query using self.tree  		   	  			  	 		  		  		    	 		 		   		 		  
        @param point: numpy ndarray, one specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @return: the prediction for the one query	   	  			  	 		  		  		    	 		 		   		 		  
        """

        node = 0
        while ~np.isnan(self.tree[node][0]):
            split_value = point[int(self.tree[node][0])]

            # relative position so new_node = curr_node + offset
            if split_value <= self.tree[node][1]:
                node += int(self.tree[node][2])
            else:
                node += int(self.tree[node][3])
        return self.tree[node][1]
    
    def get_best_feature(self, dataX, dataY):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: determine the best feature to split on  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataX: numpy ndarray, features of trainning data
        @param dataY: numpy ndarray, labels of tranning data		   	  			  	 		  		  		    	 		 		   		 		    			  	 		  		  		    	 		 		   		 		  
        @return: the index of the feature in dataX with the highest correlation with dataY	   	  			  	 		  		  		    	 		 		   		 		  
        """  

        best_feature_index = 0
        best_correlation = np.corrcoef(dataX[:,0], dataY)[0,1]

        for i in range(1, dataX.shape[1]):
            correlation = np.corrcoef(dataX[:,i], dataY)[0,1]
            if correlation > best_correlation:
                best_correlation = correlation
                best_feature_index = i
        return best_feature_index


    def build_tree(self, dataX, dataY):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: build the decision tree  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataX: numpy ndarray, features of trainning data
        @param dataY: numpy ndarray, labels of tranning data		   	  			  	 		  		  		    	 		 		   		 		    			  	 		  		  		    	 		 		   		 		  
        @return: numpy ndarray, decision tree in tabular format	   	  			  	 		  		  		    	 		 		   		 		  
        """

        
        # print(dataY.shape)
        # print(self.leaf_size)
        # print(dataY)
        # print(dataX.shape)

        # aggregated all the data left into a leaf if leaf_size or fewer entries left
        if dataX.shape[0] <= self.leaf_size or dataX.shape[1] == 0:
            return np.asarray([np.nan, np.mean(dataY), np.nan, np.nan])
        
        if np.all(np.isclose(dataY,dataY[0])):
            return np.asarray([np.nan, dataY[0], np.nan, np.nan])

        feature_index = self.get_best_feature(dataX, dataY)
        split_val = np.median(dataX[:,feature_index])

        left_mask = dataX[:,feature_index] <= split_val
        right_mask = np.logical_not(left_mask)

        # delete the column
        dataX = np.delete(dataX, feature_index, 1)

        left_tree = self.build_tree(dataX[left_mask], dataY[left_mask])

        # print("++++++++")
        # print(left_tree)
        # print(left_tree.shape)

        right_tree = self.build_tree(dataX[right_mask], dataY[right_mask])

        if left_tree.ndim == 1:
            root = np.asarray([feature_index, split_val, 1, 2])
        else:
            root = np.asarray([feature_index, split_val, 1, left_tree.shape[0] + 1])


        return np.row_stack((root, left_tree, right_tree))


  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print('not implemented')  	