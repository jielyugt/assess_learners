import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, bags, kwargs, boost, verbose):  		   	  			  	 		  		  		    	 		 		   		 		  
        
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))	 		  		  		    	 		 		   		 		  
 
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'jlyu31'
  		   	  			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self,dataX,dataY):	   	  			  	 		  		  		    	 		 		   		 		  
        for learner in self.learners:

            # without sampling, Istanbul.csv gives 0.999/0.821
            #learner.addEvidence(dataX, dataY)  		  		    	 		 		   		 		  

            # with sampling, Istanbul.csv gives 0.976/0.811
            bag_index = np.random.choice(range(dataX.shape[0]), dataX.shape[0], replace = True)
            bag_x = dataX[bag_index]
            bag_y = dataY[bag_index]
            learner.addEvidence(bag_x, bag_y)	

    def query(self,points):
        out = []		   	  			  	 		  		  		    	 		 		   		 		  
        for learner in self.learners:
            out.append(learner.query(points))
        return sum(out) / len(out)		  	 		  		  		    	 		 		   		 		  
 		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print('not implemented')