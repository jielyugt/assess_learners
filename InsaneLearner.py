import BagLearner as bl
import LinRegLearner as lrl	   	  			  	 		  		  		    	 		 		   		 		  	 		  		  		    	 		 		   		 		  
class InsaneLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False):
        self.learners = [bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)] * 20
    def author(self):
        return 'jlyu31'
    def addEvidence(self,dataX,dataY):
        for learner in self.learners:
            learner.addEvidence(dataX, dataY)	  	 		  		  		    	 		 		   		 		  
    def query(self,points):
        out = []
        for learner in self.learners:
            out.append(learner.query(points))
            return sum(out) / len(out)

if __name__=="__main__":
    print('not implemented')