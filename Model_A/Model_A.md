Model_A is a probabilstic model designed to classify probabilities for BTC spot prices around a strike, K.

Pipeline: 

 

Download matching timeframe data. 

Estimate GARCH using (1,1) window. 

Use Student-T distribution to generate progressive step (return t+1). 

Recompute new GARCH based on step. 

Build a full path to expiry (60-t steps). 

Monte-Carlo with sufficient trials and use endpoints to classify probability distributions around K. 




The model analyzes each minute and generates probabilities using the given method.
