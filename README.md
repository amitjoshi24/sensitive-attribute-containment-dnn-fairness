# sensitive-attribute-containment-dnn-fairness
CS 394N final project on mitigating sensitive attributes propagating through the deep neural network and influencing the model output.
Authors: Amit Joshi(@amitjoshi24), Priyal Belgamwar (@priyal2506), Nikhil Ajjarapu(@nikhil-ajjarapu)

Steps to accomplish:

1) Create a basic Pytorch NN model with n layers + train recidivism dataset. 
2) Freeze weights of 0th - (n - 1) layers, copy into new model with random weights initalized for last layer, and train with recidivism dataset + race as labels. 
3) Use captum to look at neurons in the frozen layers (0 - n-1) to see which corresponds most with race. Use captum code here: https://github.com/TannerGilbert/Model-Interpretation/blob/master/Captum/Getting_started_with_Captum_Insights.ipynb. Select neurons that meet certain threshold/weighted average based on importance of neuron to final output.
4) Modify weights as mentioned above, then plug in modified weights and run inference on original recidivism dataset. Tradeoff between accuracy + correlation to sensitive attributes.
5) Use Amit's code to figure out whether modified weights are still correlated with sensitive attributes (measuring improvement in mitigating sensitive attribute leakage.
