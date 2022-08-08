# nnLACTOSE
Neural Networks based on the Linear Array of Conditions for Topologies with Separated Error-backpropagation (LACTOSE) Algorithm
## Description
This algorithm presents a method to side-step the potential difficulties with providing symbolic inputs to conditional branches in the [Tensorflow](https://www.tensorflow.org/)(TF) graph. The algorithm works with implementing the condition outside of the TF graph so that the smoothness of the model architecture is not affected. This is demonstrated by the image below.    

![Model.png](/imgs/Model.png "San Juan Mountains")

In this diagram, the model may contain any (differentiable) layer. As the conditions are calculated at run time, the TF graph is "compiled" normally. 

At each step
<pre><code>
1) Condition(Input, ModelWeights1 ... ModelWeightsN)
    return Input, ModelWeight

With tf.gradient_tape as tape:
2) Model.load_weights(ModelWeight)
3) Output = Model(Input)
4) grads = tape. etc. etc. etc.
</code></pre>

This way, assuming a batch size of 1, epoch size of 1, the model is always fully differentiable, and has non-zero gradient at all points, subject to the model architecture.
