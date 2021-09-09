# Forward AD

<!-- toc -->


Forward AD is the more straightforward implementation of the two and I believe can really help someone with the intuition that makes backward AD make a bit more sense. So let's get started by again looking at our motivating example. 

We want to take the derivative of the following function:

\\[ y = [sin(x_1/x_2) + x_1/x_2 - exp(x_2)] \times [x_1/x_2 - exp(x_2)] \\]

Let's say that we want to calculate the derivative when \\( x_1 = 1.5 \\) and \\( x_2 = 2 \\)

## Execution Trace Approach

Again looking at the execution trace of our program we get the following 

1. \\( v_a = x_1 \\)
2. \\( v_b = x_2 \\)
3. \\( v_1 = v_a / v_b\\)
4. \\( v_2 = sin(v_1) \\)
5. \\( v_3 = exp(v_b) \\)
6. \\( v_4 = v_1 - v_3 \\)
7. \\( v_5 = v_2 + v_4 \\)
8. \\( v_6 = v_5 \times v_4 \\)
9. \\( y = v_6 \\)

With forward mode AD we now create another execution trace **for each input** into our function. For this example let's take the input \\( x_1 \\) or the node that outputs \\( v_a \\). In this execution trace, we first define the derivative of \\( v_a \\), \\( \dot{v_a} = 1\\). (I'm using the small dot above v1 to denote its derivative with respect to x_1). Since all the other input's don't have a \\( x_1 \\) then they are set to 0. 

With all subsequent expressions we pass along the previously calculated derivatives. For each variable we take the derivative with respect to its input, and to find the derivative of the variable w.r.t \\( x_1 \\) we use the derivatives that have been already calculted       . Below is the full trace for \\( x_1 \\)

1. \\( \dot{v_a} = 1 \\) 
2. \\( \dot{v_b} = 0 \\)
3. \\( v_1 = v_a / v_b\\)   
    - \\( \dot{v_1} = (v_b \dot{v_a} - v_a \dot{v_b})  / v_b^2\\)
4. \\( v_2 = sin(v_1) \\) 
    - \\( \dot{v_2} = cos(v_1) \times \dot{v1} \\)
5. \\( v_3 = exp(v_b) \\)
    - \\( \dot{v_3} = exp(v_b) \times \dot{v_b} \\)
6. \\( v_4 = v_1 - v_3 \\)
    - \\( \dot{v_4} = 1 \times \dot{v_1} - 1 \times \dot{v_3} \\)
7. \\( v_5 = v_2 + v_4 \\)
    - \\( \dot{v_5} = 1 \times \dot{v_2} + 1 \times \dot{v_4} \\)
8. \\( v_6 = v_5 \times v_4 \\)
    - \\( v_6 = (v_5 \times \dot{v4}) \times (v_4 \times \dot{v_5}) \\)
9. \\( y = v_6 \\)
    - \\( \dot{y} = \dot{v_6} \\)


From this we can see from the execution trace where our derivatives come from and how calculating derivatives node by node help save some time.



## Execution Graph Approach

For looking at the execution graph we will first look at how specific subsets of the graph works. Beginning at the input nodes or nodes who **instantiate**/create a computational graph.
In forward AD, input nodes to a computational graph receive two things.

1. Each nodes initial values
2. Each nodes derivative with respect to one node's values. Generally One input node is set to 1 and all other's to 0.



<center><img src="images/forward_mode/Forward Mode X1-_Va Instantiation.png"></center>

In the example above, the graph is going to take the derivative of the example equation w.r.t \\( x_1 \\). \\( {v_a}  \\)  and \\( {v_b}  \\)  are instantiated with the values \\( {x_1}  \\)  and \\( {x_2}  \\)  respectively. With \\( {v_a} \\)  receiving a derivative value of 1 and  \\( v_b \\) receiving a derivative value of 0. 

When it comes to the outputs of each node

1. They each output an output value \\( v_a \\) and \\( v_b \\)
2. They each have an output **derivative** value \\( \dot{v_a} \\) and \\( \dot{v_b} \\)

It may seem that this is overly complicating a simple calculation but looking at a subsequent child node
in the graph helps us learn why one does this. Looking at the node for \\( v_1 \\) we see 


<center><img src="images/forward_mode/Flow of Derivatives of V1.png"></center>

1. The inputs to \\( v_1 \\) are in terms of v1 and v2
2. The derivative of \\( v_1 \\), \\( \dot{v_1} \\) , is written purely as a derivative based on the input derivatives and input values \\( v_1 \\) \\( v_2 \\) \\( \dot{v_1} \\)\\( \dot{v_1} \\)
3. These derivatives could be with respect to any variable but the node does not need to know that

Numbers 2 and 3 are the most powerful parts of forward AD. The node does not need to know "what" the input derivatives themselves mean, just their values and how to take the derivative of it's own operation via the chain rule. From there one can keep adding new nodes from the outputs of any previous nodes and build a graph which automatically propagates values and their derivatives. What these derivatives mean only matters when one first instantiates the first nodes and is left up to the user to decide.

Creating a full computational graph of the examples in forward AD would look like the following.

<center><img src="images/forward_mode/Forward AD  Full Graph.png">Computational graph with forward AD derivative flow</center>

The derivatives "flow" forward along with the computation of the output y. By the time the output is calculated so is it's derivative with respect to one of the instantiated variables.


## Considerations to Implementing an Operation Node for AD

When implementing a forward AD compuation library one needs to keep the following in mind.

1. Each node needs to know what operation it does
2. Each node needs to know how to take the derivative along with the chain rule with respect to it's own operation. 

Forward AD lends itself to being implemented relatively well by a method called "operator overloading"


## Further Reading: Computing Directional Derivatives 

What if our function outputs two variables? 

TODO