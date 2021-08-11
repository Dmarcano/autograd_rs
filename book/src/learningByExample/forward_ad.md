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

With all subsequent expressions we pass along the previously calculated derivatives. For each node we take the derivative with respect to its input, and to find the derivative w.r.t \\( x_1 \\) we use the previously calculated derivative. Below is the full trace for \\( x_1 \\)

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



## Considerations to Implementing an Operation Node for AD