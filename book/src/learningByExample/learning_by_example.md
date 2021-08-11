# Learning By Example

We will use an example to motivate the differences between forward and backwards AD. I think that understanding some of the notation here
will clearly help with the intuition of what AD is trying to achieve.
 
## A Motivating Example 

Let's start with an example of a function who's derivative we might want to take. Lets say we have a function of two variables \\( x_1 \\) and \\( x_2 \\). let's say we want to take the derivative of this function with respect to (w.r.t) \\( x_1 \\) and \\( x_2 \\). Let's define this function \\(y\\) as follows 

\\[ y = [sin(x_1/x_2) + x_1/x_2 - exp(x_2)] \times [x_1/x_2 - exp(x_2)] \\]

I'm going to introduce two useful ways to represent this problem. One as a **list of intermediate values** and another as a graph representation of such list. First we will introduce a set of intermediate variables \\( v_{i} \\) where \\( i \\) can be either a character or a number. Initial input values to our equation are subcripted with letters and all intermediate values are subscripted with numbers. 


1. \\( v_a = x_1 \\)
2. \\( v_b = x_2 \\)
3. \\( v_1 = v_a / v_b\\)
4. \\( v_2 = sin(v_1) \\)
5. \\( v_3 = exp(v_b) \\)
6. \\( v_4 = v_1 - v_3 \\)
7. \\( v_5 = v_2 + v_4 \\)
8. \\( v_6 = v_5 \times v_4 \\)
9. \\( y = v_6 \\)

*(Do note that the \\( \times \\) operator here means some sort of multiplication between scalars/matrices with other scalars/matrices. So no cross products)*

This list above is sometimes called an **execution trace** or a **wengert list** but the most important part is that it separates our function into a set of intermediate steps that makes it easier to reason about how to propagate derivatives. 

I think this property is better exemplefied by the graph representation of an execution trace.

<center><img src="images/Example Execution Graph.png">Graph of Execution Trace</center>

To properly understand what this graph means we first must define what each node means.
Each node in this graph primarily represents an **operator** of some sorts. Let's look at an example for node 5 to see what this means.


<center><img src="images/Sample Node V5.png">The node of the value V5 is the addition operator on two input variables</center>

Each node in our graph is some sort of elementary operation. This includes operators like addition, multiplication, or functions like 
trigonometric functions or exponential functions to name a few. A node takes an input depending on the type of function it is. In the case
of the node above:

- takes two inputs \\( v_{2} \\) and \\( v_{4} \\) and adds them together.
- knows how to take derivatives w.r.t \\( v_{2} \\) and \\( v_{4} \\)
- outputs an output value \\( v_{5} \\)
- output value can be used as a final value or an input to any other node

The powerful idea comes from the fact that each individual node only needs to know how to take the derivative with respect to its input. With this information the node can share the value of this derivative either forwards to children nodes that take its input \\( v_{5} \\) or to it's parent nodes, the nodes of \\( v_{2} \\) and \\( v_{4} \\). The direction in which this derivative information "flows" is what defines forward AD versus backwards AD. (Maybe it's this idea of "flow" is where TensorFlow got its name from but I haven't checked to make sure)

