# Comparison of different neural network architectures for evolving reinforcement learning agents

Reinforcement learning is a type of learning algorithms in which an agent interacts with the environment and learns from the feedback it receives. It starts by performing random actions and then learns to behave by observing the state of the environment and receiving rewards. At each time step $t$ agent observes some state $s_{t}$ and chooses an action $a_{t}$. After that the state of the environment transitions to the next state $s_{t+1}$ and the agent receives a reward $r_{t+1}$. The main goal is to find a set of neural network parameters so that the agent receives the maximum possible reward. This can be thought of as a function that maps state-action pairs to corresponding rewards:

$$ f(s_{t}, a_{t}) = r_{t+1} $$

And the whole process trajectory can be represented as follows:

s_{0}, a_{0}, r_{1}, s_{1}, a_{1}, r_{2},..., s_{n-1}, a_{n-1}, r_{n}

The most common optimization strategies for artificial neural networks involve calculating weight gradients using backpropagation and minimizing the cost function. While such methods can be very effective for certain types of tasks, they can often be computationally inefficient and suffer from problems such as vanishing and exploding gradients \cite{salimans2017evolution}. An alternative approach is to use evolutionary strategies instead of gradient based methods for weight optimization.

 For our experiments we use **cart-pole** problem environment, but the method we demonstrate can be applied to other kinds of problems by implementing a few parameter adjustments. The task is to balance a pole that is hinged to a movable cart by applying forces to cart's base \cite{barto1983neuronlike}. Agent-environment interaction breaks up into subsequent episodes. Each episode terminates if one of the following conditions occurs: 1. Pole angle is greater than $\pm12^{\circ}$; 2. Center of the card reaches the edge of the display; 3. Episode length is greater than 500 ticks. Thus, a total reward of 500 would mean that we have reached the maximum possible result.
 
 FIGURE
 
 Our main goal is to examine the impact of neural network architecture on the performance of classical control tasks. We will look at how the use of different numbers of hidden layers and units can affect fitness and what rules should be followed when designing evolution-based optimization systems.
 
 ## Methods
 ### Agent
 An agent is a system controller represented by an artificial neural network. It has 4 input units corresponding to the number of observations and one output unit corresponding to the selected action. All weights and biases in a network are encoded as a single genotype. The agent's goal is to learn the mapping between observations and corresponding actions by learning the correct set of weights and biases. Every time we receive observations from the environment, we do a forward pass through the network and get an action. We then optimize our weights and continue this process until we get the best possible fitness. We are using a basic single layer perceptron as the initial setup, so the output of the last layer is given as follows:
 
 $$ &a = H\left(\sum_{i}^Nw_{i}x_{i}+b\right) $$
 
 Where $H(x)$ is a *heaviside step function*.
 
Assigning a gene sequence to a set of network coefficients is pretty intuitive, we just need to make sure that the genotype length strictly matches the number of weights and biases in a fully connected network. It is worth noting that in our experiments we use the LeakyReLU function for hidden layers, however, some studies \cite{jang2019empirical} show that there is no single winning solution, and the effectiveness of the activation function is highly dependent on the type of problem.

 ### Mutation Function

The way we do the mutation has a great impact on the optimization results. In canonical evolutionary algorithms, mutations are usually based on noise from a Gaussian distribution \cite{ajani2022adaptive}, but other approaches have also been studied. One of them is a mutation based on the Cauchy noise. One of its main advantages is that it often leads to faster convergence compared to the Gaussian method. \cite{lee2014effect}. Fig. \ref{fig:mutation} shows the effect of various mutation function parameters on mean fitness of the controller

FIGURE

We see that Cauchy-based mutation shows the fastest convergence comparing to other parameters and the worst result is achieved by Gaussian-based mutation with a standard deviation of $0.01$. It can be assumed that the use of noise from the Cauchy distribution will lead to faster convergence and increase the computational efficiency of our algorithm. 

### Selection

Another important part of our algorithm is how we do the selection. This imitates the process of natural selection in nature, when the strongest individuals remain in the population, and the weak ones are eliminated. In our case, the goal is to keep the mutation in the gene if it results in better fitness, and remove it if it doesn't. To make this choice, we need to track the previous state of the population and its fitness. We then compare the fitness of the gene in both populations and determine if the mutation was beneficial.

FIGURE Algorithm1 screenshot

### Training Algorithm

The optimization is performed by the hillclimbing algorithm. Given the number of generations, at each iteration we apply the mutation to a population of potential solutions (“genotypes”) and evaluate their fitness. We then leave the mutation for those genotypes where it was beneficial, and ``undo'' it for the rest of the population using selection. It is worth noting that for this task, we know that we have reached the global optimum, since the best result corresponds to the maximum possible duration of the episode. This allows us to evaluate how good a particular solution is compared to the best possible outcome. The number of iterations corresponds to the number of episodes. This means that in each iteration we run one episode and measure the total reward.

One of the main advantages of using this approach to tune neural networks rather than backpropagation is that evolutionary algorithms are insensitive to initial weight values, which can affect the quality of the trained network. 

FIGURE Algorithm2 screenshot

## Experimental Results

In our experiments, we use a basic single layer network to demonstrate how the controller evolves as the number of epochs increases. Fig. \ref{fig:max_fit} shows how the maximum fitness grows for multiple runs of the algorithm. The fluctuations are caused by the stochastic choice of the initial state, which means that the position of the cart is chosen randomly each time we reset the environment. Hence, we can sometimes get different fitness for the same solution. It can be seen that our algorithm is evolving and it takes it less than 50 epochs to achieve the highest result for all runs. There is also a \href{https://youtu.be/f2d3wG2FDGk}{short video demonstration} of the optimization process.

FIGURE

We then compare different neural network architectures and measure how the average performance changes. Following the principle of Occam's Razor, we don't want to complicate our model too much, so we use a small number of hidden layers and units. For our experiments, we consider the following network architectures:


1. Single layer
2. 1 hidden layer with 2 units
3. 1 hidden layer with 4 units
4. 2 hidden layers with 2 and 2 units
5.2 hidden layers with 4 and 2 units

Table \ref{table:arch} demonstrates the results of applying each type of network to our problem. We ran each of them ten times and compared the average behavior after different number of epochs. It can be seen that for each type of network, fitness gradually increases and all networks achieve the highest possible result after 80 epochs. We see that the network with two hidden layers converges more slowly than the others. However, a network with one hidden layer outperforms a single layer network in terms of convergence rate. Based on these results, we can assume that a network with one hidden layer is the most appropriate type of architecture for this task.

FIGURE table screenshot

If we plot the behavior of our controller, we can see that the pole angle stays around zero for the duration of the episode, which means that the optimization was successful.

FIGURE

To measure the accuracy of our controller, we will run it 100 times and evaluate the percentage of successful episodes. Each time we use 50 epochs for training. We see that although the algorithm performs well for one run, it may fail to generalize to other runs. Fig. \ref{fig:acc} shows the difference in accuracy between results obtained with the same algorithm.  It can be seen that one solution gives the best result in most cases, while the other fails in about 40\% of the time. This tells us that it's a good idea to evaluate the best solution in the long run to see how well it performs with different initial environment states.

FIGURE

## Discussion

We discussed how traditional gradient-based learning approaches for training reinforcement learning controllers can be replaced by evolutionary optimization methods. By not requiring backpropagation, they allow to reduce computational and memory resources, and protect against problems such as exploding gradients. We have demonstrated the impact of various neural network structures on the accuracy of the final result and the rate of convergence. We have shown that for simple tasks it is more useful to give preference to more simple architectures with a single hidden layer. We have also shown that choosing the right mutation function can reduce the number of epochs used to train our optimizer. However, our algorithm can be improved by combining the advantages of different approaches. First, although the Cauchy distribution performs very well for points far from the global optimum, Gaussian-based mutation is often the best method for points close to it. More intelligent solution to  to capitalize on both approaches is to develop an adaptive evolution strategy \cite{ajani2022adaptive}. Second, as mentioned above, evolutionary strategies are less sensitive to initial weights than gradient-based methods. Thus, we may consider combining the two approaches so that evolutionary algorithms are used to find the initial weights and supervised methods are introduced at a later stage to optimize them. Some studies indicate that such networks can be trained significantly faster and better 

## Conclusion

In this report, we reviewed evolutionary methods for finding the best parameters for a neural network in a reinforcement learning problem and showed the impact of network architecture on convergence rate and accuracy of predictions. This approach is a robust alternative to gradient descent and similar types of optimization, but it is highly dependent on design choices such as mutation function and neural network hyperparameters.
