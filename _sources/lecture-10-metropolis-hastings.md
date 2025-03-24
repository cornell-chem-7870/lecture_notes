

## The Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is a generalization of the Gibbs sampling algorithm.  In the Metropolis-Hastings algorithm, we propose a new state $x'$ from a proposal distribution $q(x' | x)$, and then accept the new state with probability

$$
\alpha(x, x') = \min\left(1, \frac{\pi(x') q(x | x')}{\pi(x) q(x' | x)}\right)
$$

If we accept the new state, we set $x = x'$, otherwise we keep the old state.  
The transition probability for Metropolis Hastings is given by 

$$
p(x' | x) = q(x' | x) \alpha(x, x') + \delta(x' - x) (1 - \int q(x' | x) \alpha(x, x') dx')
$$

where $\delta(x' - x)$ is delta function.

###  Detailed Balance

To show that the Metropolis-Hastings algorithm works, we first show that obeys a property called "detailed balance":

$$
\pi(x) p(x' | x)  = \pi(x') p(x | x')
$$

Any Markov chain that obeys detailed balance will have a stationary distribution that is equal to the target distribution,
as

$$
\int \pi(x) p(x' | x) dx = \int \pi(x') p(x | x') dx = \pi(x')
$$

To show that the Metropolis-Hastings algorithm obeys detailed balance, we consider two cases.  If $x' = x$, then the equation is trivially satisfied.  If $x' \neq x$, then the delta function is zero, and we have

$$
\pi(x) p(x' | x) =& \pi(x) q(x' | x) \alpha(x, x') 
    =  \min \left(\pi(x) q(x' | x), \pi(x') q(x | x')\right) = \pi(x') q(x | x') \alpha(x', x) = \pi(x') p(x | x')
$$

Consequently, the Metropolis-Hastings algorithm obeys detailed balance.



### Specific Examples of the Metropolis-Hastings Algorithm
(If we set $q(x' | x)$ to be the conditional distribution of the $i$'th variable given all the others, then the Metropolis-Hastings algorithm reduces to the Gibbs sampling algorithm.)