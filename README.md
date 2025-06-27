The point of this repo is to illustrate the key concepts of JAX MD, not to build a performant or production-ready library. 

The main idea of JAXMD:

1. it all starts with an energy function
2. We then take a derivative of the energy function to get the force; jax does this for us
3. We then use the force to update the positions

Other components
1. Spaces help us abstracting the update positions and also calculate distances which can be useful for the energy function
2. Partition helps update neighbor lists to make the energy calculations faster
3. Minimize can help with some initialization. 
4. Simulate tells us how to use the gradients to update the system