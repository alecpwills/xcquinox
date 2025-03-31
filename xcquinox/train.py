import equinox as eqx
import optax
import sys
import gc
import jax
import os
from jax.interpreters import xla
import jax.numpy as jnp
from typing import Callable


class xcTrainer(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    loss: eqx.Module
    steps: int
    print_every: int
    clear_every: int
    memory_profile: bool
    verbose: bool
    do_jit: bool
    opt_state: tuple
    serialize_every: int
    logfile: str
    loss_v: float

    def __init__(self, model, optim, loss, steps=50, print_every=1, clear_every=1, memory_profile=False, verbose=False, do_jit=True, serialize_every=1, logfile=''):
        '''
        The base xcTrainer class, whose forward pass computes the training loop.

        :param model: The network which will be trained
        :type model: xcquinox.xc.eXC
        :param optim: optax optimizer object, e.g. optax.adamw(1e-4)
        :type optim: optax.GradientTransformation
        :param loss: The loss function object that computes the loss to be trained against.
        :type loss: eqx.Module
        :param steps: Length of the training cycle, i.e. the number of epochs, defaults to 50
        :type steps: int, optional
        :param print_every: The number of epochs between loss information printing, defaults to 1
        :type print_every: int, optional
        :param clear_every: The number of epochs between calls to clear the cache, defaults to 1
        :type clear_every: int, optional
        :param memory_profile: If True, will write memory profiles at every epoch, to be used with `pprof`, defaults to False
        :type memory_profile: bool, optional
        :param verbose: If true, will print various extra information during the training cycle, defaults to False
        :type verbose: bool, optional
        :param do_jit: Controls whether the update function is jitted or not, useful for debugging if False, defaults to True
        :type do_jit: bool, optional
        :param serialize_every: Controls how often the checkpoint network is written to disk, defaults to 1
        :type serialize_every: int, optional
        '''
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss = loss
        self.steps = steps
        self.print_every = print_every
        self.clear_every = clear_every
        self.memory_profile = memory_profile
        self.verbose = verbose
        self.do_jit = do_jit
        self.serialize_every = serialize_every
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.logfile = logfile
        self.loss_v = 0

    def __call__(self, epoch_batch_len, model, *loss_input_lists):
        '''
        Forward pass of the xcTrainer object, which goes through the training cycle.

        *loss_input_lists are positional arguments, each a list of length [epoch_batch_len elements], corresponding to the proper input order and values for the self.loss function
        I.e., for the E_loss object, these would be [density matrix list], [reference energy list], [ao_eval list], [grid_weight list]

        .. todo: Remove model from inputs, not used since retrieved from self at first iteration

        :param epoch_batch_len: The number of batches in a given epoch (i.e., the number of molecules one is training on that are looped over)
        :type epoch_batch_len: int
        :param model: The baseline model to update in the training process
        :type model: xcquinox.xc.eXC
        :return: The updated model after the training cycle completes
        :rtype: xcquinox.xc.eXC
        '''
        BEST_LOSS = 1e10
        for step in range(self.steps):
            jax.debug.print('Epoch {}'.format(step))
            epoch_loss = 0
            if step == 0 and self.logfile:
                with open(self.logfile+'.dat', 'w') as f:
                    f.write('#Epoch\tLoss\tBest\n')
                with open(self.logfile+'batch.dat', 'w') as f:
                    f.write('#Epoch\tBatch\tLoss\tBest\n')
            if step == 0 and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            elif ((step % self.clear_every) == 0) and (step > 0) and self.do_jit:
                fmake_step = eqx.filter_jit(self.make_step)
            else:
                fmake_step = self.make_step
            if step == 0:
                print('Step = 0: initializing inp_model and inp_opt_state.')
                inp_model = self.model
                start_model = self.model
                inp_opt_state = self.opt_state
            for idx in range(epoch_batch_len):
                jax.debug.print('Epoch {} :: Batch {}/{}'.format(step, idx+1, epoch_batch_len))

                # loops over every iterable in loss_input_lists, selecting one batch's input data
                # assumes separate lists, each having inputs for multiple cases in the training set
                loss_inputs = [inp[idx] for inp in loss_input_lists]

                # this_loss = self.loss(inp_model, *loss_inputs).item()
                inp_model, inp_opt_state, this_loss = fmake_step(inp_model, inp_opt_state, *loss_inputs)

                if self.memory_profile:
                    this_loss.block_until_ready()
                    jax.profiler.save_device_memory_profile(f"memory{step}_{idx}.prof")

                jax.debug.print('Batch Loss = {}'.format(this_loss))
                if self.logfile:
                    with open(self.logfile+'batch.dat', 'a') as f:
                        f.write(f'{step}\t{idx}\t{this_loss}\t{BEST_LOSS}\n')
                epoch_loss += this_loss
                if ((step % self.clear_every) == 0) and ((step > 0) == 0):
                    eqx.clear_caches()
                    jax.clear_backends()
                    jax.clear_caches()
            # update self.loss_v to epoch's loss
            object.__setattr__(self, 'loss_v', epoch_loss.item())

            if ((step % self.serialize_every) == 0) and (epoch_loss.item() < BEST_LOSS):
                eqx.tree_serialise_leaves('xc.eqx.{}'.format(step), start_model)
                BEST_LOSS = epoch_loss.item()

            # this will persist until next pass
            # the inp_model output just now is fed into get the loss next time, so if better we want to save this
            # not the updated one.
            start_model = inp_model

            if ((step % self.print_every) == 0) or (step == self.steps - 1):
                jax.debug.print(
                    f"{step}, epoch_train_loss={epoch_loss}"
                )
            if ((step % self.clear_every) == 0) and (step > 0):
                eqx.clear_caches()
                jax.clear_backends()
                jax.clear_caches()

            if self.logfile:
                with open(self.logfile+'.dat', 'a') as f:
                    f.write(f'{step}\t{epoch_loss}\t{BEST_LOSS}\n')

        return inp_model

    def make_step(self, model, opt_state, *args):
        '''
        The update step for the training cycle.

        *args input are the inputs to the self.loss function.

        :param model: The model whose weights and biases are to be updated given the loss in self.loss
        :type model: xcquinox.xc.eXC
        :param opt_state: The state of the optimizer to drive the update
        :type opt_state: result of optim.update
        :return: The updated model
        :rtype: xcquinox.xc.eXC
        '''
        self.vprint('loss_value, grads')
        loss_value, grads = eqx.filter_value_and_grad(self.loss)(model, *args)
        self.vprint('updates, opt_state')
        updates, opt_state = self.optim.update(grads, opt_state, model)
        self.vprint('model update')
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # def __post_init__(self, attr, value):
    #     object.__setattr__(self, attr, value)

    def clear_caches(self):
        '''
        A function that attempts to clear memory associated to jax caching
        '''
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                if module_name not in ["jax.interpreters.partial_eval"]:
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if hasattr(obj, "cache_clear"):
                            try:
                                obj.cache_clear()
                            except:
                                pass
        gc.collect()

    def vprint(self, output):
        '''
        Custom print function. If self.verbose, will print the called output.

        :param output: The string or value to be printed.
        :type output: printable object
        '''
        if self.verbose:
            jax.debug.print(output)


# Pre-trainer
class Pretrainer(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    steps: int
    print_every: int
    opt_state: tuple
    inputs: jnp.array
    ref: jnp.array
    loss: Callable

    def __init__(self, model, optim, inputs, ref, loss, steps=1000, print_every=100):
        '''
        The Pretrainer object aids in the initial pre-training of enhancement factor networks to have a more physical starting point for further network optimization. This class is meant to pre-train a randomly initialized network to fit the values of a specific XC functional's enhancement factor (either X or C, in principle it could also be a combined XC enhancement facator)

        :param model: The enhancement factor network to be pre-trained
        :type model: :xcquinox.net: class
        :param optim: The optimizer than will control the weight updates given a loss and gradient
        :type optim: optax.GradientTransformation
        :param inputs: The inputs the network itself is expecting in its forward pass function
        :type inputs: jnp.array
        :param ref: The reference values the network is expected to reproduce
        :type ref: jnp.array
        :param loss: A function from :xcquinox.loss: that is decorated with @eqx.filter_value_and_grad
        :type loss: Callable
        :param steps: Number of epochs to train over, defaults to 1000
        :type steps: int, optional
        :param print_every: How often to print loss statistic, defaults to 100
        :type print_every: int, optional
        '''
        super().__init__()
        self.model = model
        self.optim = optim
        self.inputs = inputs
        self.ref = ref
        self.steps = steps
        self.print_every = print_every
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.loss = loss

    def __call__(self):
        '''
        The training loop itself. Here, a loop over the specifed epochs takes place to train the network to fit reference values.

        :return: The trained model and an array of the losses during training.
        :rtype: (:xcquinox.net: class, array)
        '''
        losses = []
        for epoch in range(self.steps):
            if epoch == 0:
                this_model = self.model
                this_opt_state = self.opt_state
            loss, this_model, this_opt_state = self.make_step(this_model, self.inputs, self.ref, this_opt_state)
            lossi = loss.item()
            losses.append(lossi)
            if epoch % self.print_every == 0:
                print(f'Epoch {epoch}: Loss = {lossi}')

        return this_model, losses

    @eqx.filter_jit
    def make_step(self, model, inputs, ref, opt_state):
        '''
        The function that does each epoch's network update. It generates a loss and gradient using the specific :xcquinox.loss: function (that must be decorated with @eqx.filter_value_and_grad and only explicitly returns the loss value inside the function proper) given the specified inputs and reference values and initial optimization state.

        :param model: The enhancement factor network to be pre-trained
        :type model: :xcquinox.net: class
        :param inputs: The inputs the network itself is expecting in its forward pass function
        :type inputs: jnp.array
        :param ref: The reference values the network is expected to reproduce
        :type ref: jnp.array
        :param opt_state: The INITIAL optimization state to work against, typically generated via :self.optim.init(eqx.filter(self.model, eqx.is_array)):
        :type opt_state: The type of the above
        :return: The loss value for this step, the updated model after that loss is calculated, and the new optimization state for this step to use next time
        :rtype: tuple
        '''
        loss, grad = self.loss(model, inputs, ref)
        updates, opt_state = self.optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state


# Optimizer
class Optimizer(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    steps: int
    print_every: int
    opt_state: tuple
    mols: list
    refs: jnp.array
    loss: Callable

    def __init__(self, model, optim, mols, refs, loss, steps=1000, print_every=100):
        '''
        The Pretrainer object aids in the initial pre-training of enhancement factor networks to have a more physical starting point for further network optimization. This class is meant to pre-train a randomly initialized network to fit the values of a specific XC functional's enhancement factor (either X or C, in principle it could also be a combined XC enhancement facator)

        :param model: The enhancement factor network to be pre-trained
        :type model: :xcquinox.net: class
        :param optim: The optimizer than will control the weight updates given a loss and gradient
        :type optim: optax.GradientTransformation
        :param inputs: The inputs the network itself is expecting in its forward pass function
        :type inputs: jnp.array
        :param ref: The reference values the network is expected to reproduce
        :type ref: jnp.array
        :param loss: A function from :xcquinox.loss: that is decorated with @eqx.filter_value_and_grad
        :type loss: Callable
        :param steps: Number of epochs to train over, defaults to 1000
        :type steps: int, optional
        :param print_every: How often to print loss statistic, defaults to 100
        :type print_every: int, optional
        '''
        super().__init__()
        self.model = model
        self.optim = optim
        self.mols = mols
        self.refs = refs
        self.steps = steps
        self.print_every = print_every
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.loss = loss
    # S: x probar
    # @eqx.filter_jit

    def __call__(self):
        '''
        The training loop itself. Here, a loop over the specifed epochs takes place to train the network to fit reference values.

        :return: The trained model and an array of the losses during training.
        :rtype: (:xcquinox.net: class, array)
        '''
        losses = []
        for epoch in range(self.steps):
            if epoch == 0:
                this_model = self.model
                this_opt_state = self.opt_state
            loss, this_model, this_opt_state = self.make_step(this_model, self.mols, self.refs, this_opt_state)
            lossi = loss.item()
            losses.append(lossi)
            if epoch % self.print_every == 0:
                print(f'Epoch {epoch}: Loss = {lossi}')

        return this_model, losses

    # @eqx.filter_jit
    def make_step(self, model, inputs, ref, opt_state):
        '''
        The function that does each epoch's network update. It generates a loss and gradient using the specific :xcquinox.loss: function (that must be decorated with @eqx.filter_value_and_grad and only explicitly returns the loss value inside the function proper) given the specified inputs and reference values and initial optimization state.

        :param model: The enhancement factor network to be pre-trained
        :type model: :xcquinox.net: class
        :param inputs: The inputs the network itself is expecting in its forward pass function
        :type inputs: jnp.array
        :param ref: The reference values the network is expected to reproduce
        :type ref: jnp.array
        :param opt_state: The INITIAL optimization state to work against, typically generated via :self.optim.init(eqx.filter(self.model, eqx.is_array)):
        :type opt_state: The type of the above
        :return: The loss value for this step, the updated model after that loss is calculated, and the new optimization state for this step to use next time
        :rtype: tuple
        '''
        loss, grad = self.loss(model, self.mols, self.refs)
        updates, opt_state = self.optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
