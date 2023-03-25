import torch
import numpy as np
import copy
import collections
import torch.nn.functional as F

def influence_function(model,
                       train_index,
                       W=None,
                       mode="stochastic",
                       batch_size=100,
                       damp=1e-3,
                       scale=1000,
                       order=1,
                       recursion_depth=1000):
    """
    Computes the influence function defined as H^-1 dLoss/d theta. This is the impact that each
    training data point has on the learned model parameters.
    """
    IF = exact_influence(model, train_index, order)
    # if mode == "stochastic":
    #     IF = influence_stochastic_estimation(model, train_index, batch_size, damp, scale, recursion_depth)

    return IF


def influence_stochastic_estimation(model,
                                    train_index,
                                    batch_size=100,
                                    damp=1e-3,
                                    scale=1000,
                                    recursion_depth=1000):
    """
    This function applies the stochastic estimation approach to evaluating influence function based on the power-series
    approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:

    H^-1 = \sum^\infty_{i=0} (I - H) ^ j

    This series converges if all the eigen values of H are less than 1.


    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        rnn: the model for which the Hessian of the loss is evaluated
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """

    NUM_SAMPLES = model.X.shape[0]
    SUBSAMPLES = batch_size

    loss = [model.loss_fn(model.y[train_index[_]], model.predict(model.X[train_index[_], :], numpy_output=False)) for _
            in range(len(train_index))]

    grads = [stack_torch_tensors(torch.autograd.grad(loss[_], model.parameters(), create_graph=True)) for _ in
             range(len(train_index))]

    IHVP_ = [grads[_].clone().detach() for _ in range(len(train_index))]

    for j in range(recursion_depth):
        sampled_indx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES, replace=False)

        sampled_loss = model.loss_fn(model.y[sampled_indx], model.predict(model.X[sampled_indx, :], numpy_output=False))

        IHVP_prev = [IHVP_[_].clone().detach() for _ in range(len(train_index))]

        hvps_ = [stack_torch_tensors(hessian_vector_product(sampled_loss, model, [IHVP_prev[_]])) for _ in
                 range(len(train_index))]

        IHVP_ = [g_ + (1 - damp) * ihvp_ - hvp_ / scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)]

    return [-1 * IHVP_[_] / (scale * NUM_SAMPLES) for _ in range(len(train_index))]


def exact_influence(model, train_index, damp=0, order=1):
    params_ = []

    for param in model.parameters():
        params_.append(param)

    num_par = stack_torch_tensors(params_).shape[0]
    tmp = torch.eye(num_par)
    if model.args.cuda:
        tmp = tmp.cuda()
        train_index = train_index.cuda()
    Hinv = torch.inverse(exact_hessian(model) + damp * tmp)

    losses = []
    for k in train_index:
        tmp = torch.unsqueeze(k, 0)
        losses.append(F.nll_loss(model.predict(tmp), model.labels[tmp]))

        n_factor = model.features.shape[0]
    grads = [stack_torch_tensors(torch.autograd.grad(losses[k], model.parameters(), create_graph=True)) for k in
             range(len(losses))]

    if order == 1:

        IFs_ = [-1 * torch.mm(Hinv, grads[k].reshape((grads[k].shape[0], 1))) / n_factor for k in range(len(grads))]


    return IFs_


def stack_torch_tensors(input_tensors):
    '''
    Takes a list of tensors and stacks them into one tensor
    '''

    unrolled = [input_tensors[k].view(-1, 1) for k in range(len(input_tensors))]

    return torch.cat(unrolled)


def get_numpy_parameters(model):
    params = []

    for param in model.parameters():
        params.append(param)

    return stack_torch_tensors(params).detach().numpy()


def exact_hessian(model):
    grad_params = torch.autograd.grad(model.loss, model.parameters(), retain_graph=True, create_graph=True)
    grad_params = stack_torch_tensors(grad_params)
    hess_params = torch.zeros((len(grad_params), len(grad_params)))
    temp = []

    for u in range(len(grad_params)):
        second_grad = torch.autograd.grad(grad_params[u], model.parameters(), retain_graph=True)

        temp.append(stack_torch_tensors(second_grad))

    Hessian = torch.cat(temp, axis=1)

    return Hessian


def exact_hessian_ij(model, loss_ij):
    grad_params = torch.autograd.grad(loss_ij, model.parameters(), retain_graph=True, create_graph=True)
    grad_params = stack_torch_tensors(grad_params)
    hess_params = torch.zeros((len(grad_params), len(grad_params)))
    temp = []

    for u in range(len(grad_params)):
        second_grad = torch.autograd.grad(grad_params[u], model.parameters(), retain_graph=True)

        temp.append(stack_torch_tensors(second_grad))

    Hessian = torch.cat(temp, axis=1)

    return Hessian


def hessian_vector_product(loss, model, v):
    """
    Multiplies the Hessians of the loss of a model with respect to its parameters by a vector v.
    Adapted from: https://github.com/kohpangwei/influence-release

    This function uses a backproplike approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians with O(p) compelxity for p parameters.

    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        rnn: the model for which the Hessian of the loss is evaluated
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """

    # First backprop
    first_grads = stack_torch_tensors(
        torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True))

    """
    # Elementwise products
    elemwise_products = 0

    for grad_elem, v_elem in zip(first_grads, v):

        elemwise_products += torch.sum(grad_elem * v_elem)
    """

    elemwise_products = torch.mm(first_grads.view(-1, first_grads.shape[0]).float(),
                                 v[0].view(first_grads.shape[0], -1).float())

    # Second backprop
    HVP_ = torch.autograd.grad(elemwise_products, model.parameters(), create_graph=True)

    return HVP_


def perturb_model_(model, perturb, data, args):
    """
    Perturbs the parameters of a model by a given vector of influences

    Arguments:
        model: a pytorch model with p parameters
        perturb: a tensors with size p designating the desired parameter-wise perturbation

    Returns:
        perturbed_model : a copy of the original model with perturbed parameters
    """

    params = []

    for param in model.parameters():
        params.append(param.clone())

    param_ = stack_torch_tensors(params)
    new_param_ = param_ - perturb

    # copy all model attributes

    perturbed_model = type(model)(data, args)

    new_model_dict = dict.fromkeys(model.__dict__.keys())
    new_model_state = collections.OrderedDict.fromkeys(model.state_dict().keys())

    for key in new_model_dict.keys():

        if type(model.__dict__[key]) == torch.Tensor:

            new_model_dict[key] = model.__dict__[key].clone()

        else:

            new_model_dict[key] = copy.deepcopy(model.__dict__[key])

    for key in new_model_state.keys():

        if type(model.state_dict()[key]) == torch.Tensor:

            new_model_state[key] = model.state_dict()[key].clone()

        else:

            new_model_state[key] = copy.deepcopy(model.state_dict()[key])

    perturbed_model.__dict__.update(new_model_dict)
    perturbed_model.load_state_dict(new_model_state)

    index = 0

    for param in perturbed_model.parameters():

        if len(param.data.shape) > 1:

            new_size = np.max((1, param.data.shape[0])) * np.max((1, param.data.shape[1]))
            param.data = new_param_[index: index + new_size].view(param.data.shape[0], param.data.shape[1])

        else:

            new_size = param.data.shape[0]
            param.data = np.squeeze(new_param_[index: index + new_size])

        index += new_size

    return perturbed_model
