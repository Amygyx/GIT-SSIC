import torch
import torch.nn as nn

def lowerbound(input, min):
    return F_Lowerbound.apply(input, min)

class F_Lowerbound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min):
        ctx.min = min
        ctx.input = input
        # ctx.save_for_backward(input)
        bound = input.new(input.size()).fill_(min)
        output = input.clone()
        output[input < bound] = min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        min = ctx.min
        input = ctx.input
        # input = ctx.saved_tensors
        bound = input.new(input.size()).fill_(min)
        grad_input = grad_output.clone()
        mask = input.new(input.size()).fill_(0)
        mask[input >= bound] = 1
        mask[grad_input < 0] = 1
        grad_input[mask == 0] = 0
        return grad_input, None

class GaussianConditional(nn.Module):
    def __init__(self, scale_bound=1e-5, likelihood_bound=1e-6, x_min=-255, x_max=255):
        super(GaussianConditional, self).__init__()
        self.scale_bound = scale_bound
        self.likelihood_bound = likelihood_bound
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

    def forward(self, input, scale, mean):
        scale = lowerbound(scale, self.scale_bound)
        likelihood = self._likelihood(input, scale, mean)
        likelihood = lowerbound(likelihood, self.likelihood_bound)
        return likelihood

    def _standardized_cumulative(self, input):
        half = .5
        const = -(2 ** -0.5)
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * input)

    def _likelihood(self, input, scale, mean):
        value = input - mean
        upper = self._standardized_cumulative((.5 - value) / scale)
        lower = self._standardized_cumulative((-.5 - value) / scale) # TODO
        likelihood = upper - lower
        upper_plus = upper # probability for edge case of 0
        lower_plus = 1 - lower # probability for edge case of 255
        cond_B = (input > self.x_upper_bound).float()
        likelihood = (cond_B * lower_plus + (1. - cond_B) * likelihood)
        cond_C = (input < self.x_lower_bound).float()
        likelihood = cond_C * upper_plus + (1. - cond_C) * likelihood
        return likelihood

class LogisticConditional(nn.Module):
    def __init__(self, scale_bound=1e-5, likelihood_bound=1e-6, x_min=-255., x_max=255.):
        super(LogisticConditional, self).__init__()
        self.scale_bound = scale_bound
        self.likelihood_bound = likelihood_bound
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

    def forward(self, input, scale, mean):
        scale = lowerbound(scale, self.scale_bound)
        likelihood = self._likelihood(input, scale, mean)
        likelihood = lowerbound(likelihood, self.likelihood_bound)
        return likelihood

    def _standardized_cumulative(self, input):
        return torch.sigmoid(input)

    def _likelihood(self, input, scale, mean):
        value = input - mean
        upper = self._standardized_cumulative((value + .5) / scale)
        lower = self._standardized_cumulative((value - .5) / scale)
        likelihood = upper - lower
        upper_plus = upper # probability for edge case of 0
        One_Minus_Lower = 1 - lower # probability for edge case of 255
        cond_B = (input > self.x_upper_bound).float()
        likelihood = (cond_B * One_Minus_Lower + (1. - cond_B) * likelihood)
        cond_C = (input < self.x_lower_bound).float()
        likelihood = cond_C * upper_plus + (1. - cond_C) * likelihood
        return likelihood