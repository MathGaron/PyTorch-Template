--
-- User: mathieu
-- Date: 19/12/16
-- Time: 8:34 AM
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'math'

local ModelBase = torch.class('ModelBase')

function ModelBase:__init(backend, learning_rate, weight_decay)
    self.net = nil
    self.backend = backend
    self.optimFunction = optim.adam
    self.config = {
        learningRate = learning_rate,
        learningRateDecay = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-08,
        momentum = 0.9,
        dampening = 0,
        nesterov = 0.9,

        weightDecay = weight_decay,
        --linear_size = linear_size,
        --convo1_filters = convo1_filters,
        --convo2_filters = convo2_filters,
    }
end

function ModelBase:show_model()
    print(string.format("Backend : %s", self.backend))
    print(self.net)
end

-- Convert tensor based on backend requested
function ModelBase:setup_tensor(ref, buffer)
    local localOutput = buffer
    if self.backend == 'cpu' then
        localOutput = ref
    else
        localOutput = localOutput or ref:clone()
        if torch.type(localOutput) ~= 'torch.CudaTensor' then
            localOutput = localOutput:cuda()
        end
        localOutput:resize(ref:size())
        localOutput:copy(ref)
    end
    return localOutput
end

function ModelBase:set_backend(module)
    if self.backend == 'cuda' then
        module = module:cuda()
    else
        module = module:float()
    end
    return module
end

function ModelBase:convert_inputs(inputs)
    -- this function is used when you have particular inputs, it handles backend transfer and any formating to the input data
    error("convert_inputs not defined!")
end

function ModelBase:convert_outputs(outputs)
    -- convert forward outputs so it can be handled in python
    error("convert_outputs not defined!")
end

function ModelBase:compute_criterion(forward_input, label)
    -- compute the criterion given the output of forward and labels, returns a dict with losses :
    -- label : the generic loss used for trainning algorithm
    -- user_defined_loss : any other loss.
    error("compute_criterion not defined!")
end

function ModelBase:init_model()
    self.net = self:set_backend(self.net)
    self.params, self.gradParams = self.net:getParameters()
end

function ModelBase:train(inputs, labels)
    self.net:training()
    local func = function(x)
        collectgarbage()
        self.gradParams:zero()
        local converted_inputs = self:convert_inputs(inputs)
        local output = self.net:forward(converted_inputs)
        losses, f_grad = self:compute_criterion(output, labels)
        self.net:backward(converted_inputs, f_grad)
       return losses['label'], self.gradParams
    end
    self.optimFunction(func, self.params, self.config)
    return losses
end

function ModelBase:test(inputs)
    collectgarbage(); collectgarbage()
    self.net:evaluate()
    local converted_inputs = self:convert_inputs(inputs)
    local output = self.net:forward(converted_inputs)
    return self:convert_outputs(output)
end

function ModelBase:Save(path)
    torch.save(path..".t7", self.net)
    torch.save(path.."_crit.t7", self.crit)
    torch.save(path.."_optim.t7", self.config)
end

function ModelBase:Load(path)
    self.net = torch.load(path..".t7")
    self.config = torch.load(path.."_optim.t7")
    self.crit = self:set_backend(torch.load(path.."_crit.t7"))
    self:init_model()
end

