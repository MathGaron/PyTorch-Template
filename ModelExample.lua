--
-- User: mathieu
-- Date: 19/12/16
-- Time: 8:35 AM
-- Description: basic example, see ModelExample2 for a "more" complicated one
--

require 'ModelBase'

local ModelExample = torch.class('ModelExample', 'ModelBase')

function ModelExample:__init(backend, learning_rate, weight_decay)
    ModelBase.__init(self, backend, learning_rate, weight_decay)

end

function ModelExample:build_model(input_shape, filter_quantity, filter_size)
    local convoluted_h = (input_shape[2] - math.floor(filter_size/2)*2)/2
    local convoluted_w = (input_shape[3] - math.floor(filter_size/2)*2)/2
    local linear_size = filter_quantity * convoluted_h * convoluted_w
    local model = nn:Sequential()
    model:add(nn.SpatialConvolution(3, filter_quantity, filter_size, filter_size))
    model:add(nn.SpatialBatchNormalization(12))
    model:add(nn.ELU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.View(linear_size))
    model:add(nn.Linear(linear_size, 2))
    model:add(nn.ReLU())
    self.net = model
end

function ModelExample:convert_inputs(inputs)
    self.inputTensor = self:setup_tensor(inputs, self.inputTensor)
    return self.inputTensor
end

function ModelExample:convert_outputs(outputs)
    return outputs:float()
end

function ModelExample:compute_criterion(forward_input, label)
    self.labelTensor = self:setup_tensor(label, self.labelTensor)
    if self.crit == nil then
        self.crit = self:set_backend(nn.MSECriterion())
    end
    local label_loss = self.crit:forward(forward_input, self.labelTensor)
    local label_grad = self.crit:backward(forward_input, self.labelTensor)
    return {label=label_loss}, label_grad
end
