--
-- User: mathieu
-- Date: 19/12/16
-- Time: 10:28 AM
-- Description: This example is the same as the first one with a particular loss.
--              The goal of this small template is to be able to handle and retrieve information
--              When a particular loss is needed, and make it generic on python side.
--

require 'ModelBase'

local ModelExample2 = torch.class('ModelExample2', 'ModelBase')

function ModelExample2:__init(backend, learning_rate, weight_decay)
    ModelBase.__init(self, backend, learning_rate, weight_decay)

end

function ModelExample2:build_model(input_shape, filter_quantity, filter_size)
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

    local model2 = model:sharedClone('weight', 'bias', 'gradWeight', 'gradBias')

    local parallel = nn.ParallelTable()
    parallel:add(model)
    parallel:add(model2)

    self.net = parallel
end

function ModelExample2:convert_inputs(inputs)
    self.inputTensor1 = self:setup_tensor(inputs[1], self.inputTensor1)
    self.inputTensor2 = self:setup_tensor(inputs[2], self.inputTensor2)
    return {self.inputTensor1, self.inputTensor2}
end

function ModelExample2:convert_outputs(outputs)
    return {outputs[1]:float(), outputs[2]:float()}
end

function ModelExample2:compute_criterion(forward_input, label)
    self.labelTensor1 = self:setup_tensor(label[1], self.labelTensor1)
    self.labelTensor2 = self:setup_tensor(label[2], self.labelTensor2)

    if self.crit == nil then
        -- sometimes, a ParallelCriterion is not exactly what we need for some reasons
        self.crit = {self:set_backend(nn.MSECriterion()), self:set_backend(nn.MSECriterion())}
    end
    local label_loss1 = self.crit[1]:forward(forward_input[1], self.labelTensor1)
    local label_grad1 = self.crit[1]:backward(forward_input[1], self.labelTensor1)

    local label_loss2 = self.crit[2]:forward(forward_input[2], self.labelTensor2)
    local label_grad2 = self.crit[2]:backward(forward_input[2], self.labelTensor2)

    return {label=label_loss1, other=label_loss2}, {label_grad1, label_grad2}
end
