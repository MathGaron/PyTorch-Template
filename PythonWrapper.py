import PyTorchHelpers
import numpy as np


def Example1():
    # init
    model_class = PyTorchHelpers.load_lua_class("ModelExample.lua", 'ModelExample')
    torch_model = model_class("cuda", 0.001)
    torch_model.build_model((3, 100, 100), 12, 5)
    torch_model.init_model()

    torch_model.show_model()

    # define inputs/labels
    img = np.ones((2, 3, 100, 100), dtype=np.float32)
    img[0, :, :, :] = -0.5
    img[1, :, :, :] = 0.5
    label = np.ones((2, 2), dtype=np.float32)
    label[0, :] = 0

    return torch_model, img, label


def Example2():
    # init
    model_class = PyTorchHelpers.load_lua_class("ModelExample2.lua", 'ModelExample2')
    torch_model = model_class("cuda", 0.001)
    torch_model.build_model((3, 100, 100), 12, 5)
    torch_model.init_model()

    torch_model.show_model()

    # define inputs/labels
    img1 = np.ones((2, 3, 100, 100), dtype=np.float32)
    img1[0, :, :, :] = -0.5
    img1[1, :, :, :] = 0.5

    img2 = np.ones((2, 3, 100, 100), dtype=np.float32)
    img2[0, :, :, :] = 0.5
    img2[1, :, :, :] = -0.5

    label1 = np.ones((2, 2), dtype=np.float32)
    label1[0, :] = 0

    label2 = np.ones((2, 2), dtype=np.float32)
    label2[0, :] = 0

    return torch_model, [img1, img2], [label1, label2]


if __name__ == '__main__':
    # Example1()
    model, inputs, labels = Example2()

    # train
    for i in range(100):
        loss = model.train(inputs, labels)
        print("loss : {}".format(loss))

    # test
    prediction = model.test(inputs)
    print("prediction : {}".format(prediction))
