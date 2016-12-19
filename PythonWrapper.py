import PyTorchHelpers
import numpy as np

def Example1():
    # init
    model = PyTorchHelpers.load_lua_class("ModelExample.lua", 'ModelExample')
    torch_model = model("cuda", 0.001, 0)
    torch_model.build_model((3, 100, 100), 12, 5)
    torch_model.init_model()

    torch_model.show_model()

    # define inputs/labels
    img = np.ones((2, 3, 100, 100), dtype=np.float32)
    img[0, :, :, :] = -0.5
    img[1, :, :, :] = 0.5
    label = np.ones((2, 2), dtype=np.float32)
    label[0, :] = 0

    # train
    for i in range(100):
        loss = torch_model.train(img, label)
        print("loss : {}".format(loss))

    # test
    prediction = torch_model.test(img).asNumpyTensor()
    print("prediction : {}".format(prediction))

def Example2():
    # init
    model = PyTorchHelpers.load_lua_class("ModelExample2.lua", 'ModelExample2')
    torch_model = model("cuda", 0.001, 0)
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

    # train
    for i in range(100):
        loss = torch_model.train([img1, img2], [label1, label2])
        print("loss : {}".format(loss))

    # test
    prediction = torch_model.test([img1, img2])
    print("prediction : {}".format(prediction))


if __name__ == '__main__':
    #Example1()
    Example2()
