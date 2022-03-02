import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import datetime
print(datetime.datetime.now())


def create_regular_grid(shape):
    if len(shape) == 2:
        num1, num2 = (shape[0]) // 1, (shape[1]) // 1
        grid = np.meshgrid(np.linspace(-1, 1, num1),
                           np.linspace(-1, 1, num2))

    elif len(shape) == 3:
        num1, num2, num3 = (shape[0]) // 1, (shape[1]) // 1, (shape[2]) // 1
        grid = np.meshgrid(np.linspace(-1, 1, num1),
                           np.linspace(-1, 1, num2),
                           np.linspace(-1, 1, num3))
    else:
        grid = np.linspace(-1, 1, shape[0])
    grid = np.stack(grid, axis=-1)
    return grid


def grid2contour(grid, title):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    assert grid.ndim == 3
    x = np.arange(-1, 1, 2.0 / grid.shape[1])
    y = np.arange(-1, 1, 2.0 / grid.shape[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2

    plt.figure()
    plt.contour(X, Y, Z1, 15, levels=50, colors='k')  # 改变levels的值，可以改变形变场的密集程度
    plt.contour(X, Y, Z2, 15, levels=50, colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title(title)
    plt.show()
    # plt.savefig(path)  # 保存图像


# ****************************************************
# 生成网格图片
def create_grid(size, path):
    num1, num2 = (size[0] + 10) // 1, (size[1] + 10) // 1  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.savefig(path)  # 保存图像
    plt.show()


# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


if __name__ == "__main__":
    print("end")
