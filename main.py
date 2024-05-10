import numpy as np
import matplotlib.pyplot as plt

def sum_squared_error(x_train, y_train, w, b):
    m = x_train.shape[0]
    ret = 0
    for i in range(m):
        pred = np.dot(x_train[i], w) + b
        err = pred - y_train[i]
        ret += err ** 2
    return ret

def calculate_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]
    dj_dw = np.zeros(w.shape[0])
    dj_db = 0

    for i in range(m):
        pred = np.dot(x_train[i], w) + b
        diff = pred - y_train[i]

        for j in range(w.shape[0]):
            # jth parameter in ith data row
            dj_dw[j] += diff * x_train[i, j]
        dj_db += diff
    
    return dj_dw / m, dj_db / m

def gradient_descent(x_train, y_train, w, b, alpha):
    dj_dw, dj_db = calculate_gradient(x_train, y_train, w, b)
    return w - alpha * dj_dw, b - alpha * dj_db


def main():
    x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([10, 11, 12])
    
    itns = 10000
    alpha = 1e-2
    epsilon = 1e-3
    error_history = []

    w, b = gradient_descent(x_train, y_train, np.zeros(x_train.shape[1]), 0, alpha)
    error_history.append(sum_squared_error(x_train, y_train, w, b))

    for i in range(itns):
        w, b = gradient_descent(x_train, y_train, w, b, alpha)
        sse = sum_squared_error(x_train, y_train, w, b)
        sse_diff = abs(sse - error_history[-1])

        if sse_diff < epsilon:
            break
        if i % 50 == 0:
            print(f"itn {i}:    y={w}x+{b}  sse={sse}")
        error_history.append(sse)

    print(f"multivariable linear regression eqn.: y={w}x+{b}")

    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("J(w,b)")
    plt.plot(np.arange(len(error_history)) + 1, error_history)
    plt.show()

if __name__ == "__main__":
    main()