import random
import csv
import math

import matplotlib
import matplotlib.pyplot
import numpy

iterations = 1000


def calc_grad(i, ys, alphas, data):
    sum = 0.0
    n = len(ys)
    for j in range(n):
        sum += alphas[j] * ys[j] * data[i][j]
    return -1.0 + 0.5 * ys[i] * sum


def tuple_plus(a, b):
    res = [0.0] * len(a)
    for i in range(len(a)):
        res[i] = a[i] + b[i]
    return res


def tuple_sub(a, b):
    res = [0.0] * len(a)
    for i in range(len(a)):
        res[i] = a[i] - b[i]
    return res


def tuple_mul(a, b):
    res = [0.0] * len(a)
    for i in range(len(a)):
        res[i] = a[i] * b[i]
    return res


def tuple_mul_number(a, k):
    res = [0.0] * len(a)
    for i in range(len(a)):
        res[i] = a[i] * k
    return res


def calc_b(alphas, xs, ys, kernel_func, kernel_arg):
    b = 0
    n = len(ys)
    w = [0.0] * len(xs[0])
    for i in range(n):
        w = tuple_plus(w, tuple_mul_number(xs[i], alphas[i] * ys[i]))
    for i in range(n):
        b += kernel_func(w, xs[i], kernel_arg) - ys[i]
    return -b / n


def calc_alphas(ys, data, c, step):
    n = len(ys)
    alphas = [0.0] * n
    for i in range(iterations):
        ind1 = random.randint(0, n - 1)
        ind2 = random.randint(0, n - 1)
        if ind1 == ind2:
            ind1 += 1
            if ind1 == n:
                ind1 = 0
        grad = calc_grad(ind1, ys, alphas, data)
        new_alpha1 = alphas[ind1] - step * grad
        new_alpha2 = alphas[ind2]
        if ys[ind1] != ys[ind2]:
            new_alpha2 -= step * grad
        else:
            new_alpha2 += step * grad
        if check(new_alpha1, c) and check(new_alpha2, c):
            alphas[ind1] = new_alpha1
            alphas[ind2] = new_alpha2
    return alphas


def check(u, c):
    return 0 <= u <= c


def scalar(a, b):
    res = 0.0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def calc_poly_kernel(x1, x2, p):
    return scalar(x1, x2) ** p


def euclidean(v):
    res = 0.0
    for i in range(len(v)):
        res += v[i] ** 2
    return res


def calc_gaussian_kernel(x1, x2, beta):
    return math.exp(-beta * euclidean(tuple_sub(x1, x2)))


def calc_linear_kernel_on_xs(xs):
    n = len(xs)
    data = [[0.0] * n] * n
    for i in range(n):
        for j in range(n):
            data[i][j] = calc_poly_kernel(xs[i], xs[j], 1)
    return data


def calc_poly_kernel_on_xs(xs, p):
    n = len(xs)
    data = [[0.0] * n] * n
    for i in range(n):
        for j in range(n):
            data[i][j] = calc_poly_kernel(xs[i], xs[j], p)
    return data


def calc_gaussian_kernel_on_xs(xs, beta):
    n = len(xs)
    data = [[0.0] * n] * n
    for i in range(n):
        for j in range(n):
            data[i][j] = calc_gaussian_kernel(xs[i], xs[j], beta)
    return data


def sgn(u):
    if u <= 0:
        return -1
    else:
        return 1


def get_class(x, alphas, b, xs, ys, kernel_func, kernel_arg):
    sum = 0.0
    n = len(ys)
    for i in range(n):
        sum += alphas[i] * ys[i] * kernel_func(x, xs[i], kernel_arg)
    return sum + b


def cls(x, alphas, b, xs, ys, kernel_func, kernel_arg):
    return sgn(get_class(x, alphas, b, xs, ys, kernel_func, kernel_arg))


def calc_accuracy(alphas, b, xs, ys, kernel_func, kernel_arg):
    right = 0.0
    n = len(ys)
    for i in range(n):
        if ys[i] == cls(xs[i], alphas, b, xs, ys, kernel_func, kernel_arg):
            right += 1
    return right / float(n)


def draw_plot(alphas, b, xs, ys, kernel_func, kernel_arg, graph_file):
    N = 100
    X, Y = numpy.mgrid[-1.5:1.5:complex(0, N), -1.5:1.5:complex(0, N)]
    matrix = []
    for i in range(len(X)):
        matrix.append([])
        for j in range(len(Y)):
            matrix[i].append(get_class([X[i][0], Y[0][j]], alphas, b, xs, ys, kernel_func, kernel_arg))
    fig, ax0 = matplotlib.pyplot.subplots()
    eps = 0.6
    c = ax0.pcolor(X, Y, matrix, cmap='rainbow', vmin=-eps, vmax=eps)
    fig.colorbar(c, ax=ax0)
    for i in range(len(xs)):
        color = "black" if cls(xs[i], alphas, b, xs, ys, kernel_func, kernel_arg) == ys[i] else "white"
        symb = '+' if ys[i] == 1 else '_'
        matplotlib.pyplot.plot(xs[i][0], xs[i][1], symb, color=color)
    matplotlib.pyplot.savefig(graph_file)


random.seed(1337228)
filename = 'chips.csv'

xs = []
ys = []
with open(filename) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        xs.append([float(x) for x in row[:-1]])
        ys.append(float(row[-1]))

# data = calc_gaussian_kernel_on_xs(xs, 5)
# alphas = calc_alphas(ys, data, 0.1, 0.1)
# draw_plot(alphas, calc_b(alphas, xs, ys, calc_gaussian_kernel, 5), xs, ys, calc_gaussian_kernel, 5,
#           "gaussian_chips.png")

# cs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
# ps = [2, 3, 4, 5]
# betas = [1, 2, 3, 4, 5]
# steps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# output_filename = 'gaussian_kernel_without_cross_validation_chips_info.txt'
# output_file = open(output_filename, 'w')
# print("STARTED CALCULATIONS FOR GAUSSIAN KERNEL, PLS WAIT")
# for beta in betas:
#     gaussian_data = calc_gaussian_kernel_on_xs(xs, beta)
#     for step in steps:
#         for c in cs:
#             gaussian_alphas = calc_alphas(ys, gaussian_data, c, step)
#             gaussian_b = calc_b(gaussian_alphas, xs, ys, calc_gaussian_kernel, beta)
#             output_file.write("beta = " + str(beta) + '\n')
#             output_file.write("c = " + str(c) + '\n')
#             output_file.write("step = " + str(step) + '\n')
#             output_file.write(
#                 "accuracy = " + str(
#                     calc_accuracy(gaussian_alphas, gaussian_b, xs, ys, calc_gaussian_kernel, beta)) + '\n')
#             print("CALCULATED beta = " + str(beta) + " c = " + str(c) + " step = " + str(step))
# print("FINISHED CALCULATIONS FOR GAUSSIAN KERNEL")
#
# output_filename = 'poly_kernel_without_cross_validation_chips_info.txt'
# output_file = open(output_filename, 'w')
# print("STARTED CALCULATIONS FOR POLY KERNEL, PLS WAIT")
# for p in ps:
#     poly_data = calc_poly_kernel_on_xs(xs, p)
#     for step in steps:
#         for c in cs:
#             poly_alphas = calc_alphas(ys, poly_data, c, step)
#             poly_b = calc_b(poly_alphas, xs, ys, calc_poly_kernel, p)
#             output_file.write("p = " + str(p) + '\n')
#             output_file.write("c = " + str(c) + '\n')
#             output_file.write("step = " + str(step) + '\n')
#             output_file.write(
#                 "accuracy = " + str(calc_accuracy(poly_alphas, poly_b, xs, ys, calc_poly_kernel, p)) + '\n')
#             print("CALCULATED p = " + str(p) + " c = " + str(c) + " step = " + str(step))
# print("FINISHED CALCULATIONS FOR POLY KERNEL")
#
# output_filename = 'linear_kernel_without_cross_validation_geyser_info.txt'
# output_file = open(output_filename, 'w')
# print("STARTING CALCULATIONS FOR LINEAR KERNEL, PLS WAIT")
# linear_data = calc_linear_kernel_on_xs(xs)
# for c in cs:
#     for step in steps:
#         linear_alphas = calc_alphas(ys, linear_data, c, step)
#         linear_b = calc_b(linear_alphas, xs, ys, calc_poly_kernel, 1)
#         output_file.write("c = " + str(c) + " step = " + str(step) + '\n')
#         output_file.write(
#             "accuracy = " + str(calc_accuracy(linear_alphas, linear_b, xs, ys, calc_poly_kernel, 1)) + '\n')
# print("FINISHED CALCULATIONS FOR LINEAR KERNEL")

# output_filename = 'linear_kernel_with_cross_validation_geyser_info.txt'
# output_file = open(output_filename, 'w')
# print('STARTED CALCULATIONS')
# for c in cs:
#     for step in steps:
#         right = 0
#         for ind_to_skip in range(len(xs)):
#             cur_xs = xs[:ind_to_skip] + xs[(ind_to_skip + 1):]
#             cur_ys = ys[:ind_to_skip] + ys[(ind_to_skip + 1):]
#             linear_data = calc_linear_kernel_on_xs(cur_xs)
#             linear_alphas = calc_alphas(cur_ys, linear_data, c, step)
#             linear_b = calc_b(linear_alphas, cur_xs, cur_ys, calc_poly_kernel, 1)
#             if cls(xs[ind_to_skip], linear_alphas, linear_b, cur_xs, cur_ys, calc_poly_kernel, 1) == ys[ind_to_skip]:
#                 right += 1
#         output_file.write("c = " + str(c) + " step = " + str(step) + '\n')
#         output_file.write(
#             "accuracy = " + str(right / float(len(xs))) + '\n')
#         print("CALCULATED c = " + str(c) + " step = " + str(step))
# print("FINISHED CALCULATIONS FOR LINEAR KERNEL")
#

# output_filename = 'poly_kernel_with_cross_validation_geyser_info.txt'
# output_file = open(output_filename, 'w')
# print("STARTED CALCULATIONS FOR POLY KERNEL, PLS WAIT")
# for p in ps:
#     poly_data = calc_poly_kernel_on_xs(xs, p)
#     for step in steps:
#         for c in cs:
#             right = 0
#             for ind_to_skip in range(len(xs)):
#                 cur_xs = xs[:ind_to_skip] + xs[(ind_to_skip + 1):]
#                 cur_ys = ys[:ind_to_skip] + ys[(ind_to_skip + 1):]
#                 poly_alphas = calc_alphas(cur_ys, poly_data, c, step)
#                 poly_b = calc_b(poly_alphas, cur_xs, cur_ys, calc_poly_kernel, p)
#                 if cls(xs[ind_to_skip], poly_alphas, poly_b, cur_xs, cur_ys, calc_poly_kernel, 1) == ys[ind_to_skip]:
#                     right += 1
#             output_file.write("p = " + str(p) + '\n')
#             output_file.write("c = " + str(c) + '\n')
#             output_file.write("step = " + str(step) + '\n')
#             output_file.write(
#                 "accuracy = " + str(right / float(len(xs))) + '\n')
#             print("CALCULATED p = " + str(p) + " c = " + str(c) + " step = " + str(step))
# print("FINISHED CALCULATIONS FOR POLY KERNEL")

# output_filename = 'gaussian_kernel_with_cross_validation_geyser_info.txt'
# output_file = open(output_filename, 'w')
# print("STARTED CALCULATIONS FOR GAUSSIAN KERNEL, PLS WAIT")
# for beta in betas:
#     gaussian_data = calc_gaussian_kernel_on_xs(xs, beta)
#     for step in steps:
#         for c in cs:
#             right = 0
#             for ind_to_skip in range(len(xs)):
#                 cur_xs = xs[:ind_to_skip] + xs[(ind_to_skip + 1):]
#                 cur_ys = ys[:ind_to_skip] + ys[(ind_to_skip + 1):]

#                 gaussian_alphas = calc_alphas(cur_ys, gaussian_data, c, step)
#                 gaussian_b = calc_b(gaussian_alphas, cur_xs, cur_ys, calc_gaussian_kernel, beta)
#                 if cls(xs[ind_to_skip], gaussian_alphas, gaussian_b, cur_xs, cur_ys, calc_poly_kernel, 1) == \
#                         ys[ind_to_skip]:
#                     right += 1
#             output_file.write("beta = " + str(beta) + '\n')
#             output_file.write("c = " + str(c) + '\n')
#             output_file.write("step = " + str(step) + '\n')
#             output_file.write(
#                 "accuracy = " + str(right / float(len(xs))) + '\n')
#             print("CALCULATED beta = " + str(beta) + " c = " + str(c) + " step = " + str(step))
# print("FINISHED CALCULATIONS FOR GAUSSIAN KERNEL")
