import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score

""" 遗传算法选取特征
"""


def initilization_of_population(size, n_feat):
    """初始化种群

    Args:
        size ([type]): 种群大小
        n_feat ([type]): 特征数

    Returns:
        [type]: 初始化的种群
    """
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[:int(0.3*n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def selection(pop_after_fit, n_parents):
    """选择需要进行交叉变异的个体

    Args:
        pop_after_fit ([type]): 计算了适应度后的整个种群
        n_parents ([type]): 需要交叉变异的个数

    Returns:
        [type]: [description]
    """
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    """交叉

    Args:
        pop_after_sel ([type]): 待交叉的个体

    Returns:
        [type]: 交叉后的个体
    """
    population_nextgen = pop_after_sel
    for i in range(len(pop_after_sel)):
        child = pop_after_sel[i]
        child[3:7] = pop_after_sel[(i+1) % len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate):
    """变异

    Args:
        pop_after_cross ([type]): 交叉后的个体
        mutation_rate ([type]): 变异率

    Returns:
        [type]: 变异后的个体
    """
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j] = not chromosome[j]
        population_nextgen.append(chromosome)
    # print(population_nextgen)
    return population_nextgen


def generations(model, size, n_feat, n_parents, mutation_rate, n_gen, X_train,
                X_test, y_train, y_test):

    def fitness_score(population):
        """计算适应度

        Args:
            population ([type]): [description]

        Returns:
            [type]: [description]
        """
        scores = []
        for chromosome in population:
            model.fit(X_train.iloc[:, chromosome], y_train)
            predictions = model.predict(X_test.iloc[:, chromosome])
            scores.append(accuracy_score(y_test, predictions))
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        return list(scores[inds][::-1]), list(population[inds, :][::-1])

    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score


# m = ExtraTreesRegressor()
# chromo, score = generations(m, size=200, n_feat=225, n_parents=100, mutation_rate=0.10,
#                             n_gen=38, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
# logmodel.fit(X_train.iloc[:, chromo[-1]], y_train)
# predictions = logmodel.predict(X_test.iloc[:, chromo[-1]])
