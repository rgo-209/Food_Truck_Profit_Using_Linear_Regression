"""
    This program implements linear regression with one variable.
    By: Rahul Golhar
""" 
import numpy
import matplotlib.pyplot as plt
from ipython_genutils.py3compat import xrange
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def h(theta, X):
    """
        This function returns the hypothesis value. (x*theta)
    :param theta:   the theta vector
    :param X:       the X value matrix
    :return:        hypothesis value
    """
    return numpy.dot(X, theta)


def costFunction(theta, X, y):
    """
        This function returns the values calculated by the
        cost function.
    :param theta:   theta vector to be used
    :param X:       X matrix
    :param y:       y Vector
    :return:        Cost Function results
    """
    # note to self: *.shape is (rows, columns)
    m = len(y)
    return float((1. / (2 * m)) * numpy.dot((h(theta, X) - y).T, (h(theta, X) - y)))


def gradientDescent(X, y, alpha, iterations, initialTheta=numpy.zeros(2)):
    """
            This function minimizes the gradient descent.
    :param X:                   X matrix
    :param y:                   y vector
    :param alpha:               value of learning rate
    :param iterations:          no of iterations to run
    :param initialTheta:        initial values of theta vector to start with
    :return: theta, calculatedTheta, jTheta
    """
    theta = initialTheta
    # values of cost functions will be stored here
    jTheta = []
    # calculated Theta values will be stored here
    calculatedTheta = []
    # no of training examples used
    m = y.size

    print("\n\tMinimizing the cost function and finding the optimal theta values.")
    for itr in xrange(iterations):
        thetaTemp = theta

        jTheta.append(costFunction(theta, X, y))

        calculatedTheta.append(list(theta[:, 0]))

        # Update values of theta
        for j in xrange(len(thetaTemp)):
            thetaTemp[j] = theta[j] - (alpha / m) * numpy.sum((h(theta, X) - y) * numpy.array(X[:, j]).reshape(m, 1))
        theta = thetaTemp

    print("\tDone with minimizing of the cost function and finding the optimal theta values.")

    return theta, calculatedTheta, jTheta


def plotConvergence(jTheta):
    """
        This function plots the convergence graph.
    :param jTheta:  the minimized cost function value
    :return:        None
    """
    print("\n\tPlotting the convergence.")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(jTheta)), jTheta, 'bo')
    plt.grid(True)
    plt.title("Convergence graph for the Cost Function")
    plt.xlabel("Iteration")
    plt.ylabel("Cost function value")
    dummy = plt.ylim([4, 7])
    plt.savefig("convergenceOfCostFunction.jpg")
    print("\tSaved the convergence graph to convergenceOfCostFunction.jpg.")


def plotInitialData(X, y):
    """
        This function plots the initial data and saves it.
    :param X: X matrix
    :param y: y vector
    :return: None
    """
    print("\n\tPlotting the initial data.")
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], y[:, 0], 'rx', markersize=10)
    plt.grid(True)
    plt.title("Initial graph for the data")
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig("initialDataPlot.jpg")
    print("\tSaved the initial data plotted to initialDataPlot.jpg.")


def plotHypothesis(X, y, results, filename):
    """
        This function plots the hypothesis line
        using X, y and the results.
    :param X:           X matrix
    :param y:           y vector
    :param results:     calculated results
    :param filename:    file to save the graph to
    :return:            None
    """
    print("\n\tPlotting the user defined hypothesis.")
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], y[:, 0], 'rx', markersize=10, label='Training Data')
    plt.plot(X[:, 1], results, 'b-')
    plt.grid(True)
    plt.title("Graph with hypothesis for the data")
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig(filename)
    print("\tSaved the hypothesis plotted to ", filename)


# def plotInBuiltHypothesis(X,y,predResults, filename):
#     """
#         This function plots the hypothesis line
#         using X, y and the calculated theta.
#     :param X:       X matrix
#     :param y:       y vector
#     :param theta:   theta vector
#     :return:        None
#     """
#     filename = filename+".jpg"
#     print("\n\tPlotting the inbuilt hypothesis.")
#     plt.figure(figsize=(10, 6))
#     plt.plot(X[:, 1], y[:, 0], 'rx', markersize=10, label='Training Data')
#     plt.plot(X[:, 1], predResults, 'b-')
#     plt.grid(True)
#     plt.title("Graph with hypothesis for the data")
#     plt.ylabel('Profit in $10,000s')
#     plt.xlabel('Population of City in 10,000s')
#     plt.savefig("inbuiltHypothesis.jpg")
#     print("\tSaved the in built function hypothesis plotted to inbuiltHypothesis.jpg")


def plotMinimizationPath(X, y, calculatedTheta, jTheta):
    """
        This function plots the minimization path followed by the algorithm.
    :param X:                   X matrix
    :param y:                   y vector
    :param calculatedTheta:     final theta values calculated
    :param jTheta:              minimized cost functio values
    :return:
    """
    print("\n\tPlotting the minimization path.")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')

    # range of Xvalues
    xValues = numpy.arange(-10, 10, .5)
    yValues = numpy.arange(-1, 4, .1)
    xVals, yVals, zVals = [], [], []

    for xVal in xValues:
        for yVal in yValues:
            xVals.append(xVal)
            yVals.append(yVal)
            zVals.append(costFunction(numpy.array([[xVal], [yVal]]), X, y))

    scat = ax.scatter(xVals, yVals, zVals, c=numpy.abs(zVals), cmap=plt.get_cmap('YlOrRd'))

    plt.xlabel('theta1', fontsize=20)
    plt.ylabel('theta1', fontsize=20)
    plt.title('Cost Minimization Path', fontsize=25)
    plt.plot([x[0] for x in calculatedTheta], [x[1] for x in calculatedTheta], jTheta, 'bo-')
    plt.savefig("minimizationPath.jpg")
    print("\tSaved the minimization path to minimizationPath.jpg.")


def predictValue(X, theta):
    """
        This function returns the predicted value
        using theta values calculated.
    :param X:    X vector
    :param theta:   theta vector
    :return:        predicted value
    """

    result = []
    for i in range(len(X)):
        result.append(theta[0] + theta[1] * X[i][1])
    return result

def accuracy(actualResult, predictedResult):
    """
        This function calculates the accuracy of the results obtained.
    :param actualResult:    the results of inbuilt function
    :param predictedResult: the results of user defined function
    :return:                accuracy of the model
    """
    return (100-abs(numpy.mean(((actualResult-predictedResult)/actualResult)*100)))

def main():
    """
        This is the main function.
    :return: None
    """
    print("******************* Starting execution **********************")

    # Read the data
    data = numpy.loadtxt('data/foodTruck.txt', delimiter=',', usecols=(0, 1), unpack=True)
    print("\nSuccessfully read the data.")

    # ***************************************** Step1: Initial data *****************************************
    print("Getting the data ready.")
    # X matrix
    X = numpy.transpose(numpy.array(data[:-1]))
    # y vector
    y = numpy.transpose(numpy.array(data[-1:]))
    # no of training examples
    m = y.size

    # Insert a column of 1's into the X matrix
    X = numpy.insert(X, 0, 1, axis=1)

    # Split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X = X_train
    y = y_train

    # plot initial data and save it
    plotInitialData(X, y)

    # ******************* Step 2: calculate the cost function values***********************
    # set number of iterations
    iterations = 5000
    # set the learning rate
    alpha = 0.01

    # running computeCost will return 32.07 on value of 0
    # initial_theta = numpy.zeros((X.shape[1], 1))
    # print(costFunction(initial_theta, X, y))

    # run gradient descent algorithm to find best theta values
    initial_theta = numpy.zeros((X.shape[1], 1))
    theta, calculatedTheta, jTheta = gradientDescent(X, y, alpha, iterations, initial_theta)

    # Plot the Convergence graph
    plotConvergence(jTheta)

    # Plot the path followed for minimization
    plotMinimizationPath(X, y, calculatedTheta, jTheta)

    # in built linear regression function
    linearRegressionVar = LinearRegression()
    linearRegressionVar.fit(X, y)

    # ******************* Step 4: test the calculated theta values ***********************

    print("\n\t __________________________ Results __________________________")

    # results from user defined function
    userFuncResult = predictValue(X_test, theta)

    # results from in built function
    builtInFuncResults = linearRegressionVar.predict(X_test)

    # Plot the hypothesis line by user defined function
    plotHypothesis(X_test, y_test, userFuncResult, "userDefinedHypothesis.jpg")

    # Plot the hypothesis line by in built function
    plotHypothesis(X_test, y_test, builtInFuncResults, "inBuiltHypothesis.jpg")

    print("\n\tFinal result theta parameters: ")
    print("\t\tUsing user function: ", theta[0][0], theta[1][0])
    print("\t\tUsing built in function: ", str(linearRegressionVar.intercept_[0]), str(linearRegressionVar.coef_[0][1]))

    print("\n\tBuilt in VS User function:")
    print("\t\tRoot Mean Squared error for Built in function: ",
          numpy.sqrt(metrics.mean_squared_error(y_test, builtInFuncResults)))
    print("\t\tRoot Mean Squared error for user function: ",
          numpy.sqrt(metrics.mean_squared_error(y_test, userFuncResult)))
    print("\t\tRoot Mean Squared error between user function and built-in function: ",
          numpy.sqrt(metrics.mean_squared_error(builtInFuncResults, userFuncResult)))

    print("\n\t\tAccuracy of the model from in-built functions: ",accuracy(y_test,builtInFuncResults))
    print("\n\t\tAccuracy of the model from user defined function: ",accuracy(y_test,userFuncResult))

    print("\n******************* Exiting **********************")


if __name__ == '__main__':
    main()
