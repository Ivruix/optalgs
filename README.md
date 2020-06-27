# optalgs
This is a lightweight single header C++ library for derivative-free optimization.

## Installation
`optalgs` is a header only library. Just add [optalgs.h](https://github.com/Ivruix/optalgs/blob/master/src/optalgs.h) header file to your project.

## Implemented algorithms
* Differential Evolution (DE)
* Particle Swarm Optimization (PSO)
* Pattern Search (PS) 

## Usage
Let's use `optalgs` to find the global minimum of [Bukin function N.6](http://benchmarkfcns.xyz/benchmarkfcns/bukinn6fcn.html). This is a 2D test function that has the global minimum at x = -10 and y = 1, where it evaluates to 0, and the input domain of x ∈ [-15, -5] and y ∈ [-3, 3]. Below is a plot of this function.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bukin_function_6.pdf/page1-800px-Bukin_function_6.pdf.jpg)

We will use Pattern Search with population size set to 200 and number of iteration set to 100.
```cpp
#include <vector>
#include <utility>
#include "optalgs.h"

double BukinFunction6(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    return 100 * sqrt(abs(y - 0.01 * x * x)) + 0.01 * abs(x + 10);
}

int main() {
    std::vector<std::pair<double, double>> bounds = {{-15.0, -5.0}, {-3.0, 3.0}};
    PatternSearch ps(BukinFunction6, bounds, 200);
    ps.Optimize(100, VerbosityLevels::High, 25);
    return 0;
}
```
This code gives us the following output:
```
Iteration: 25
Mean step size: 7.18164e-05
Mean cost: 0.206575
Best cost: 0.0194928
Best agent: {-9.48921, 0.900452}

Iteration: 50
Mean step size: 9.57571e-10
Mean cost: 0.0243959
Best cost: 0.000773605
Best agent: {-9.94863, 0.989753}

Iteration: 75
Mean step size: 1.62396e-14
Mean cost: 0.0237922
Best cost: 2.13159e-05
Best agent: {-10.0017, 1.00033}

Iteration: 100
Mean step size: 1.80116e-20
Mean cost: 0.0237902
Best cost: 1.66038e-05
Best agent: {-10.0017, 1.00033}
```
You can also make your own custom termination conditions:
```cpp
#include <vector>
#include <utility>
#include "optalgs.h"

double BukinFunction6(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    return 100 * sqrt(abs(y - 0.01 * x * x)) + 0.01 * abs(x + 10);
}

int main() {
    std::vector<std::pair<double, double>> bounds = {{-15.0, -5.0}, {-3.0, 3.0}};
    PatternSearch ps(BukinFunction6, bounds, 200);
    while((ps.GetBestCost() > 0.1) && (ps.GetCurrentIteration() < 100)) {
        ps.Optimize(1, VerbosityLevels::Silent);
    }
    ps.PrintBestAgent(true); // additionaly prints cost of best agent
    return 0;
}
```
This code prints the first agent found with a cost less than 0.1 and it's cost:
```
{-8.54993, 0.731013} : 0.073437
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
