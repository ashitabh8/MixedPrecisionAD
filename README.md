### Mixed Precision Automatic Differentiation

All the data used for Linear Regression is at: LinearRegression/data
All the results reported are in: LinearRegression/results

To run the model:  
```
cd LinearRegression  
gcc LinearRegression.cpp -lstdc++ -I ../include -w  
./a.out    
```

To run the root solver:  
```
cd NewtonRaphsonMethod  
gcc NewtonRaphson.cpp -I ../include -lstdc++ -w  
./a.out   
```

### Mixed AD Interface

```
Constant<fpm::fixed_16_16> c(1.2); // Defining a constant
Variable<float> v_1(2); // Defining a floating point variable with 2 as the initial value
Variable<float> v_2(4); 
Variable<float> v_copy;
std::vector<Variable<float>> vec{v_1, v_2};
    {
        Variable<float> x = getsum(vec,0); // Copy construction works in this library. getsum() is a helper function LogisticRegression.cpp
        v_copy = x; // x would die in this scope. So If you want to access it outside you should use new Variable<T>(0) and access the pointers.
                    // There are a lot of examples of this in LogisticRegression.cpp
    }
    
Variable<fpm::fixed_16_16> z = v_copy; // v_copy cannot be used after that. v_copy has been detached from the computational graph at this point.
```

Known Issues:  

Sometimes there is a segmentation fault  while running Regression but that occurs after the training has taken place during a clean up function  
so it can be safely ignored for the purposes of the results.


## Credits

For Fixed Point Support: https://github.com/MikeLankamp/fpm  
For Half-Floating Point Support: http://half.sourceforge.net/
