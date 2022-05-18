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

### Examples

```
void simpleConstantsTest()
{
    // Test Constants
    Constant<fpm::fixed_16_16> A(5.453);
    std::cout << "Constant A ID: " << A.get_unq_node_idx() << std::endl;
    std::cout << A.getValue<float>() << std::endl; // Get in a specific type.
    std::cout << A.getValue() <<std::endl; // Get value in the type it was instantiated.
    A.setValue(32.1232);
    std::cout << A.getValue<float>() << std::endl;
}

void simpleVariableTests()
{
    Variable<float> v_1(2.3);
    Variable<float> v_2(4.6);
    std::cout << v_1.getValue() << std::endl;
    std::cout << "V_1 ID: " << v_1.get_unq_node_idx() << std::endl;
    std::cout << v_1.diff(v_1.get_unq_node_idx())<<std::endl;
    std::cout << v_1.diff(v_2.get_unq_node_idx())<<std::endl;
}

int simpleExpressionTests()
{
    Constant<float> c_1(100);
    Variable<float> v_1(2.5);
    Variable<float> v_2(4.5);
    Variable<float> z = (v_1+v_2);
    Variable<float> a = z+v_1+c_1+v_1;
    std::cout <<"Should be 7: "<< z.getValue()<<std::endl;
    std::cout <<"Should be 112: "<< a.getValue()<<std::endl;
    std::cout <<"Should be 112: "<< a.getValue()<<std::endl;
}

```


## Credits

For Fixed Point Support: https://github.com/MikeLankamp/fpm  
For Half-Floating Point Support: http://half.sourceforge.net/
