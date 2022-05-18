#include "shuffleAD.h"
#include "include/half.hpp"

template<typename T>
Variable<T> &getsum(std::vector<Variable<T>> &input, unsigned counter)
{
    if(input.size() - 1 == counter)
    {
        return input.at(counter);
    }
    // Variable<T> *sum = new Variable<T>(0);
    return input.at(counter) + getsum(input, counter+1);
    // return sum;
}

// void simpleConstantsTest()
// {
//     // Test Constants
//     Constant<fpm::fixed_16_16> A(5.453);
//     std::cout << "Constant A ID: " << A.get_unq_node_idx() << std::endl;
//     std::cout << A.getValue<float>() << std::endl; // Get in a specific type.
//     std::cout << A.getValue() <<std::endl; // Get value in the type it was instantiated.
//     A.setValue(32.1232);
//     std::cout << A.getValue<float>() << std::endl;
// }

// void simpleVariableTests()
// {
//     Variable<float> v_1(2.3);
//     Variable<float> v_2(4.6);
//     std::cout << v_1.getValue() << std::endl;
//     std::cout << "V_1 ID: " << v_1.get_unq_node_idx() << std::endl;
//     std::cout << v_1.diff(v_1.get_unq_node_idx())<<std::endl;
//     std::cout << v_1.diff(v_2.get_unq_node_idx())<<std::endl;
// }

// int simpleExpressionTests()
// {
//     Constant<float> c_1(100);
//     Variable<float> v_1(2.5);
//     Variable<float> v_2(4.5);
//     Variable<float> z = (v_1+v_2);
//     Variable<float> a = z+v_1+c_1+v_1;
//     std::cout <<"Should be 7: "<< z.getValue()<<std::endl;
//     std::cout <<"Should be 112: "<< a.getValue()<<std::endl;
//     std::cout <<"Should be 112: "<< a.getValue()<<std::endl;
// }

// void simpleDiffwFP()
// {
//     Constant<fpm::fixed_16_16> c_1(100);
//     Variable<fpm::fixed_16_16> v_1(2);
//     Variable<fpm::fixed_16_16> v_2(4);
//     Variable<fpm::fixed_16_16> x;
//     x = v_1;
//     Variable<fpm::fixed_16_16>z  = x * v_1;
//     std::cout << "x NODE ID: " << x.get_unq_node_idx() <<std::endl;
//     std::cout << "v_1 NODE ID: "  << v_1.get_unq_node_idx() << std::endl;
//     std::cout << "z NODE ID: "  << z.get_unq_node_idx() << std::endl;
//     std::cout << "Value should be 2: " << x.diff(v_2) <<std::endl;
//     std::cout << "Value should be 2: " << z.diff(v_1) <<std::endl;
// }


template<typename T>
Variable<T> &sum(Variable<T> &a, Variable<T> &b)
{
    return a+b;
}

// Run this file to see whether everything is working properly.
int main()
{
    // Variable<float> v_1(2);
    // Variable<float> v_2(4);
    // Variable<float> v_copy;
    // std::vector<Variable<float>> vec{v_1, v_2};
    // {
    //     Variable<float> x = getsum(vec,0);
    //     v_copy = x;
    // }
    
    // // // // Variable<fpm::fixed_16_16> z = getsum(vec, 0);
    // std::cout << "Value: " << v_copy.getValue() << std::endl;

    // using fixed_40_24 = fpm::fixed<std::int_fast32_t, std::int64_t, 16>;

    // fixed_40_24 x(1000);
    // fixed_40_24 y(1000);
    // // fixed_40_24 y(32.8459292);
    // std::cout << "Check value: " << std::numeric_limits<fixed_40_24>::max() << std::endl;
    // std::cout << "Check this: " << x*y << std::endl;

    Constant<half_float::half> c(9);
    Variable<half_float::half> v(2);
    Variable<half_float::half> z = c * v;
    std::cout<< z.getValue() << " Half floating value\n";
}