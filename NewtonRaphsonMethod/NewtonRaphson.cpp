// #include "iostream"
#include "../include/toyAD.h"
#include "../include/fixed.hpp"
#include "../include/ios.hpp"
#include "../include/math.hpp"
#include <cstdlib>
#include <vector>
#include <ctime>
#include <random>
#include <fstream>
#include <iostream>



// template <class... T>
// class Sampler
// {
//   public:
//     Sampler

// };

template <typename T>
void solver(T &expr)
{
  // T expr;
  std::cout << expr.getValue();
}

template <typename T>
T NewtonSolver(Expression<T> &expr,Variable<T> &var)
{
  // static void result;
  // result = static_cast<T> expr.getValue();

  T x_1 = var.getValue() - expr.getValue()/expr.diff(var);
  return x_1;
}


int main()
{
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(-100, 100);


  std::fstream file;
  file.open("test3.txt",std::ios_base::out);
  // std::random_device rd;

  // std::mt19937 gen(rd());
  std::discrete_distribution<> d({70, 30});
  Expression<float> expr;
  Variable<float> v(8); //Need to initialise with some value currently
  Constant<float> one(1);
  Constant<float> A(64);
  Constant<float> c(2);
  expr <<= (v-A)*(v+A)*(v-one); // 4, 4, 1

  Expression<fpm::fixed_16_16> expr_fixed;
  Variable<fpm::fixed_16_16> v_fixed(8); //Need to initialise with some value currently
  Constant<fpm::fixed_16_16> one_fixed(1);
  Constant<fpm::fixed_16_16> A_fixed(64);
  Constant<fpm::fixed_16_16> c_fixed(2);
  expr_fixed <<= (v_fixed-A_fixed)*(v_fixed+A_fixed)*(v_fixed-one_fixed);

  // void * result;

  fpm::fixed_16_16 result;
  float tolerable_error = 0.01;
  fpm::fixed_16_16 fp_tolerable_error{0.01};

  std::vector<fpm::fixed_16_16> roots;

  int max_iters = 10000;
  int i =0;

  std::map<std::string, int> dist_count;
  dist_count["float"] =0;
  dist_count["fixed"] = 0;

  do
  {
    if(i> max_iters)
    {
      break;
    }
    if(d(gen))
    {
      dist_count["float"]+=1;
      float temp = NewtonSolver(expr, v);
      v.setValue(temp);
      v_fixed.setValue(static_cast<fpm::fixed_16_16>(temp));
      // if(expr.getValue() <)
      float f_x_1 = expr.getValue();
      if(std::abs(f_x_1) < tolerable_error){
        roots.push_back(static_cast<fpm::fixed_16_16>(temp));
        int rd_temp = distr(gen);
        v.setValue(rd_temp);
        v_fixed.setValue(static_cast<fpm::fixed_16_16>(rd_temp));
        // break;
      }
    }
    else
    {
      dist_count["fixed"]+=1;
      fpm::fixed_16_16 temp = NewtonSolver(expr_fixed, v_fixed);
      v_fixed.setValue(temp);
      v.setValue(static_cast<float>(temp));
      if(fpm::abs(expr_fixed.getValue()) < fp_tolerable_error)
      {
        std::cout << "Current fixed f(x1): " << fpm::abs(expr_fixed.getValue()) << ", root: " << temp << std::endl;
        
        roots.push_back(temp);
        int rd_temp = distr(gen);
        v.setValue(rd_temp);
        v_fixed.setValue(static_cast<fpm::fixed_16_16>(rd_temp));
        // break;
      }
    }
    i++;
  } while (true);

  // std::cout << "Floadist_count << std::endl;
  for(auto const &[key, value]: dist_count)
  {
    std::cout << key << ": " << value <<std::endl ;
  }
  for (auto root: roots)
  {
    file<< root <<std::endl;
  }
  file.close();
  

  // std::cout << "Value: " << expr.getValue() << "\n";
  // v.setValue(10);
  // // std::cout << "Value: " << expr.getValue() << "\n";
  // std::cout << "Diff: " << expr.diff(v) << "\n";

  // std::cout << "Fixed: Value: " << expr_fixed.getValue() << "\n";
  // std::cout << "Fixed: Diff: " << expr_fixed.diff(v_fixed) << "\n";

}


