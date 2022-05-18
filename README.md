### Mixed Precision Automatic Differentiation

All the data used for Linear Regression is at: LinearRegression/data
All the results reported are in: LinearRegression/results

To run the model:  
cd LinearRegression  
gcc LinearRegression.cpp -lstdc++ -I ../include -w  
./a.out    

To run the root solver:  
cd NewtonRaphsonMethod  
gcc NewtonRaphson.cpp -I ../include -lstdc++ -w  
./a.out   


Known Issues:  

Sometimes there is a segmentation fault  while running Regression but that occurs after the training has taken place during a clean up function
so it can be safely ignored for the purposes of the results.


## Credits

For Fixed Point Support: https://github.com/MikeLankamp/fpm  
For Half-Floating Point Support: http://half.sourceforge.net/
