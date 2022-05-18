#include "../include/shuffleAD.h"
#include <iostream>
#include "../include/rapidcsv.h"
#include <string>
#include <random>
#include <typeinfo>
#include "../include/half.hpp"

template<typename T>
std::vector<std::vector<Constant<T>*>> data(std::string filename, std::vector<std::string> &input_labels, std::string output_label)
{
    std::cout << " Reading file...";
    rapidcsv::Document Dataset(filename);
    std::cout << " Read file...";

    std::vector<std::vector<Constant<T>*>> dataset;
    // getting only two columns for one variable linear regression for starters.
    for(std::string curr_input_label: input_labels)
    {
        std::vector<float> curr = Dataset.GetColumn<float>(curr_input_label);
        std::vector<Constant<T>*> curr_type_T;
        for(float elem: curr)
        {
            curr_type_T.push_back(new Constant<T>(elem));
        }
        dataset.push_back(curr_type_T);
    }
    // std::vector<float> age = Dataset.GetColumn<float>("age");
    std::vector<float> medv = Dataset.GetColumn<float>(output_label);

    // std::cout << "Age(float): " << age.at(2) <<std::endl;
    // std::cout << "MEDV(float): " << medv.at(2) <<std::endl;

    // std::vector<Constant<T>> param_age;
    std::vector<Constant<T>*> output; // y
    // for(auto curr_age :age)
    // {
    //     param_age.push_back(Constant<T>(curr_age));
    // }
    for(auto curr_medv :medv)
    {
        output.push_back(new Constant<T>(curr_medv));
    }

    // std::vector<std::vector<Constant<T>>> dataset;
    // dataset.push_back(param_age);
    dataset.push_back(output);

    return dataset;
}

template<typename T>
std::pair<std::vector<Variable<T>*>, Variable<T>*> getInitWeights(std::vector<std::vector<Constant<T>*>> &dataset)
{
    
    Variable<T> *bias = new Variable<T>(2);
    size_t num_inputs = dataset.at(0).size();
    size_t num_weights = dataset.size() -1;
    // Weights initialised by 0
    std::vector<Variable<T>* > weights;
    for(unsigned j = 0; j < num_weights ; j++)
    {
        weights.push_back(new Variable<T>(-1));
    }
    
    std::cout << "Number of rows: "<< dataset.back().size()<<std::endl;
    std::cout << "Get num weights: " << weights.size()<<std::endl;
    return std::pair<std::vector<Variable<T>*>, Variable<T>*>{weights, bias};
}

template<typename T>
Variable<T> &getsum(std::vector<Variable<T> *> &input, unsigned counter)
{
    if(input.size() - 1 == counter)
    {
        return *(input.at(counter));
    }
    // Variable<T> *sum = new Variable<T>(0);
    return *(input.at(counter)) + getsum(input, counter+1);
    // return sum;
}

template<typename T>
Variable<T> * makeErrorFunction(std::vector<std::vector<Constant<T>*>> &dataset,std::vector<Variable<T> *> &weights, Variable<T> *bias)
{
    // Constant<T> d(1/(dataset.back().size()));
    // Variable<T> Error;
    std::vector<Constant<T>*> Y = dataset.back();
    size_t NUM_OF_POINTS = Y.size();
    size_t DIMS = weights.size();
    

    // This gets y_hat = [mx_1+c, mx_2+c , ...]
    
    std::vector<std::vector<Variable<T> *>> mx_temp;
    // std::vector<Variable<T>> weighted_sum;
    for(unsigned j =0; j < NUM_OF_POINTS; j++)
    {
        std::vector<Variable<T> *> temp;
        for(unsigned k =0; k < DIMS; k ++)
        {
            temp.push_back(new Variable<T>(0));
        }
        for(unsigned k =0; k < DIMS; k ++)
        {
            Variable<T> t_val = (*weights.at(k)) * (*dataset.at(k).at(j));
            *(temp.at(k)) = t_val;
            // temp.push_back(weights.at(k) * dataset.at(k).at(j));
        }
        mx_temp.push_back(temp);
    }

    // return mx_temp;

    // std::cout << "Running mx...\n";
    // for(auto curr_row: mx_temp)
    // {
    //     for(auto mx : curr_row)
    //     {
    //         std::cout << mx.getValue() << " ";
    //     }
    //     std::cout << "\n";
    // }
    

   
    std::vector<Variable<T> *> weighted_sum;
    // Initialise weighted_sum
    for(unsigned i =0; i < mx_temp.size() ; i++)
    {
        weighted_sum.push_back(new Variable<T>(0));
    }

    for(unsigned i =0; i <  mx_temp.size() ;i++)
    {
        Variable<T> x = getsum(mx_temp.at(i),0);
        *(weighted_sum.at(i)) = x;
    }
    // return weighted_sum;
    //  std::cout << "Check weighted sum: " << std::endl;
    // for(auto elem: weighted_sum)
    // {
    //     std::cout << elem.getValue() << std::endl;
    // }
    std::vector<Variable<T> *> Y_hat;
    for(unsigned i =0; i < weighted_sum.size(); i++)
    {
        Y_hat.push_back(new Variable<T>(0));
    }
    for(unsigned i =0; i < weighted_sum.size(); i++)
    {
        
        Variable<T> temp = *(weighted_sum.at(i))+(*bias);
        *(Y_hat.at(i)) = temp;
    }
    // std::cout << "Check Y hats..."<<std::endl;
    // for(auto elem: Y_hat)
    // {
    //     std::cout << elem.getValue() << std::endl;
    // }

    // // std::cout << " Checking weighted sum...2\n";
    // // std::cout << weighted_sum.back().getValue()<<std::endl;



    
    
    // // assert(all_Y_hats.size() == Y.size());
    // // std::cout << "Checking Y hats..." << std::endl;
    // // for(auto elems: all_Y_hats)
    // // {
    // //     std::cout << elems.getValue() << std::endl;
    // // }
    // // //get Y-Y_hat;
    std::vector<Variable<T>*> Y_minus_Y_hat;
    for(unsigned i = 0; i < NUM_OF_POINTS; i++)
    {
        Y_minus_Y_hat.push_back(new Variable<T>(0));
    }

    // // std::vector<Variable<T>> Y_minus_Y_hat;
    for(unsigned i = 0; i < NUM_OF_POINTS; i++)
    {
        Variable<T> x = *(Y.at(i)) - *(Y_hat.at(i));
        *(Y_minus_Y_hat.at(i)) = x;
    }
    // std::cout << " Check Y - Y_hat \n";
    // for(auto elem: Y_minus_Y_hat)
    // {
    //     std::cout << elem.getValue() << std::endl;
    // }
    // // // get (Y-Y_hat)^2
    std::vector<Variable<T> *> Y_minus_Y_hat_square;
    for(unsigned i = 0; i < NUM_OF_POINTS; i++)
    {
        Y_minus_Y_hat_square.push_back(new Variable<T>(0));
    }
    for(unsigned i = 0; i < NUM_OF_POINTS; i++)
    {
        Variable<T> x = (*(Y_minus_Y_hat.at(i)))*(* Y_minus_Y_hat.at(i));
        *(Y_minus_Y_hat_square.at(i)) = x;
    }

    // return Y_minus_Y_hat_square;
    // std::cout << " Check (Y - Y_hat)^2 \n";
    // for(auto elem: Y_minus_Y_hat_square)
    // {
    //     std::cout << elem.getValue() << std::endl;
    // }

    // // Sum of squares
    Variable<T> *sum_of_squares = new Variable<T>(0);
    Variable<T> temp = getsum(Y_minus_Y_hat_square, 0);
    *sum_of_squares = temp;
    // return sum_of_squares;
    // // std::cout << "Sum of squares value: " << sum_of_squares->getValue() <<std::endl;
    int num_of_points = static_cast<int>(NUM_OF_POINTS);
    float d_temp = 1.0/num_of_points;
    Constant<T> * d = new Constant<T>(d_temp);
    Variable<T> *final_result_error = new Variable<T>(0);
    *final_result_error = (*d) * (*sum_of_squares);

    return final_result_error;
}

template<typename T>
void fit(Variable<T> *Error,std::vector<Variable<T>*> weights,Variable<T> *Bias, T learning_rate, T threshold, int max_iterations=1000)
{
    // std::cout << "Initial loss :" << Error->getValue() << std::endl;
    // int max_iterations = 10000;
    int curr_iter = 0;
    T curr_error;
    do{
        // std::cout << curr_iter; 
        std::vector<T> new_weights;
        T new_bias;
        for(Variable<T>* curr_weight : weights)
        {
            T m_prev = curr_weight->getValue();
            // std::cout << "previous weight: " << m_prev << std::endl;
            T D = Error->diff(*curr_weight);
            // std::cout << "Loss : " << D << std::endl;
            T m_new = m_prev - learning_rate * D;
            // std::cout << "New weight walue : " << m_new << std::endl;
            // curr_weight->setValue(m_new);
            new_weights.push_back(m_new);
        }

        T bias_prev = Bias->getValue();
        // std::cout << "previous bias: " << bias_prev << std::endl;
        T D_bias = Error->diff(*Bias);
        // std::cout << "Loss : " << D_bias << std::endl;
        T bias_new = bias_prev - learning_rate * D_bias;
        // std::cout << "New weight bias : " << bias_new << std::endl;
        Bias->setValue(bias_new);
        for(unsigned i =0; i < weights.size(); i++)
        {
            weights.at(i)->setValue(new_weights.at(i));
        }
        curr_iter++;
        curr_error = Error->getValue();
        std::cout << ", " << curr_error;
        std::cout << ", " << typeid(T).name() << std::endl;
    }while(curr_error > threshold && curr_iter < max_iterations);
}

template<typename T>
void weight_update(Variable<T> *Error,std::vector<Variable<T>*> &weights,Variable<T> *Bias, T learning_rate)
{
    // std::cout << "Initial loss :" << Error->getValue() << std::endl;
    // int max_iterations = 10000;
    // static int curr_iter = 0;
    T curr_error;
        // std::cout << curr_iter; 
        std::vector<T> new_weights;
        T new_bias;
        for(Variable<T>* curr_weight : weights)
        {
            T m_prev = curr_weight->getValue();
            // std::cout << "previous weight: " << m_prev << std::endl;
            T D = Error->diff(*curr_weight);
            // std::cout << "Loss : " << D << std::endl;
            T m_new = m_prev - learning_rate * D;
            // std::cout << "New weight walue : " << m_new << std::endl;
            // curr_weight->setValue(m_new);
            new_weights.push_back(m_new);
        }

        T bias_prev = Bias->getValue();
        // std::cout << "previous bias: " << bias_prev << std::endl;
        T D_bias = Error->diff(*Bias);
        // std::cout << "Loss : " << D_bias << std::endl;
        T bias_new = bias_prev - learning_rate * D_bias;
        // std::cout << "New weight bias : " << bias_new << std::endl;
        Bias->setValue(bias_new);
        for(unsigned i =0; i < weights.size(); i++)
        {
            weights.at(i)->setValue(new_weights.at(i));
        }
        // curr_iter++;
        // std::cout<< curr_iter;
        // curr_iter++;
        curr_error = Error->getValue();
        std::cout << curr_error;
        std::cout << ", " << typeid(T).name() << std::endl;
}

// void test
//Gradient Descent - Single Precision Implementation
template<typename T>
void Boston()
{
    std::cout<< "works\n";
    // crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat,medv
    std::vector<std::string> input_labels{"crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax", "ptratio", "b","lstat"};
    std::vector<std::vector<Constant<T> *>> dataset = data<T>(std::string("./data/Boston.csv"), input_labels,std::string("medv"));
    std::pair<std::vector<Variable<T>*>, Variable<T>*> weights_bias =getInitWeights<T>(dataset);
    std::vector<Variable<T>*> Weights = weights_bias.first;
    Variable<T> *Bias = weights_bias.second;

    Variable<T> * Error = makeErrorFunction(dataset,Weights, Bias);

    // std::cout << "Returned Error Value: " << sum_error->getValue() << std::endl;
    T threshold{2};
    T learning_rate{0.001};
    fit(Error, Weights,Bias, learning_rate, threshold,1000);
}

// Boston Gradient Descent Pick different precisions


// Batch gradient descent
template<typename T1, typename T2, int P1, int P2, int Batchsize>
void Boston_2()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({P1, P2});
    std::vector<std::string> input_labels{"crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax", "ptratio", "b","lstat"};
    // std::vector<std::string> input_labels{"label_1"};
    std::vector<std::vector<Constant<T1> *>> dataset_t1 = data<T1>(std::string("./data/Boston.csv"), input_labels,std::string("medv"));
    std::vector<std::vector<Constant<T2> *>> dataset_t2 = data<T2>(std::string("./data/Boston.csv"), input_labels,std::string("medv"));

    
    std::pair<std::vector<Variable<T1>*>, Variable<T1>*> weights_bias_t1 =getInitWeights<T1>(dataset_t1);
    std::vector<Variable<T1>*> Weights_t1 = weights_bias_t1.first;
    Variable<T1> *Bias_t1 = weights_bias_t1.second;
    Variable<T1> * Error_t1 = (makeErrorFunction(dataset_t1,Weights_t1, Bias_t1));
    T1 threshold_t1{2};
    T1 learning_rate_t1{0.001};

    unsigned NUM_Weights = Weights_t1.size();
    std::pair<std::vector<Variable<T2>*>, Variable<T2>*> weights_bias_t2 =getInitWeights<T2>(dataset_t2);
    std::vector<Variable<T2>*> Weights_t2 = weights_bias_t2.first;
    Variable<T2> *Bias_t2 = weights_bias_t2.second;
    Variable<T2> * Error_t2 = (makeErrorFunction(dataset_t2,Weights_t2, Bias_t2));
    T2 threshold_t2{2};
    T2 learning_rate_t2{0.001};

    int random_pick = 0; // 0 or 1
    int prev_pick = random_pick;
    int max_iterations = 1000;
    int curr_iters = 0;

    int num_T1 = 0;
    int num_T2 = 0;

    do{
        std::cout << curr_iters;
        random_pick = d(gen);
        if(random_pick){ // 1 means pick T2
        // std::cout << " Running T2\n";
        num_T2++;
        if(prev_pick != random_pick) // means previous one was T1 so update weights;
        {
            prev_pick = random_pick;
            for(unsigned i =0 ; i<NUM_Weights; i++)
            {
                Weights_t2.at(i)->setValue(Weights_t1.at(i)->getValue());
            }
        }
        fit(Error_t2, Weights_t2,Bias_t2, learning_rate_t2, threshold_t2,Batchsize);
        T2 curr_error = Error_t2->getValue();
        // std::cout << " Error recieved in T2: " << curr_error << std::endl;
        if(curr_error < threshold_t2)
        {
            std::cout<<"Error below threshold... terminating."<<std::endl;
            break;
        }
            
    }
    else{ // means pick T1
    // std::cout << " Running T1\n";
    num_T1++;
        if(prev_pick != random_pick) // means previous one was T2 so update weights;
        {
            prev_pick = random_pick;
            for(unsigned i =0 ; i<NUM_Weights; i++)
            {
                // std::cout << " Value to convert: " << Weights_t2.at(i)->getValue() << std::endl;
                Weights_t1.at(i)->setValue(Weights_t2.at(i)->getValue());
                // std::cout << "Value received: " << Weights_t1.at(i)->getValue() << std::endl;
            }
        }
        fit(Error_t1, Weights_t1,Bias_t1, learning_rate_t1, threshold_t1,Batchsize);
        T1 curr_error = Error_t1->getValue();
        // std::cout << " Error recieved in T1: " << curr_error << std::endl;
        if(curr_error < threshold_t1)
        {
            std::cout<<"Error below threshold... terminating."<<std::endl;
            break;
        }
    }
    curr_iters++;
    }while(curr_iters < max_iterations);

    
}


template<typename T>
std::vector<std::vector<Constant<T> *>> getdataset(std::vector<std::vector<Constant<T> *>> &dataset, int minibatch_size)
{
    // std::cout << "Getting dataset...\n";
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t NUM_OF_ENTRIES = dataset.back().size();
    size_t NUM_OF_WEIGHTS = dataset.size();
    std::vector<std::vector<Constant<T> *>> final_mini_ds;
    std::vector<int> num_idx;
    for(unsigned j =0; j < NUM_OF_ENTRIES; j++)
    {
        num_idx.push_back(j);
    }

    std::shuffle(num_idx.begin(), num_idx.end(), gen);
    // std::cout << " Num of weights: " << NUM_OF_WEIGHTS << std::endl;
    // std::cout << " Num of entries: " << (int)NUM_OF_ENTRIES << std::endl;
    // std::cout << " Size of num_idx: " << num_idx.size() << std::endl;
    minibatch_size = std::min(minibatch_size, (int)NUM_OF_ENTRIES);
    for(unsigned j =0; j < NUM_OF_WEIGHTS; j++)
    {
        std::vector<Constant<T> *> curr_cols;
        for(int i =0; i < minibatch_size; i++)
        {
            int curr_row_id = num_idx.at(i);
            // std::cout << "Current row id: " << curr_row_id;
            curr_cols.push_back(dataset.at(j).at(curr_row_id));
        }
        final_mini_ds.push_back(curr_cols);
    }
    
    return final_mini_ds;
}

template<typename T>
void print_ds(std::vector<std::vector<Constant<T> *>> &dataset)
{
    std::cout << "Printing...." << std::endl;
    size_t NUM_OF_ENTRIES = dataset.back().size();
    size_t NUM_OF_WEIGHTS = dataset.size();
    for(unsigned i = 0; i < NUM_OF_ENTRIES; i++)
    {
        for(unsigned j = 0; j < NUM_OF_WEIGHTS; j++)
        {
            std::cout<< dataset.at(j).at(i)->getValue() << " ";
        }
        std::cout << "\n";
    }
}


void tester()
{
    std::vector<std::string> input_labels{"crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax", "ptratio", "b","lstat"};
    // std::vector<std::string> input_labels{"label_1"};
    std::vector<std::vector<Constant<float> *>> dataset_t1 = data<float>(std::string("./data/Boston.csv"), input_labels,std::string("medv"));
    std::vector<std::vector<Constant<float> *>> dataset_mini = getdataset(dataset_t1, 32);
    print_ds(dataset_mini);

}

template<typename T1, typename T2, int P1, int P2, int Batchsize_AD, int mini_batch_size>
std::pair<std::vector<Variable<float>*>, Variable<float>*> LinearRegression_fit(std::string datafile, std::vector<std::string> input_labels,std::string ouput_label, float threshold)
{
    std::cout << datafile;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({P1, P2});
    // std::vector<std::string> input_labels{"crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax", "ptratio", "b","lstat"};
    // std::vector<std::string> input_labels{"label_1"};
    std::vector<std::vector<Constant<T1> *>> dataset_t1 = data<T1>(datafile, input_labels,ouput_label);
    std::vector<std::vector<Constant<T2> *>> dataset_t2 = data<T2>(datafile, input_labels,ouput_label);
    std::pair<std::vector<Variable<T1>*>, Variable<T1>*> weights_bias_t1 =getInitWeights<T1>(dataset_t1);
    std::vector<Variable<T1>*> Weights_t1 = weights_bias_t1.first;
    Variable<T1> *Bias_t1 = weights_bias_t1.second;
    T1 threshold_t1{threshold};
    T1 learning_rate_t1{0.001};
    std::pair<std::vector<Variable<T2>*>, Variable<T2>*> weights_bias_t2 =getInitWeights<T2>(dataset_t2);
    std::vector<Variable<T2>*> Weights_t2 = weights_bias_t2.first;
    Variable<T2> *Bias_t2 = weights_bias_t2.second;
    T2 threshold_t2{threshold};
    T2 learning_rate_t2{0.001};
    unsigned NUM_Weights = Weights_t1.size();
    int max_iterations =1000;
    int num_T1 =0;
    int num_T2 =0;
    std::vector<Variable<float>*> final_weights;
    Variable<float> * final_bias = new Variable<float>(0);
    for(unsigned i =0; i < Weights_t1.size(); i++)
    {
        final_weights.push_back(new Variable<float>(0));
    }

    // Max iterations 10000
    // Pick a batch size
    // get the datasets in that batch size
    // weights have already been defined so they dont need to change
    // you want to make a new error function

    // std::pair<std::vector<Variable<T1>*>, Variable<T1>*> weights_bias_t1 =getInitWeights<T1>(dataset_t1);
    // std::vector<Variable<T1>*> Weights_t1 = weights_bias_t1.first;
    // Variable<T1> *Bias_t1 = weights_bias_t1.second;
    int prev_pick = -1;
    for(int k = 0 ; k < max_iterations; k++)
    {
        // std::cout << "Iteration number: "<< k<< ", ";
        int random_pick = d(gen); // 0 or 1
        if(random_pick == 0)
        {
            if(prev_pick != -1)
            {
                if(prev_pick != random_pick) // means previous one was T2 so update weights;
                {
                    prev_pick = random_pick;
                    // std::cout << "Converting weights(T2->T1)" << std::endl;
                    for(unsigned i =0 ; i<NUM_Weights; i++)
                    {
                        // std::cout << " Input value : " << Weights_t2.at(i)->getValue() << std::endl;
                        Weights_t1.at(i)->setValue(Weights_t2.at(i)->getValue());
                        // std::cout << " Output value: " << Weights_t1.at(i)->getValue() << std::endl;
                    }
                    // std::cout << " Input bias : " << Bias_t2->getValue() << std::endl;
                    Bias_t1->setValue(Bias_t2->getValue());
                    // std::cout << " output bias : " << Bias_t1->getValue() << std::endl;
                }
            }
            prev_pick = random_pick;
            std::vector<std::vector<Constant<T1> *>> dt_t1= getdataset(dataset_t1, mini_batch_size);
            // print_ds(dt_t1);
            Variable<T1> * Error_t1 = makeErrorFunction(dt_t1,Weights_t1, Bias_t1);
            // std::cout << " Error before update(T1): " << Error_t1->getValue() << std::endl;
            for(int j = 0; j < Batchsize_AD ; j++) // Run weight update in a single type for Batchsize_AD times.
            {
                weight_update(Error_t1, Weights_t1, Bias_t1, learning_rate_t1);
                k++;
                num_T1++;
            }
            T1 curr_error = Error_t1->getValue();
        // std::cout << " Error recieved in T2: " << curr_error << std::endl;
            if(curr_error < threshold_t1)
            {
                continue;
                for(unsigned i =0 ; i<NUM_Weights; i++)
                    {
                        // std::cout << " Input value : " << Weights_t2.at(i)->getValue() << std::endl;
                        final_weights.at(i)->setValue(Weights_t1.at(i)->getValue());
                        // std::cout << " Output value: " << Weights_t1.at(i)->getValue() << std::endl;
                    }
                    // std::cout << " Input bias : " << Bias_t2->getValue() << std::endl;
                    final_bias->setValue(Bias_t1->getValue());
                std::cout<<"Error below threshold... terminating."<<std::endl;
                std::cout << " TYPE 1 num of iterations: " << num_T1<<std::endl;
                std::cout << " TYPE 2 num of iterations: " << num_T2<<std::endl;
                return std::pair<std::vector<Variable<float>*>, Variable<float>*>(final_weights, final_bias);
                // break;
            }
        }
        if(random_pick == 1)
        {
            if(prev_pick!= -1)
            {
                if(prev_pick != random_pick) // means previous one was T1 so update weights;
                {
                    prev_pick = random_pick;
                    // std::cout << "Converting weights(T1->T2): " <<std::endl;
                    for(unsigned i =0 ; i<NUM_Weights; i++)
                    {
                        // std::cout << "Input value: " << Weights_t1.at(i)->getValue();
                        Weights_t2.at(i)->setValue(Weights_t1.at(i)->getValue());
                        // std::cout << ", Output value: " << Weights_t2.at(i)->getValue() << std::endl;
                    }
                    // std::cout << " Input bias : " << Bias_t1->getValue() << std::endl;
                    Bias_t2->setValue(Bias_t1->getValue());
                    // std::cout << " Output bias : " << Bias_t2->getValue() << std::endl;
                }
            }
            prev_pick = random_pick;
            std::vector<std::vector<Constant<T2> *>> dt_t2= getdataset(dataset_t2, mini_batch_size);
            // print_ds(dt_t2);
            Variable<T2> * Error_t2 = makeErrorFunction(dt_t2,Weights_t2, Bias_t2);
            // std::cout << " Error before update(T2): " << Error_t2->getValue() << std::endl;
            for(int j = 0; j < Batchsize_AD ; j++) // Run weight update in a single type for Batchsize_AD times.
            {
                weight_update(Error_t2, Weights_t2, Bias_t2, learning_rate_t2);
                k++;
                num_T2++;
            }

            T2 curr_error = Error_t2->getValue();
        // std::cout << " Error recieved in T2: " << curr_error << std::endl;
            if(curr_error < threshold_t2)
            {
                continue;
                for(unsigned i =0 ; i<NUM_Weights; i++)
                    {
                        // std::cout << " Input value : " << Weights_t2.at(i)->getValue() << std::endl;
                        final_weights.at(i)->setValue(Weights_t2.at(i)->getValue());
                        // std::cout << " Output value: " << Weights_t1.at(i)->getValue() << std::endl;
                    }
                    // std::cout << " Input bias : " << Bias_t2->getValue() << std::endl;
                    final_bias->setValue(Bias_t2->getValue());
                std::cout<<"Error below threshold... terminating."<<std::endl;
                std::cout << " TYPE 1 num of iterations: " << num_T1<<std::endl;
                std::cout << " TYPE 2 num of iterations: " << num_T2<<std::endl;
                return std::pair<std::vector<Variable<float>*>, Variable<float>*>(final_weights, final_bias);
                // std::cout<<"Error below threshold... terminating."<<std::endl;
                // break;
            }
        }
    }
    
}


template<typename T>
void test(std::string datafile, std::vector<std::string> &input_labels, std::string output_label,std::vector<Variable<T>*> &weights, Variable<T>*bias )
{
    std::cout << " Testing..." << std::endl;
    std::vector<std::vector<Constant<T> *>> dataset = data<T>(datafile, input_labels,output_label);
    Variable<T>* Error = makeErrorFunction(dataset,weights, bias);

    std::cout << " Test error: " << Error->getValue() << std::endl;
}


int main()
{

    std::cout << "Running Boston Dataset...\n";
    std::cout << "Running Batch Gradient Descent so training might take sometime...\n";
    Boston_2<float, fpm::fixed_16_16,50,50,1>(); // Batchsize =1 -- stochastic precision selection





    // std::string datafile = "./data/Boston.csv";
    // std::vector<std::string> input_labels = {"crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax", "ptratio", "b","lstat"}; 
    // std::string output = "medv";
    // std::pair<std::vector<Variable<float>*>, Variable<float>*> final_weights = LinearRegression_fit<fpm::fixed_24_8, float, 50,50,2,32>(datafile,input_labels,output,0.1); // int Batchsize_AD, int mini_batch_size



    std::cout << "Running Life Expectancy Dataset with Half16 and Float32\n";
    std::string datafile_life = "./data/life_expectancy.csv";
    
    std::vector<std::string> input_labels = {"Country", "Year", "Status", "Adult Mortality",
       "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B",
       "Measles ", " BMI ", "under-five deaths ", "Polio", "Total expenditure",
       "Diphtheria ", " HIV/AIDS", "GDP", "Population",
       " thinness  1-19 years", " thinness 5-9 years",
       "Income composition of resources", "Schooling"};
    std::string output = "Life expectancy ";

    LinearRegression_fit<half_float::half, float, 95,5,32,32>(datafile_life,input_labels,output,0.1); // int Batchsize_AD, int mini_batch_size

}
