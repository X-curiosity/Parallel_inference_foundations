/*

INSTRUCTIONS : General instructions for this exercise
In this task, you will implement parallel sorting algorithms that outperform the single-threaded std::sort.

Interface
We have already defined the following data type 
that represents 64-bit unsigned integers:

typedef unsigned long long data_t;
You need to implement the following function:

void psort(int n, data_t* data)
Here n is the size of the input array data. 
All input elements are of type data_t, i.e., 64-bit unsigned integers.


*/


#include <algorithm>
#include <vector>

typedef unsigned long long data_t;

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.

    std::vector<data_t> cpy_data(n);

    for(int i =0; i < n ; i++){

        cpy_data[i] = data[i];


    }






    std::sort(data, data + n);
}
