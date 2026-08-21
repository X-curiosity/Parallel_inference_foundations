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


void merge(data_t *data, int left, 
                     int mid, int right){
                         
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temp vectors
    std::vector<data_t> L(n1), R(n2);

    // Copy data to temp vectors L[] and R[]
    for (int i = 0; i < n1; i++){
        L[i] = data[left + i];
    }
        
    for (int j = 0; j < n2; j++){
         R[j] = data[mid + 1 + j];
    }
       

    int i = 0, j = 0;
    int k = left;

    // Merge the temp vectors back 
    // into arr[left..right]
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            data[k] = L[i];
            i = i+1;
        }
        else {
            data[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], 
    // if there are any
    while (i < n1) {
        data[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], 
    // if there are any
    while (j < n2) {
        data[k] = R[j];
        j++;
        k++;
    }
}



void mergeSort(data_t *data, int left, int right){
    
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;
    mergeSort(data, left, mid);
    mergeSort(data, mid + 1, right);
    merge(data, left, mid, right);
}



void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    mergeSort(data, 0, n - 1);
}
