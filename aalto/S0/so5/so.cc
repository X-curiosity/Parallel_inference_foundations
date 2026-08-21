#include <algorithm>
#include <vector>

typedef unsigned long long data_t;


void quicksort(data_t *data, int left, int right) {
    if (right - left <= 1) {
        return;
    }

    data_t a = data[left];
    data_t b = data[left + (right - left) / 2];
    data_t c = data[right - 1];
    
    data_t pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));

    int i = left;
    int j = right - 1;

    while (i <= j) {


        while (data[i] < pivot) {
            i++;
        }


        while (data[j] > pivot) {

            j--;
        }

        if (i <= j) {
            std::swap(data[i], data[j]);
            i++;
            j--;
        }
    }

    quicksort(data, left, j + 1);
    quicksort(data, i, right);
}

void psort(int n, data_t *data) {
    quicksort(data, 0, n);
}



