/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <omp.h>
#include <vector>

void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> normalized(ny * nx);
    std::vector<double> normalized_T(nx * ny);
    int cycles;  
    int width;
    bool strassen = false;
    int number = (ny*ny)-(ny*(ny-1)/2); 




    // Checking for the width
    if (nx % 4 == 0) {
        width = 4;
        
    } else if (nx % 2 == 0) {
        width = 2;  
    } else {
        width = 1; 
    }

    // Checking the number of correlation to compute 

    if(number%8==0){cycles = 8;}
    else if(number%4==0){cycles = 4;}
    else if(number%2==0){cycles = 2;}
    else{cycles = 1;}



    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,4)
        for (int y = 0; y < ny; y++) {
            double sum = 0.0;

            for (int x = 0; x < nx; x++) {
                sum += static_cast<double>(data[x + y * nx]);
            }

            double mean = sum / nx;
            double squared_sum = 0.0;
 

            if (width == 4) {
                double c01 = 0;
                double c02 = 0;

                double c11 = 0;
                double c12 = 0;

                double c21 = 0;
                double c22 = 0;

                double c31 = 0;
                double c32 = 0;

                for (int x = 0; x < nx; x = x + 4) {
                    c01 = data[x + y * nx] * (data[x + y * nx] - mean);
                    c11 = data[x + 1 + y * nx] * (data[x + 1 + y * nx] - mean);
                    c21 = data[x + 2 + y * nx] * (data[x + 2 + y * nx] - mean);
                    c31 = data[x + 3 + y * nx] * (data[x + 3 + y * nx] - mean);

                    c02 = mean * (data[x + y * nx] - mean);
                    c12 = mean * (data[x + 1 + y * nx] - mean);
                    c22 = mean * (data[x + 2 + y * nx] - mean);
                    c32 = mean * (data[x + 3 + y * nx] - mean);

                    normalized[x + y * nx] = data[x + y * nx] - mean;
                    normalized[x + 1 + y * nx] = data[x + 1 + y * nx] - mean;
                    normalized[x + 2 + y * nx] = data[x + 2 + y * nx] - mean;
                    normalized[x + 3 + y * nx] = data[x + 3 + y * nx] - mean;

                    squared_sum +=
                        ((c01 - c02) + (c11 - c12) +
                        (c21 - c22) + (c31 - c32));
                }
            } else if (width == 2) {

                double c01 = 0;
                double c02 = 0;

                double c11 = 0;
                double c12 = 0;

    
                for (int x = 0; x < nx; x = x + 2) {
                    c01 = data[x + y * nx] * (data[x + y * nx] - mean);
                    c11 = data[x + 1 + y * nx] * (data[x + 1 + y * nx] - mean);

                    c02 = mean * (data[x + y * nx] - mean);
                    c12 = mean * (data[x + 1 + y * nx] - mean);

                    normalized[x + y * nx] = data[x + y * nx] - mean;
                    normalized[x + 1 + y * nx] = data[x + 1 + y * nx] - mean;

                    squared_sum +=((c01 - c02) + (c11 - c12));
                }
            } else if (width == 1) {
                double c01 = 0;
                double c02 = 0;

                for (int x = 0; x < nx; x++) {
                    c01 = data[x + y * nx] * (data[x + y * nx] - mean);
                    c02 = mean * (data[x + y * nx] - mean);

                    normalized[x + y * nx] = data[x + y * nx] - mean;
                    squared_sum += c01 - c02;
                }
            }

            double scale = 1.0 / std::sqrt(squared_sum);

            if (width == 4) {
                for (int x = 0; x < nx; x = x + 4) {
                    normalized[x + y * nx] *= scale;
                    normalized[x + 1 + y * nx] *= scale;
                    normalized[x + 2 + y * nx] *= scale;
                    normalized[x + 3 + y * nx] *= scale;
                }
            } else if (width == 2) {
                for (int x = 0; x < nx; x = x + 2) {
                    normalized[x + y * nx] *= scale;
                    normalized[x + 1 + y * nx] *= scale;
                }
            } else if (width == 1) {
                for (int x = 0; x < nx; x = x + 1) {
                    normalized[x + y * nx] *= scale;
                }
            }
        }


        // Let's construct the Transpose :
        
        #pragma omp for schedule(static)
        for(int y = 0; y<ny;y++){
            for(int x = 0; x<nx;x++){
                normalized_T[y + x * ny] = normalized[x + y * nx];
            }
        }

        
        if (cycles == 8 && strassen == false) {
            #pragma omp for schedule(dynamic,4)
            for (int j = 0; j < ny; j++) {
                int i = j;
                for (; i + 7 < ny; i += 8) {
                    double correlation1 = 0.0;
                    double correlation2 = 0.0;
                    double correlation3 = 0.0;
                    double correlation4 = 0.0;
                    double correlation5 = 0.0;
                    double correlation6 = 0.0;
                    double correlation7 = 0.0;
                    double correlation8 = 0.0;

                    for(int x = 0; x < nx; x++){
                        double value_j = normalized_T[j + x * ny];
                        correlation1 += normalized[x + i * nx] * value_j;
                        correlation2 += normalized[x + (i + 1) * nx] * value_j;
                        correlation3 += normalized[x + (i + 2) * nx] * value_j;
                        correlation4 += normalized[x + (i + 3) * nx] * value_j;
                        correlation5 += normalized[x + (i + 4) * nx] * value_j;
                        correlation6 += normalized[x + (i + 5) * nx] * value_j;
                        correlation7 += normalized[x + (i + 6) * nx] * value_j;
                        correlation8 += normalized[x + (i + 7) * nx] * value_j;
                    }

                    result[i + j * ny] = correlation1;
                    result[i + 1 + j * ny] = correlation2;
                    result[i + 2 + j * ny] = correlation3;
                    result[i + 3 + j * ny] = correlation4;
                    result[i + 4 + j * ny] = correlation5;
                    result[i + 5 + j * ny] = correlation6;
                    result[i + 6 + j * ny] = correlation7;
                    result[i + 7 + j * ny] = correlation8;
                }

                for (; i < ny; i++) {
                    double correlation = 0.0;
                    for(int x = 0; x < nx; x++){
                        correlation += normalized[x + i * nx] * normalized_T[j + x * ny];
                    }
                    result[i + j * ny] = correlation;
                }
            }
        } else if (cycles == 4 && strassen == false) {
            #pragma omp for schedule(dynamic,4)
            for (int j = 0; j < ny; j++) {
                int i = j;
                for (; i + 3 < ny; i += 4) {
                    double correlation1 = 0.0;
                    double correlation2 = 0.0;
                    double correlation3 = 0.0;
                    double correlation4 = 0.0;

                    for(int x = 0; x < nx; x++){
                        double value_j = normalized_T[j + x * ny];
                        correlation1 += normalized[x + i * nx] * value_j;
                        correlation2 += normalized[x + (i + 1) * nx] * value_j;
                        correlation3 += normalized[x + (i + 2) * nx] * value_j;
                        correlation4 += normalized[x + (i + 3) * nx] * value_j;
                    }

                    result[i + j * ny] = correlation1;
                    result[i + 1 + j * ny] = correlation2;
                    result[i + 2 + j * ny] = correlation3;
                    result[i + 3 + j * ny] = correlation4;
                }

                for (; i < ny; i++) {
                    double correlation = 0.0;
                    for(int x = 0; x < nx; x++){
                        correlation += normalized[x + i * nx] * normalized_T[j + x * ny];
                    }
                    result[i + j * ny] = correlation;
                }
            }
        } else if (cycles == 2 && strassen == false) {
            #pragma omp for schedule(dynamic,2)
            for (int j = 0; j < ny; j++) {
                int i = j;
                for (; i + 1 < ny; i += 2) {
                    double correlation1 = 0.0;
                    double correlation2 = 0.0;

                    for(int x = 0; x < nx; x++){
                        double value_j = normalized_T[j + x * ny];
                        correlation1 += normalized[x + i * nx] * value_j;
                        correlation2 += normalized[x + (i + 1) * nx] * value_j;
                    }

                    result[i + j * ny] = correlation1;
                    result[i + 1 + j * ny] = correlation2;
                }

                for (; i < ny; i++) {
                    double correlation = 0.0;
                    for(int x = 0; x < nx; x++){
                        correlation += normalized[x + i * nx] * normalized_T[j + x * ny];
                    }
                    result[i + j * ny] = correlation;
                }
            }
        } else if (cycles == 1 && strassen == false) {
            #pragma omp for schedule(dynamic,2)
            for (int j = 0; j < ny; j++) {
                for (int i = j; i < ny; i++) {
                    double correlation = 0.0;

                    for(int x = 0; x < nx; x++){
                        correlation += normalized[x + i * nx] * normalized_T[j + x * ny];
                    }

                    result[i + j * ny] = correlation;
                }
            }
        }
    }
}
