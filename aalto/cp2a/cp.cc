/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/


/*

GRADING :

In this task your submission will be graded using benchmarks/2: the input contains 4000 × 1000 pixels, and the output should contain 4000 × 4000 pixels.

The point thresholds are as follows. If you submit your solution no later than on Sunday, 3 May 2026, at 23:59:59 (Helsinki), your score will be:

Running time	Points
≤ 7.000 s	1
≤ 6.000 s	2
≤ 5.000 s	3
If you submit your solution after the deadline, but before the course ends on Sunday, 31 May 2026, at 23:59:59 (Helsinki), your score will be:

Running time	Points
≤ 6.000 s	1
≤ 5.000 s	2

*/








#include <cmath>
#include <vector>

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(ny * nx);
    int width;

    //Checking for the width
    if(nx%4==0){width=4;}
    else if(nx%3==0){width=3;}
    else if(nx%2==0){width=2;}
    else{width=1;}

    for (int y = 0; y < ny; y++) {
        double sum = 0.0;

        for (int x = 0; x < nx; x++) {
            sum += static_cast<double>(data[x + y * nx]);
        }

        double mean = sum / nx;
        double squared_sum = 0.0;

        for (int x = 0; x < nx; x++) {
            double centered =
                static_cast<double>(data[x + y * nx]) - mean;

            normalized[x + y * nx] = centered;
            squared_sum += centered * centered;
        }

        double scale = 1.0 / std::sqrt(squared_sum);

        for (int x = 0; x < nx; x++) {
            normalized[x + y * nx] *= scale;
        }
    }
    

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double correlation = 0.0;

            if(width==4){
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;
                double c3 = 0.0;

                for (int x = 0; x < nx; x += 4) {

                    c0 += normalized[x + i * nx] *normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                    c2 += normalized[x+2 + i * nx] *normalized[x+2 + j * nx];
                    c3 += normalized[x+3 + i * nx] *normalized[x+3+ j * nx];

                }
                
                correlation = c0 + c1 + c2 + c3;

            }

            else if (width==3){
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;

                 for (int x = 0; x < nx; x += 3) {

                    c0 += normalized[x + i * nx] *normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                    c2 += normalized[x+2 + i * nx] *normalized[x+2 + j * nx];
                }

                correlation = c0 + c1 + c2;

            }


            else if (width==2){
                double c0 = 0.0;
                double c1 = 0.0;

                 for (int x = 0; x < nx; x += 2) {
                    c0 += normalized[x + i * nx] *normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                }

                correlation = c0 + c1;

            }

            else {
                double c0 = 0.0;

                 for (int x = 0; x < nx; x++) {
                    correlation += normalized[x + i * nx] *normalized[x + j * nx];
                }

            }

        result[i + j * ny] = static_cast<float>(correlation); 
   
        }
    }
}