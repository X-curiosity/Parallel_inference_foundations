/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
/* GRADING :
In this task your submission will be graded using benchmarks/4: the input contains 9000 × 9000 pixels, and the output should contain 9000 × 9000 pixels.
The point thresholds are as follows. If you submit your solution no later than on Sunday, 10 May 2026, at 23:59:59 (Helsinki), your score will be:
Running time	Points
≤ 15.000 s	      1
≤ 9.000 s	      2
≤ 6.000 s	      3
≤ 4.500 s	      4
≤ 3.000 s	      5
If you submit your solution after the deadline, but before the course ends on Sunday, 31 May 2026, at 23:59:59 (Helsinki), your score will be:
Running time	Points
≤ 8.000 s	      1
≤ 5.000 s	      2
≤ 3.000 s	      3
*/
/* INSTRUCTIONS :
Using all resources that you have in the CPU, solve the task as fast as possible.
You are encouraged to exploit instruction-level parallelism, multithreading, and vector instructions whenever possible,
and also to optimize the memory access pattern. Please do all arithmetic with double-precision floating-point numbers.
*/
#include <cmath>
#include <vector>
#include <omp.h>


void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(ny * nx);
    int width;

    //Checking for the width
    if(nx%8==0){width=8;}
    else if(nx%4==0){width=4;}
    else if(nx%2==0){width=2;}
    else{width=1;}

    


    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;

        for (int x = 0; x < nx; x++) {
            sum += static_cast<double>(data[x + y * nx]);
        }

        double mean = sum / nx;
        double squared_sum = 0.0;
        std::vector<double>centered(width);

        if (width==8){
            
            for (int x = 0; x < nx; x = x+8){
                centered[0] = data[x + y * nx] - mean;              
                centered[1] = data[x+1+ y * nx] - mean;
                centered[2] = data[x+2+ y * nx] - mean;
                centered[3] = data[x+3+ y * nx] - mean;
                centered[4] = data[x+4+ y * nx] - mean;
                centered[5] = data[x+5+ y * nx] - mean;
                centered[6] = data[x+6+ y * nx] - mean;
                centered[7] = data[x+7+ y * nx] - mean;

                normalized[x + y * nx] = centered[0];
                normalized[x+1 + y * nx] = centered[1];
                normalized[x+2 + y * nx] = centered[2];
                normalized[x+3 + y * nx] = centered[3];
                normalized[x+4 + y * nx] = centered[4];
                normalized[x+5 + y * nx] = centered[5];
                normalized[x+6 + y * nx] = centered[6];
                normalized[x+7 + y * nx] = centered[7];
                
                
                squared_sum += (((centered[0] * centered[0]) + (centered[1] * centered[1]) +
                (centered[2] * centered[2]) + (centered[3] * centered[3]) + (centered[4] * centered[4]))
                +(centered[5] * centered[5]) + (centered[6] * centered[6]) + (centered[7] * centered[7]));
            }
        }

        else if (width==4){
            
            for (int x = 0; x < nx; x = x+4){
                centered[0] = data[x + y * nx] - mean;              
                centered[1] = data[x+1+ y * nx] - mean;
                centered[2] = data[x+2+ y * nx] - mean;
                centered[3] = data[x+3+ y * nx] - mean;
                



                normalized[x + y * nx] = centered[0];
                normalized[x+1 + y * nx] = centered[1];
                normalized[x+2 + y * nx] = centered[2];
                normalized[x+3 + y * nx] = centered[3];
                


                squared_sum += ((centered[0] * centered[0]) + (centered[1] * centered[1]) +
                (centered[2] * centered[2]) + (centered[3] * centered[3]));

            }


        }





        else if (width==2){
            
            for (int x = 0; x < nx; x = x+2) {
                

                centered[0] = data[x + y * nx] - mean;
                centered[1] = data[x+1+ y * nx] - mean;

                
                normalized[x + y * nx] = centered[0];
                normalized[x+1 + y * nx] = centered[1];
                
                squared_sum += ((centered[0] * centered[0]) + (centered[1] * centered[1])) ;

            }


        }

        else if (width==1){
            
            for (int x = 0; x < nx; x++) {
                
            centered[0] =static_cast<double>(data[x + y * nx]) - mean;
                
            normalized[x + y * nx] = centered[0];
            squared_sum += centered[0] * centered[0];

            }


        }

        

        double scale = 1.0 / std::sqrt(squared_sum);


         
        for (int x = 0; x < nx; x++) {
            normalized[x + y * nx] *= scale;
        }
    }
    


    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double correlation = 0.0;


            if(width==8){
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;
                double c3 = 0.0;
                double c4 = 0.0;
                double c5 = 0.0;
                double c6 = 0.0;
                double c7 = 0.0;
                


                //#pragma omp parallel for reduction(+:c0, c1, c2, c3,c4,c5,c6)
                for (int x = 0; x < nx; x += 8) {

                    c0 += normalized[x + i * nx] * normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                    c2 += normalized[x+2 + i * nx] *normalized[x+2 + j * nx];
                    c3 += normalized[x+3 + i * nx] *normalized[x+3 + j * nx];
                    c4 += normalized[x+4 + i * nx] *normalized[x+4 + j * nx];
                    c5 += normalized[x+5 + i * nx] *normalized[x+5 + j * nx];
                    c6 += normalized[x+6 + i * nx] *normalized[x+6 + j * nx];
                    c7 += normalized[x+7 + i * nx] *normalized[x+7 + j * nx];

                }
                
                correlation = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;

            }


            
            else if(width==4){
                double c0 = 0.0;
                double c1 = 0.0;
                double c2 = 0.0;
                double c3 = 0.0;
                


                //#pragma omp parallel for reduction(+:c0, c1, c2, c3,c4)
                for (int x = 0; x < nx; x += 4) {


                    c0 += normalized[x + i * nx] * normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                    c2 += normalized[x+2 + i * nx] *normalized[x+2 + j * nx];
                    c3 += normalized[x+3 + i * nx] *normalized[x+3 + j * nx];
        
                }
                
                correlation = c0 + c1 + c2 + c3;

            }

            


            else if(width==2){
                double c0 = 0.0;
                double c1 = 0.0;

                //#pragma omp parallel for reduction(+:c0, c1)
                for (int x = 0; x < nx; x += 2) {
                    c0 += normalized[x + i * nx] *normalized[x + j * nx];
                    c1 += normalized[x+1 + i * nx] *normalized[x+1 + j * nx];
                }

                correlation = c0 + c1;

            }

            else if(width==1) {
                double c0 = 0.0;
                //#pragma omp parallel for reduction(+:c0)
                for (int x = 0; x < nx; x++) {
                    c0 += normalized[x + i * nx] *normalized[x + j * nx];
                }

                correlation = c0 ;


            }

        result[i + j * ny] = static_cast<float>(correlation); 
   
        }
    }
}