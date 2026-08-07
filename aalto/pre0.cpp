// Instructions : You need to write a function that takes as input a bitmap image and 
// the coordinates of a rectangle, and it has to calculate the average color 
// of all pixels inside the rectangle.

//Here data is a color image with ny*nx pixels, and each pixel consists of three color components, 
//red, green, and blue. In total, there are ny*nx*3 floating-point numbers in the array data.


// The color components are numbered 0 <= c < 3, x coordinates are numbered 0 <= x < nx, y coordinates are numbered 0 <= y < ny, 
// and the value of this color component is stored in data[c + 3 * x + 3 * nx * y].
// The parameters y0, x0, y1, and x1 indicate the location of the rectangle. The upper left corner of the rectangle is at coordinates (x0, y0), and the lower right corner is at coordinates (x1-1, y1-1). 
// That is, the width of the rectangle is x1-x0 pixels and the height is y1-y0 pixels. 
// The coordinates satisfy 0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx.
// In the result that you return, avg[c] has to contain the arithmetic mean of the color component c for all pixels inside the rectangle.


#include <iostream>




struct Result {
    float avg[3];
};


Result calculate(int ny, int nx, const float *data,
                 int y0, int x0, int y1, int x1)
{

    Result average = {{0.0f, 0.0f, 0.0f}};
    float nb_pixels = (x1-x0)*(y1-y0);

    double r1 = 0; 
    double r2 = 0;
    double r3 = 0;

    for (int y = y0; y < y1; y++){
        for (int x = x0; x < x1; x++){ 
            r1 += double(data[0 + 3*x + 3*nx*y]);
            r2 += double(data[1 + 3*x + 3*nx*y]);
            r3 += double(data[2 + 3*x + 3*nx*y]);

        }

    }
    
    average.avg[0] = float(r1/nb_pixels);
    average.avg[1] = float(r2/nb_pixels);
    average.avg[2] = float(r3/nb_pixels);


    return average;

}


// Testing values 

const int nx = 4;
const int ny = 3;

const float image[ny * nx * 3] = {
    // y = 0
     10,  20,  30,    // (0,0)
     40,  50,  60,    // (1,0)
     70,  80,  90,    // (2,0)
    100, 110, 120,    // (3,0)

    // y = 1
    130, 140, 150,    // (0,1)
    160, 170, 180,    // (1,1)
    190, 200, 210,    // (2,1)
    220, 230, 240,    // (3,1)

    // y = 2
    250, 260, 270,    // (0,2)
    280, 290, 300,    // (1,2)
    310, 320, 330,    // (2,2)
    340, 350, 360     // (3,2)
};

int y0 = 0;
int x0 = 1;
int y1 = 2;
int x1 = 3;


// ************************************************



int main(){

    // Test 1: single pixel (0,0)
    Result r1 = calculate(ny, nx, image,
                      0, 0, 1, 1);
    // Expected: 10 20 30
    
    // Test 2: entire image
    Result r2 = calculate(ny, nx, image,
                      0, 0, 3, 4);
    // Expected: 175 185 195
    
    // Test 3: bottom two rows
    Result r3 = calculate(ny, nx, image,
                      1, 0, 3, 4);
    // Expected: 235 245 255
    
    std::cout << r1.avg[0] << " "
         << r1.avg[1] << " "
         << r1.avg[2] << std::endl;


    std::cout << r2.avg[0] << " "
        << r2.avg[1] << " "
        << r2.avg[2] << std::endl;


    std::cout << r3.avg[0] << " "
    << r3.avg[1] << " "
    << r3.avg[2] << std::endl;



    return 0;
}