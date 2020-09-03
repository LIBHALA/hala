#include <iostream>
#include "hala.hpp"

using std::cout;
using std::endl;

int main(void){

    std::vector<float> x = {1.0, 2.0, 3.0};
    float *ax = x.data();
    ax[0] = 0.0f;

    // make sure that arrays cannot be used in overloads
    // that infer vector sizes from the size of the vector
    #ifdef TEST_ARRAY_SIZE_BAD
    hala::scal(2.0, &ax[1]);
    #endif

    #ifdef TEST_ARRAY_SIZE_OK
    hala::scal(2.0, x);
    #endif

    std::vector<double> y = {3.0, 4.0, 5.0};
    y[0] = 4.0;

    // make sure that types cannot be mixed
    // float-double cannot be used together
    #ifdef TEST_TYPES_BAD
    hala::axpy(2.0, x, y);
    #endif

    #ifdef TEST_TYPES_OK
    std::vector<double> yy(3, 1.0);
    hala::axpy(-1.0, y, yy);
    #endif

    // const test correctness test
    std::vector<std::complex<float>> const z = {{1.0, 0.0}, {2.0, 0.5}};
    std::complex<float> const *az = z.data();
    auto wz = hala::wrap_array(az, 2);

    #ifdef TEST_CONST_BAD
    hala::scal(2.0, wz);
    #endif

    #ifdef TEST_CONST_OK
    hala::norm(wz);
    #endif

    return 0;
}
