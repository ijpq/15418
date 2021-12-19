#include <iostream>
#include <cstdio>
#include "simple.h"
using namespace std;

extern void simple(float&);

int main() {
    float a = 0.1f;
    printf("before : %f \n", a );
    printf("after  : %.15f \n", a );
    ispc::simple(a);
    return 0;
}
