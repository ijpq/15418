#include <algorithm>

// Generate random data
void initRandom(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
    }
}

// Generate data that gives high relative speedup
void initGood(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        values[i] = 1.0f;
    }
}

// Generate data that gives low relative speedup
void initBad(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        values[i] = 1.0f;
    }
}

