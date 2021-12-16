#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace std;


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[],
                vector<double> &iter_times)
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;
        float convergence = 0.f;

        // float error = fabs(guess * guess * x - convergence);
        float error = fabs(fabs(guess * guess - x) - convergence);

        while (error > kThreshold) {
            iter_times[i]++;
            // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            // guess = (double)(1 + x * guess * guess) / (double)(2 * guess * x);
            guess = (x + guess * guess) / (2.f * guess);
            // error = fabs(guess * guess * x - convergence);
            error = fabs(fabs(guess * guess - x) - convergence);
        }

        // output[i] = x * guess;
        output[i] = guess;
    }
}

