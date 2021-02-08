#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <getopt.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

typedef enum { DATA_RANDOM, DATA_GOOD, DATA_BAD } data_t;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

extern void initRandom(float *values, int N);
extern void initGood(float *values, int N);
extern void initBad(float *values, int N);

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -d  --data r|g|b   Initialize with random|good|bad data\n");
    printf("  -?  --help         This message\n");
}


static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

int main(int argc, char *argv[]) {

    const int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    data_t dmode = DATA_RANDOM;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"data", 1, 0, 'd'},
        {"help",  0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "d:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'd':
	    if (strlen(optarg) != 1) {
		fprintf(stderr, "Invalid data option '%s'\n", optarg);
		return 1;
	    }
	    switch (optarg[0]) {
	    case 'r':
		dmode = DATA_RANDOM;
		break;
	    case 'g':
		dmode = DATA_GOOD;
		break;
	    case 'b':
		dmode = DATA_BAD;
		break;
	    default:
		fprintf(stderr, "Invalid data option '%s'\n", optarg);
		return 1;
	    }
	    break;
        case '?':
        default:
            usage(argv[0]);
            return 0;
        }
    }
    switch (dmode) {
    case DATA_RANDOM:
	initRandom(values, N);
	break;
    case DATA_GOOD:
	initGood(values, N);
	break;
    case DATA_BAD:
	initBad(values, N);
	break;
    default:
	fprintf(stderr, "Error.  Unknown data mode %d\n", dmode);
	return 1;
    }

    // generate a gold version to check results
    for (int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);

    delete[] values;
    delete[] output;
    delete[] gold;

    return 0;
}
