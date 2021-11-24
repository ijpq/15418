#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// implementation of absolute value using 15418 instrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// Set mask according to predicate
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	float result = 1.f;
	int y = exponents[i];
	float xpower = x;
	while (y > 0) {
	    if (y & 0x1)
		result *= xpower;
	    xpower = xpower * xpower;
	    y >>= 1;
	}
	if (result > 4.18f) {
	    result = 4.18f;
	}
	output[i] = result;
    }
}

__cmu418_mask _my_int_to_mask(__cmu418_vec_int &src, __cmu418_mask mask) {
    __cmu418_mask ret_mask = _cmu418_init_ones(0);
    for (int i =0 ; i < VECTOR_WIDTH; i++) {
        ret_mask.value[i] = mask.value[i] ? src.value[i]: ret_mask.value[i];
    }
    return ret_mask;
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
    //  ...
    //
    __cmu418_vec_float x, xpower, result; // define x, xpower, result, \
    initialization at below even if it is not recommended.
    __cmu418_vec_int zero = _cmu418_vset_int(0);
    __cmu418_vec_int y; // define y
    __cmu418_mask maskAll = _cmu418_init_ones();
    __cmu418_vec_int Allones  = _cmu418_vset_int(1);
    __cmu418_mask mask_true = _cmu418_init_ones(0);
    __cmu418_vec_int int_lowest_bit = _cmu418_vset_int(0);
    __cmu418_mask mask_lowest_bit;
    __cmu418_vec_float thr = _cmu418_vset_float(4.18f);
    __cmu418_mask remainder_mask;
    if (N % VECTOR_WIDTH) {
        remainder_mask = _cmu418_init_ones(N % VECTOR_WIDTH);
    }
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // get a mask when N cannot be divided by vec_width
        if (i+VECTOR_WIDTH > N) {
            maskAll = _cmu418_mask_and(maskAll, remainder_mask);   
        }
        _cmu418_vload_float(x, values+i, maskAll); // x = values[i]
        result = _cmu418_vset_float(1.f); // result = 1.f
        _cmu418_vload_int(y, exponents+i, maskAll); // y = exponents[i]
        _cmu418_vmove_float(xpower, x, maskAll); // xpower = x
        while (1) { 
            // while (y > 0)
            _cmu418_vgt_int(mask_true, y, zero, maskAll);
            if(!_cmu418_cntbits(mask_true)) break;

            _cmu418_vbitand_int(int_lowest_bit, y, Allones, mask_true); // y & 0x01
            mask_lowest_bit = _my_int_to_mask(int_lowest_bit, mask_true);
            _cmu418_vmult_float(result, result, xpower, mask_lowest_bit); // result *= xpower
            _cmu418_vmult_float(xpower, xpower, xpower, mask_true); // xpower = xpower * xpower
            _cmu418_vshiftright_int(y, y, Allones, mask_true); // y >>= 1;
        }
        _cmu418_vgt_float(mask_true, result, thr, maskAll); 
        _cmu418_vset_float(result, 4.18f, mask_true);
        _cmu418_vstore_float(output+i, result, maskAll);
    }
    return ;
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
	sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
    //  ...
}
