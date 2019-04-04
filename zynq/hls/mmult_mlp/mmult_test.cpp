#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "mmult.h"

void matrix_multiply_ref(T offsets_0[LAYER_SIZE], T offsets_1[CLASSES], T weights_0[LAYER_SIZE][FEAT],
		T weights_1[CLASSES][LAYER_SIZE],  T in[BATCH][FEAT], T out[BATCH][CLASSES])
{
	T a0[BATCH][LAYER_SIZE];
	for (int i = 0; i < BATCH; ++i) {
		for (int j = 0; j < LAYER_SIZE; ++j) {
			T sum = offsets_0[j];
			for (int k = 0; k < FEAT; ++k) {
				sum += in[i][k] * weights_0[j][k];
			}
			a0[i][j] = sum * (sum > 0); // relu activate
		}
	}
	// matrix multiplication of a A*B matrix
	for (int i = 0; i < BATCH; ++i) {
		for (int j = 0; j < CLASSES; ++j) {
			T sum = offsets_1[j];
			for (int k = 0; k < LAYER_SIZE; ++k) {
				sum += a0[i][k] * weights_1[j][k];
			}
			out[i][j] = 1./(1. + exp(-sum)); // sigmoid
		}
	}
	return;
}


int main(void)
{
	int i,j,err;

	union
	{
		axi_T packet;
		struct {T f0; T f1;} val;
	} converter;

	T offsets_0[LAYER_SIZE];
	T offsets_1[CLASSES];
	T weights_0[LAYER_SIZE][FEAT];
	T weights_1[CLASSES][LAYER_SIZE];
	T inputs[BATCH][FEAT];
	T matMult_sw[BATCH][CLASSES];
	T matMult_hw[BATCH][CLASSES];

	/** Matrix Initiation */
	for(i = 0; i<CLASSES; i++) {
		offsets_1[i] = (T) (i);
	}
	for(i = 0; i<LAYER_SIZE; i++) {
		offsets_0[i] = (T) (i);
	}

	for(i = 0; i<LAYER_SIZE; i++) {
		for(j = 0; j<FEAT; j++) {
			weights_0[i][j] = (T) (i*j);
		}
	}

	for(i = 0; i<CLASSES; i++) {
		for(j = 0; j<LAYER_SIZE; j++) {
			weights_1[i][j] = (T) (i*j);
		}
	}

	for(i = 0; i<BATCH; i++) {
		for(j = 0; j<FEAT; j++) {
			inputs[i][j] = (T) (i+j);
		}
	}
	/** End of Initiation */


	printf("DEBUGGING AXI4 STREAMING DATA TYPES!\r\n");

	// prepare data for the DUT
	AXI_VAL in_stream[IS_SIZE];
	AXI_VAL out_stream[OS_SIZE];

	// input and output stream indices
	int is_idx = 0;
	int os_idx = 0;

	// stream in the offset vector
	for(int i=0; i<LAYER_SIZE; i+=WIDTH_RATIO) {
		converter.val.f0 = offsets_0[i+0];
		converter.val.f1 = offsets_0[i+1];
		in_stream[is_idx++] = push_stream(converter.packet, 0);
	}
	// stream in the offset vector
	for(int i=0; i<CLASSES; i+=WIDTH_RATIO) {
		converter.val.f0 = offsets_1[i+0];
		converter.val.f1 = offsets_1[i+1];
		in_stream[is_idx++] = push_stream(converter.packet, 0);
	}

	// stream in the weight_0 matrix
	for(int i=0; i<LAYER_SIZE; i++) {
		for(int j=0; j<FEAT; j+=WIDTH_RATIO) {
			converter.val.f0 = weights_0[i][j+0];
			converter.val.f1 = weights_0[i][j+1];
			in_stream[is_idx++] = push_stream(converter.packet, 0);
		}
	}

	// stream in the weight_1 matrix
	for(int i=0; i<CLASSES; i++) {
		for(int j=0; j<LAYER_SIZE; j+=WIDTH_RATIO) {
			converter.val.f0 = weights_1[i][j+0];
			converter.val.f1 = weights_1[i][j+1];
			in_stream[is_idx++] = push_stream(converter.packet, 0);
		}
	}

	// stream in the input matrix
	for(int i=0; i<BATCH; i++) {
		for(int j=0; j<FEAT; j+=WIDTH_RATIO) {
			converter.val.f0 = inputs[i][j+0];
			converter.val.f1 = inputs[i][j+1];
			in_stream[is_idx++] = push_stream(converter.packet, is_idx==(IS_SIZE));
		}
	}

	//call the DUT
	mmult_hw(in_stream, out_stream);

	// extract the output matrix from the out stream
	for(int i=0; i<BATCH; i++) {
		for(int j=0; j<CLASSES; j+=WIDTH_RATIO) {
			converter.packet = pop_stream(out_stream[os_idx++]);
			matMult_hw[i][j+0] = converter.val.f0;
			matMult_hw[i][j+1] = converter.val.f1;
		}
	}

	/* reference Matrix Multiplication */
	matrix_multiply_ref(offsets_0, offsets_1, weights_0, weights_1, inputs, matMult_sw);

	/** Matrix comparison */
	err = 0;
	for (i = 0; i<BATCH; i++) {
		for (j = 0; j<CLASSES; j++) {
			if (matMult_sw[i][j] != matMult_hw[i][j]) {
				err++;
				std::cout << i << "," << j << ": expected " << matMult_sw[i][j] << " but got " << matMult_hw[i][j] << std::endl;
			}
		}
	}

	if (err == 0)
		printf("Matrices identical ... Test successful!\r\n");
	else
		printf("Test failed!\r\n");

	return err;

}
