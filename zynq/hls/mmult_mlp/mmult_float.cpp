#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mmult.h"

// --------------------------------------------------------------------
// function to be accelerated in HW wrapped with AXI4-Stream interface
void mmult_hw (AXI_VAL in_stream[IS_SIZE], AXI_VAL out_stream[OS_SIZE])
{
#pragma HLS INTERFACE s_axilite port=return     bundle=CONTROL_BUS
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream

	// Assertions (to avoid out of array bound writes)
	assert(CLASSES%WIDTH_RATIO==0);
	assert(FEAT%WIDTH_RATIO==0);
	assert(FEAT%WIDTH_RATIO==0);
	assert((BATCH*CLASSES)%WIDTH_RATIO==0);

	// Union used for type conversion
	union
	{
		axi_T packet;
		struct {T f0; T f1;} val;
	} converter;

	// Hardware buffers
	T offset_buf_0[LAYER_SIZE];
	T offset_buf_1[CLASSES];
	T weight_buf_0[LAYER_SIZE][FEAT];
#pragma HLS ARRAY_PARTITION variable=weight_buf_0 block factor=4 dim=2
	T weight_buf_1[CLASSES][LAYER_SIZE];
#pragma HLS ARRAY_PARTITION variable=weight_buf_1 block factor=4 dim=2
	T in_buf[TILE_SIZE][FEAT];
#pragma HLS ARRAY_PARTITION variable=in_buf block factor=8 dim=2
	T activation_buf[TILE_SIZE][LAYER_SIZE];
	T out_buf[TILE_SIZE][CLASSES];

	// Input and output AXI stream indices
	int is_idx = 0;
	int os_idx = 0;

	// Stream in offset vector
	LOAD_OFF_0: for (int i = 0; i < LAYER_SIZE; i+=WIDTH_RATIO) {
		converter.packet = pop_stream(in_stream[is_idx++]);
		offset_buf_0[i+0] = converter.val.f0;
		offset_buf_0[i+1] = converter.val.f1;
	}

	// Stream in offset vector
	LOAD_OFF_1: for (int i = 0; i < CLASSES; i+=WIDTH_RATIO) {
		converter.packet = pop_stream(in_stream[is_idx++]);
		offset_buf_1[i+0] = converter.val.f0;
		offset_buf_1[i+1] = converter.val.f1;
	}

	// Stream in weight matrix
	LOAD_W0_1: for (int i = 0; i < LAYER_SIZE; i++) {
		LOAD_W0_2: for (int j = 0; j < FEAT; j+=WIDTH_RATIO) {
#pragma HLS PIPELINE II=1
			// Pop AXI data packet
			converter.packet = pop_stream(in_stream[is_idx++]);
			weight_buf_0[i][j+0]  = converter.val.f0;
			weight_buf_0[i][j+1]  = converter.val.f1;
		}
	}

	// Stream in weight matrix
	LOAD_W1_1: for (int i = 0; i < CLASSES; i++) {
		LOAD_W1_2: for (int j = 0; j < LAYER_SIZE; j+=WIDTH_RATIO) {
#pragma HLS PIPELINE II=1
			// Pop AXI data packet
			converter.packet = pop_stream(in_stream[is_idx++]);
			weight_buf_1[i][j+0]  = converter.val.f0;
			weight_buf_1[i][j+1]  = converter.val.f1;
		}
	}


	// Stream in input matrix
	LT: for (int t = 0; t < BATCH; t += TILE_SIZE) {
		LOAD_I_1: for (int i = 0; i < TILE_SIZE; i++) {
			LOAD_I_2: for (int j = 0; j < FEAT; j+=WIDTH_RATIO) {
	#pragma HLS PIPELINE II=1
				// Pop AXI data packet
				converter.packet = pop_stream(in_stream[is_idx++]);
				in_buf[i][j+0]  = converter.val.f0;
				in_buf[i][j+1]  = converter.val.f1;
			}
		}


		// Iterate over batch elements
		L1: for (int i = 0; i < TILE_SIZE; i++) {
			// Iterate over output classes
			L2: for (int j = 0; j < LAYER_SIZE; j++) {
	#pragma HLS PIPELINE II=1
				// Perform the dot product
				T tmp = offset_buf_0[j];
				L3: for(int k = 0; k < FEAT; k++) {
	#pragma HLS PIPELINE II=1
					T mult = in_buf[i][k] * weight_buf_0[j][k];
					tmp += mult;
				}
				activation_buf[i][j] = tmp * (tmp > 0); // relu
			}
		}

		// Iterate over batch elements
		L4: for (int i = 0; i < TILE_SIZE; i++) {
			// Iterate over output classes
			L5: for (int j = 0; j < CLASSES; j++) {
	#pragma HLS PIPELINE II=1
				// Perform the dot product
				T tmp = offset_buf_1[j];
				L6: for(int k = 0; k < LAYER_SIZE; k++) {
	#pragma HLS PIPELINE II=1
					T mult = activation_buf[i][k] * weight_buf_1[j][k];
					tmp += mult;
				}
				out_buf[i][j] = tmp / (1. + abs(tmp)); // approximated sigmoid
			}
		}

		// Stream out output matrix
		STORE_O_1: for (int i = 0; i < TILE_SIZE; i++) {
			STORE_O_2: for (int j = 0; j < CLASSES; j+=WIDTH_RATIO) {
	#pragma HLS PIPELINE II=1
				// Push output element into AXI stream
				converter.val.f0 = out_buf[i][j+0];
				converter.val.f1 = out_buf[i][j+1];
				out_stream[os_idx++] = push_stream(converter.packet, os_idx == (OS_SIZE));
			}
		}
	}
}


// --------------------------------------------------------
// functions to insert and extract elements from an axi stream
// includes conversion to correct data type
axi_T pop_stream(AXI_VAL const &e)
{
#pragma HLS INLINE

	axi_T ret = e.data;

	volatile ap_uint<sizeof(axi_T)> strb = e.strb;
	volatile ap_uint<sizeof(axi_T)> keep = e.keep;
	volatile ap_uint<AXI_U> user = e.user;
	volatile ap_uint<1> last = e.last;
	volatile ap_uint<AXI_TI> id = e.id;
	volatile ap_uint<AXI_TD> dest = e.dest;

	return ret;
}

AXI_VAL push_stream(axi_T const &v, bool last = false)
{
#pragma HLS INLINE

	AXI_VAL e;

	e.data = v;
	e.strb = (1<<sizeof(axi_T))-1;
	e.keep = (1<<sizeof(axi_T))-1;
	e.user = 0;
	e.last = last ? 1 : 0;
	e.id = 0;
	e.dest = 0;
	return e;
}

