__kernel void linear_layer(
	__global const float* input,
	__global float* output,
	__global const float* weights,
	__global const float* bias,
	const int token_size,
	const int input_size,
	const int output_size) {
	
	int token_index = get_global_id(0);
	int index = get_global_id(1);

	if (token_index >= token_size || index >= output_size) {
		return;
	}

	float sum = bias[index];

    for (int i = 0; i < input_size; i++) {
        sum += input[token_index * input_size + i] * weights[index * input_size + i];
    }

    output[token_index * output_size + index] = sum;
}