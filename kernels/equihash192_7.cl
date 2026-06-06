

//#define PRINT 1

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant ulong blake_iv[] =
{
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

inline static uint2 ror64(const uint2 x, const uint y)
{
    return (uint2)(((x).x>>y)^((x).y<<(32-y)),((x).y>>y)^((x).x<<(32-y)));
}
inline static uint2 ror64_2(const uint2 x, const uint y)
{
    return (uint2)(((x).y>>(y-32))^((x).x<<(64-y)),((x).x>>(y-32))^((x).y<<(64-y)));
}

#define gFunc(va, vb, vc, vd, x, y) \
va = (va + vb + x); \
((uint2*)&vd)[0] = ((uint2*)&vd)[0].yx ^ ((uint2*)&va)[0].yx; \
vc = (vc + vd); \
((uint2*)&vb)[0] = ror64( ((uint2*)&vb)[0] ^ ((uint2*)&vc)[0], 24U); \
va = (va + vb + y); \
((uint2*)&vd)[0] = ror64( ((uint2*)&vd)[0] ^ ((uint2*)&va)[0], 16U); \
vc = (vc + vd); \
((uint2*)&vb)[0] = ror64_2( ((uint2*)&vb)[0] ^ ((uint2*)&vc)[0], 63U);

#define gFunc0(va, vb, vc, vd) \
va = (va + vb); \
((uint2*)&vd)[0] = ((uint2*)&vd)[0].yx ^ ((uint2*)&va)[0].yx; \
vc = (vc + vd); \
((uint2*)&vb)[0] = ror64( ((uint2*)&vb)[0] ^ ((uint2*)&vc)[0], 24U); \
va = (va + vb); \
((uint2*)&vd)[0] = ror64( ((uint2*)&vd)[0] ^ ((uint2*)&va)[0], 16U); \
vc = (vc + vd); \
((uint2*)&vb)[0] = ror64_2( ((uint2*)&vb)[0] ^ ((uint2*)&vc)[0], 63U);


inline uint swapByteEndian(uint input) {
	uint tmp0 = input & 0x0F0F0F0F;
	uint tmp1 = input & 0xF0F0F0F0;

	tmp0 = tmp0 << 4;
	tmp1 = tmp1 >> 4;

	uint tmpIn = tmp0 | tmp1;
	
	tmp0 = tmpIn & 0x33333333;
	tmp1 = tmpIn & 0xCCCCCCCC;

	tmp0 = tmp0 << 2;
	tmp1 = tmp1 >> 2;

	tmpIn = tmp0 | tmp1;

	tmp0 = tmpIn & 0x55555555;
	tmp1 = tmpIn & 0xAAAAAAAA;

	tmp0 = tmp0 << 1;
	tmp1 = tmp1 >> 1;

	return tmp0 | tmp1;
}


uint8 shr_7(uint8 input, uint sh0, uint sh1) {
	uint8 tmp = (input >> sh0);
	uint8 tmp2 = (input << 32-sh0);

	tmp.s0 = input.s0 >> sh1;

	tmp.s0123 |= tmp2.s1234;
	tmp.s45 |= tmp2.s56;

	tmp.s7 = input.s7;
	return tmp;
}


void round0(ulong8 blake_state, __global uint8 *resultsHi, __global uint2 *resultsLo , __global uint *counters, uint tId, uint gId) {
       	ulong v[16];
   
	ulong word1 = ((ulong)tId << 32) | gId;
	// init vector v
	v[0] = blake_state.s0;
	v[1] = blake_state.s1;
	v[2] = blake_state.s2;
	v[3] = blake_state.s3;
	v[4] = blake_state.s4;
	v[5] = blake_state.s5;
	v[6] = blake_state.s6;
	v[7] = blake_state.s7;
	v[8] =  blake_iv[0];
	v[9] =  blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4];
	v[13] = blake_iv[5];
	v[14] = blake_iv[6];
	v[15] = blake_iv[7];
	// gFunc in length of data
	v[12] ^= 144 /* length of "i" */;
	// last block
	v[14] ^= (ulong)-1;

	// round 1
	gFunc(v[0], v[4], v[8],  v[12], 0, word1);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 2
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc(v[0], v[5], v[10], v[15], word1, 0);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 3
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc(v[2], v[7], v[8],  v[13], 0, word1);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 4
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc(v[1], v[5], v[9],  v[13], 0, word1);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 5
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc(v[0], v[5], v[10], v[15], 0, word1);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 6
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc(v[3], v[4], v[9],  v[14], word1, 0);
	// round 7
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc(v[1], v[5], v[9],  v[13], word1, 0);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 8
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc(v[2], v[6], v[10], v[14], 0, word1);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 9
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc(v[2], v[7], v[8],  v[13], word1, 0);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 10
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc(v[3], v[7], v[11], v[15], word1, 0);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 11
	gFunc(v[0], v[4], v[8],  v[12], 0, word1);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc0(v[0], v[5], v[10], v[15]);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9],  v[14]);
	// round 12
	gFunc0(v[0], v[4], v[8],  v[12]);
	gFunc0(v[1], v[5], v[9],  v[13]);
	gFunc0(v[2], v[6], v[10], v[14]);
	gFunc0(v[3], v[7], v[11], v[15]);
	gFunc(v[0], v[5], v[10], v[15], word1, 0);
	gFunc0(v[1], v[6], v[11], v[12]);
	gFunc0(v[2], v[7], v[8],  v[13]);
	gFunc0(v[3], v[4], v[9], v[14]);	

        v[0] = v[0] ^ blake_state.s0 ^ v[8];
        v[1] = v[1] ^ blake_state.s1 ^ v[9];
        v[2] = v[2] ^ blake_state.s2 ^ v[10];
        v[3] = v[3] ^ blake_state.s3 ^ v[11];
        v[4] = v[4] ^ blake_state.s4 ^ v[12];
        v[5] = v[5] ^ blake_state.s5 ^ v[13];
        v[6] = v[6] ^ blake_state.s6 ^ v[14];
	v[7] = v[7] ^ blake_state.s7 ^ v[15]; 

	uint8 output0, output1, output2;
	int addme,shift, bucket;

	output0.s0 = v[0] & 0xFFFFFFFF;
	output0.s1 = v[0] >> 32; 
	output0.s2 = v[1] & 0xFFFFFFFF;  
	output0.s3 = v[1] >> 32;
	output0.s4 = v[2] & 0xFFFFFFFF; 
	output0.s5 = v[2] >> 32; 
	output0.s7 = 0;
	output0.s6 = 2*tId;

	output1.s0 = v[3] & 0xFFFFFFFF;
	output1.s1 = v[3] >> 32; 
	output1.s2 = v[4] & 0xFFFFFFFF;  
	output1.s3 = v[4] >> 32;
	output1.s4 = v[5] & 0xFFFFFFFF; 
	output1.s5 = v[5] >> 32; 
	output1.s7 = 0;
	output1.s6 = 2*tId+1;

	uint2 addr;
	addr.s0 = atomic_inc(&counters[output0.s0 & 0x1FFF]);
	addr.s0 += 4592 * (output0.s0 & 0x1FFF);
	resultsHi[addr.s0] = shr_7(output0,13,13);

	addr.s1 = atomic_inc(&counters[output1.s0 & 0x1FFF]);
	addr.s1 += 4592 * (output1.s0 & 0x1FFF);
	resultsHi[addr.s1] = shr_7(output1,13,13);
}


__kernel void clearCounter (__global uint8 * buffer0,
		__global uint8 * buffer1,
		__global uint8 * buffer2,
		__global uint4 * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {

	uint gId = get_global_id(0);
	counters[gId] = (uint4) 0;

	if (gId == 0) {
		res[0] = (uint4) 0; 
	}
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void blake (
		__global uint8 * output0,
		__global uint2 * buffer1,
		__global uint8 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	
	uint gId = get_global_id(0);
	
	round0(hashState, output0, buffer1, counters, gId, (uint) (nonce & 0xFFFFFFFF));
}


inline int masking8_7(uint8 input, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		
		if (pos < 1166) {
			uint value  = atomic_xchg(&scratch[654 + ((input.s0 >> 2) & 0x1FF)], pos); 
			uint high = value >> 12;

			value &= 0x7FF;
			value |= (input.s0 & 0xFFFFF800);	
			
			scratch[1166+pos] = value;
			scratch[2332+pos] = input.s1;
			scratch[3498+pos] = input.s2;
			scratch[4664+pos] = input.s3;
			scratch[5830+pos] = input.s4;
			scratch[6996+pos] = input.s5;
	 
			if (pos < 654) {
				scratch[pos] = input.s6 << 12;
			} else {
				atomic_or(&scratch[pos], input.s6 << 12); 
			}	

			if (high != 0) {
				atomic_or(&scratch[654 + ((input.s0 >> 2) & 0x1FF)], high << 12); 
			}				

		}

		return pos;
	}	
	
	return -1;
}



inline void masking4_4b(uint4 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		
		uint value  = atomic_xchg(&scratch[(input.s0 >> 5) & 0x1FF], pos); 
		scratch[1792+pos] = input.s1;
		scratch[3072+pos] = input.s2;
		scratch[4352+pos] = input.s3;
		scratch[5632+pos] = idx;
 
		value |= (input.s0 & 0xFFFFE000);	
		scratch[512+pos] = value;
	}
}

inline void masking4_4bt(uint4 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		if (pos < 1280) {
			uint value  = atomic_xchg(&scratch[(input.s0 >> 5) & 0x1FF], pos); 
			scratch[1792+pos] = input.s1;
			scratch[3072+pos] = input.s2;
			scratch[4352+pos] = input.s3;
			scratch[5632+pos] = idx;
	 
			value |= (input.s0 & 0xFFFFE000);	
			scratch[512+pos] = value;		
		}
	}
}

void masking4_4(uint4 input, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);

		uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x1FF], pos);  
		value |= (input.s0 & 0xFFFFF800);	
		scratch[512+pos] = value;
		scratch[1728+pos] = input.s1;
		scratch[2944+pos] = input.s2;
		scratch[4160+pos] = input.s3;

		
	}
}

void masking4_4t(uint4 input, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		if (pos < 1216) {
			uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x1FF], pos);  
			value |= (input.s0 & 0xFFFFF800);	
			scratch[512+pos] = value;
			scratch[1728+pos] = input.s1;
			scratch[2944+pos] = input.s2;
			scratch[4160+pos] = input.s3;
		}
	}
}


void masking4_3b(uint4 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);

		uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x3FF], pos);  
		value |= (input.s0 & 0xFFFFF000);	
		scratch[1024+pos] = value;
		scratch[3328+pos] = input.s1;
		scratch[5632+pos] = (input.s2 & 0x3FFF) | (idx << 14);
		//scratch[7936+pos] = idx;
	}
}


void masking4_3bt(uint4 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		if (pos < 2304) {
			uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x3FF], pos);  
			value |= (input.s0 & 0xFFFFF000);	
			scratch[1024+pos] = value;
			scratch[3328+pos] = input.s1;
			scratch[5632+pos] = (input.s2 & 0x3FFF) | (idx << 14);
			//scratch[7936+pos] = idx;
		}
	}
}


void masking2_2b(uint2 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);

		uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x3FF], pos);  
		value |= (input.s0 & 0xFFFFF000);	
		scratch[1024+pos] = value;
		scratch[3328+pos] = (input.s1 & 0x3FFFF) | (idx << 18);
	}
}

void masking2_2bt(uint2 input, uint idx, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		if (pos < 2304) {
			uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x3FF], pos);  
			value |= (input.s0 & 0xFFFFF000);	
			scratch[1024+pos] = value;
			scratch[3328+pos] = (input.s1 & 0x3FFFF) | (idx << 18);
		}
	}
}

void masking2_2(uint2 input, __local uint* scratch, __local uint* cnt, uint mask, uint check) {
	if ((input.s0 & check) == mask) {
		uint pos = atomic_inc(&cnt[0]);
		if (pos < 2304) {
			uint value  = atomic_xchg(&scratch[(input.s0 >> 2) & 0x3FF], pos);  
			value |= (input.s0 & 0xFFFFF000);	
			scratch[1024+pos] = value;
			scratch[3328+pos] = input.s1;
		}
	}
}


uint2 compress(uint in0, uint in1) {
	uint hi, lo;
	uint hi2, lo2;

	if (in0 < in1) {
		hi = in1 >> 12;
		hi2 = in1 & 0xFFF;
		lo = in0 >> 12;
		lo2 = in0 & 0xFFF;
	} else {
		hi = in0 >> 12;
		hi2 = in0 & 0xFFF;
		lo = in1 >> 12;
		lo2 = in1 & 0xFFF;
	}

	hi++;

	uint2 tmp;
	tmp.s0 = mul24(hi,(hi-1));
	tmp.s0 = tmp.s0 >> 1;
	tmp.s0 += lo;
	
	tmp.s0 = tmp.s0 << 6;
	tmp.s0 |= (hi2 & 0x3F);

	tmp.s1 = hi2 >> 6;	
	tmp.s1 |= (lo2 << 6);

	return tmp;	
}

uint compress2(uint in0, uint in1) {
	uint hi, lo;
	
	if (in0 < in1) {
		hi = in1;
		lo = in0;
	} else {
		hi = in0;
		lo = in1;
	}

	uint tmp;
	tmp = mul24(hi,(hi-1));
	tmp = tmp >> 1;
	tmp += lo;

	return tmp;	
}

uint2 decompress(uint2 in) {
	double inFl = (double) (in.s0 >> 6);
	
	inFl *= 2.0;
	inFl += 1.0;

	uint2 res; 
	res.s0 = (uint) round(sqrt(inFl));


	uint tmp = res.s0 * (res.s0-1);
	tmp = tmp >> 1;	

	res.s1 = (uint) ((in.s0 >> 6) - tmp);
	res.s0--;

	res.s0 = res.s0 << 12;
	res.s1 = res.s1 << 12;

	res.s0 |= (in.s0 & 0x3F);
	res.s0 |= ((in.s1 & 0x3F) << 6);

	res.s1 |= (in.s1 >> 6);
	
	return res;	
}

uint2 decompress2(uint in) {
	double inFl = (double) in;
	
	inFl *= 2.0;
	inFl += 1.0;

	uint2 res; 
	res.s0 = (uint) round(sqrt(inFl));


	uint tmp = res.s0 * (res.s0-1);
	tmp = tmp >> 1;	

	res.s1 = (uint) (in - tmp);
	
	return res;	
}


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round1 (				// Round 1
		__global uint8 * buffer0,
		__global uint8 * buffer1,
		__global uint8  * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {

	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = (grp & 3);

	__global uint8 * output = buffer1;
	__global uint8 * input = &buffer0[bucket*4592];

	__local uint scratch[8162];
	__local uint * ht = &scratch[654];
	__local uint * scratch0 = &scratch[1166];
	__local uint * scratch1 = &scratch[2332];
	__local uint * scratch2 = &scratch[3498];
	__local uint * scratch3 = &scratch[4664];
	__local uint * scratch4 = &scratch[5830];
	__local uint * scratch5 = &scratch[6996];
	__local uint * scratch6 = &scratch[0];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[0];
	__global uint * outCounter = &counters[16384];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R0: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif

	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 4592);
	} 

	ht[lId] = 0x7FF;
	ht[lId+256] = 0x7FF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint8 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5;

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	if ((lId + 3584) < iCNT[0]) iScr2 = input[lId + 3584]; 

	masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);	

	if ((lId + 3840) < iCNT[0]) iScr3 = input[lId + 3840];
	if ((lId + 4096) < iCNT[0]) iScr4 = input[lId + 4096];
	if ((lId + 4352) < iCNT[0]) iScr5 = input[lId + 4352];

	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	int pos = masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 3584) < iCNT[0]) masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 3840) < iCNT[0]) masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4096) < iCNT[0]) masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4352) < iCNT[0]) masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);

	barrier(CLK_LOCAL_MEM_FENCE);	
	uint inlim = min(iCNT[1], (uint) 1166);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);	

	int ownPos = atomic_dec(&iCNT[0]);
	uint own = scratch0[ownPos];
	uint othPos = own & 0x7FF;

	while ((othPos == 0x7FF) && (ownPos >= 0)) {
		ownPos = atomic_dec(&iCNT[0]);
		if (ownPos >= 0) own = scratch0[ownPos];
		othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
	}

	othPos = (ownPos < inlim) ? othPos : 0x7FF;

	uint addr, elem, val;
	uint2 el0, el1;
	uint8 oScrT;

	while (othPos < inlim) {
		elem = scratch0[othPos];
		oScrT.s0 = (own ^ elem) >> 11;
	
		if (oScrT.s0 != 0) {
			val = oScrT.s0 & 0x1FFF;
			addr = atomic_inc(&outCounter[val]);

			oScrT.s1 = scratch1[ownPos] ^ scratch1[othPos];
			oScrT.s2 = scratch2[ownPos] ^ scratch2[othPos];
			oScrT.s3 = scratch3[ownPos] ^ scratch3[othPos];
			oScrT.s4 = scratch4[ownPos] ^ scratch4[othPos];

			el0.s0 = scratch5[ownPos];
			el0.s1 = scratch6[ownPos];
			el1.s0 = scratch5[othPos];
			el1.s1 = scratch6[othPos];

			oScrT.s5 = (el0.s0 ^ el1.s0) & 0x7FFFF;

			el0.s0 = el0.s0 >> 19;
			el1.s0 = el1.s0 >> 19;
			el0.s1 = (el0.s1 >> 12) << 13;
			el1.s1 = (el1.s1 >> 12) << 13;

			el0.s1 |= el0.s0;
			el1.s1 |= el1.s0;

			oScrT.s0 = (oScrT.s0 >> 13) | (oScrT.s1 << 8);
			oScrT.s1 = (oScrT.s1 >> 24) | (oScrT.s2 << 8);
			oScrT.s2 = (oScrT.s2 >> 24) | (oScrT.s3 << 8);
			oScrT.s3 = (oScrT.s3 >> 24) | (oScrT.s4 << 8);
			oScrT.s4 = (oScrT.s4 >> 24) | (oScrT.s5 << 8);
			
			//if (get_global_id(0) == 0) printf("%d %d \n ", el0, el1);

			oScrT.s5 = el0.s1 | (el1.s1 << 25);
			oScrT.s6 = el1.s1 >> 7;
			oScrT.s7 = 0;
			addr += 4592*val;

			output[addr] = oScrT; 
		} 

		othPos = elem & 0x7FF;

		while ((othPos == 0x7FF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
		}
	} 	
} 


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round2 (				// Round 2
		__global uint8 * buffer0,
		__global uint8 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 
			
	uint bucket = grp >> 2; //0x3FFF;
	uint mask = grp & 3;

	__global uint8 * output = buffer0;
	__global uint8 * input = &buffer1[bucket*4592];

	__local uint scratch[8162];
	__local uint * ht = &scratch[654];
	__local uint * scratch0 = &scratch[1166];
	__local uint * scratch1 = &scratch[2332];
	__local uint * scratch2 = &scratch[3498];
	__local uint * scratch3 = &scratch[4664];
	__local uint * scratch4 = &scratch[5830];
	__local uint * scratch5 = &scratch[6996];
	__local uint * scratch6 = &scratch[0];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[16384];
	__global uint * outCounter = &counters[32768];

	if (lId == 0) {
		iCNT[0] = min(inCounter[bucket],(uint) 4592);
		iCNT[1] = 0;
	} 

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R1: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif

	ht[lId] = 0x7FF;
	ht[lId+256] = 0x7FF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint8 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5;

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	if ((lId + 3584) < iCNT[0]) iScr2 = input[lId + 3584]; 

	masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 3840) < iCNT[0]) iScr3 = input[lId + 3840];
	if ((lId + 4096) < iCNT[0]) iScr4 = input[lId + 4096];
	if ((lId + 4352) < iCNT[0]) iScr5 = input[lId + 4352];

	masking8_7(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking8_7(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 3584) < iCNT[0]) masking8_7(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 3840) < iCNT[0]) masking8_7(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4096) < iCNT[0]) masking8_7(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4352) < iCNT[0]) masking8_7(iScr5, &scratch[0], &iCNT[1], mask, 0x3);

	barrier(CLK_LOCAL_MEM_FENCE);

	barrier(CLK_LOCAL_MEM_FENCE);	
	uint inlim = min(iCNT[1], (uint) 1166);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);	

	int ownPos = atomic_dec(&iCNT[0]);
	uint own = scratch0[ownPos];
	uint othPos = own & 0x7FF;

	while ((othPos == 0x7FF) && (ownPos >= 0)) {
		ownPos = atomic_dec(&iCNT[0]);
		if (ownPos >= 0) own = scratch0[ownPos];
		othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
	}

	othPos = (ownPos < inlim) ? othPos : 0x7FF;

	uint addr, elem, val;
	uint el0;
	uint8 oScrT;

	while (othPos < inlim) {
		elem = scratch0[othPos];
		oScrT.s0 = (own ^ elem) >> 11;
	
		if (oScrT.s0 != 0) {
			val = oScrT.s0 & 0x1FFF;
			addr = atomic_inc(&outCounter[val]);

			oScrT.s1 = scratch1[ownPos] ^ scratch1[othPos];
			oScrT.s2 = scratch2[ownPos] ^ scratch2[othPos];
			oScrT.s3 = scratch3[ownPos] ^ scratch3[othPos];
			oScrT.s4 = scratch4[ownPos] ^ scratch4[othPos];

			oScrT.s7 = scratch5[othPos];
			oScrT.s6 = scratch5[ownPos];
			oScrT.s5 = (scratch6[othPos] & 0xFFFFF000);

			oScrT.s0 = (oScrT.s0 >> 10) | (oScrT.s1 << 11);
			oScrT.s1 = (oScrT.s1 >> 21) | (oScrT.s2 << 11);
			oScrT.s2 = (oScrT.s2 >> 21) | (oScrT.s3 << 11);
			oScrT.s3 = (oScrT.s3 >> 21) | (oScrT.s4 << 11);
			oScrT.s4 = (oScrT.s4 >> 21);
			oScrT.s4 |= (scratch6[ownPos] & 0xFFFFF000);

			addr += 4592*val;

			output[addr] = oScrT; 	

			//if (get_global_id(0) == 0) printf("%x %x %x %x \n", oScrT.lo);
		} 

		othPos = elem & 0x7FF;

		while ((othPos == 0x7FF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
		}
	}  	
}


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round3 (				// Round 1 / 3
		__global uint8 * buffer0,
		__global uint4 * buffer1,
		__global uint * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = (grp & 3) << 3;

	__global uint4 * output = buffer1;
	__global uint8 * input = &buffer0[bucket*4592];

	__global uint * sideLoadR13 = (__global uint *) &buffer1[37748736 + grp*328];

	__local uint scratch[6912];
	__local uint * ht = &scratch[0];
	__local uint * scratch0 = &scratch[512];
	__local uint * scratch1 = &scratch[1792];
	__local uint * scratch2 = &scratch[3072];
	__local uint * scratch3 = &scratch[4352];
	__local uint * scratch4 = &scratch[5632];
	__local uint iCNT[2];
	__local uint pCNT[1];

	__global uint * inCounter = &counters[32768];
	__global uint * outCounter = &counters[49152];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R2: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif

	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 4592);
		pCNT[0] = 0;
	} 

	ht[lId] = 0xFFF;
	ht[lId+256] = 0xFFF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint8 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5;

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking4_4b(iScr0.lo, lId + 0, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr1.lo, lId + 256, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr2.lo, lId + 512, &scratch[0], &iCNT[1], mask, 0x18);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking4_4b(iScr3.lo, lId + 768, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr4.lo, lId + 1024, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr5.lo, lId + 1280, &scratch[0], &iCNT[1], mask, 0x18);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking4_4b(iScr0.lo, lId + 1536, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr1.lo, lId + 1792, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr2.lo, lId + 2048, &scratch[0], &iCNT[1], mask, 0x18);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	if ((lId + 3584) < iCNT[0])iScr2 = input[lId + 3584]; 

	masking4_4b(iScr3.lo, lId + 2304, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr4.lo, lId + 2560, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4b(iScr5.lo, lId + 2816, &scratch[0], &iCNT[1], mask, 0x18);

	if ((lId + 3840) < iCNT[0])iScr3 = input[lId + 3840];
	if ((lId + 4096) < iCNT[0])iScr4 = input[lId + 4096];
	if ((lId + 4352) < iCNT[0])iScr5 = input[lId + 4352];

	masking4_4bt(iScr0.lo, lId + 3072, &scratch[0], &iCNT[1], mask, 0x18);
	masking4_4bt(iScr1.lo, lId + 3328, &scratch[0], &iCNT[1], mask, 0x18);
	if ((lId + 3584) < iCNT[0])masking4_4bt(iScr2.lo, lId + 3584, &scratch[0], &iCNT[1], mask, 0x18);

	if ((lId + 3840) < iCNT[0])masking4_4bt(iScr3.lo, lId + 3840, &scratch[0], &iCNT[1], mask, 0x18);
	if ((lId + 4096) < iCNT[0])masking4_4bt(iScr4.lo, lId + 4096, &scratch[0], &iCNT[1], mask, 0x18);
	if ((lId + 4352) < iCNT[0])masking4_4bt(iScr5.lo, lId + 4352, &scratch[0], &iCNT[1], mask, 0x18);

	barrier(CLK_LOCAL_MEM_FENCE);	
	//if (lId == 0) printf("%d \n", iCNT[1]);
	uint inlim = min(iCNT[1], (uint) 1280);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);

	int ownPos = atomic_dec(&iCNT[0]);
	uint own = scratch0[ownPos];
	uint othPos = own & 0xFFF;
	
	while ((othPos == 0xFFF) && (ownPos >= 0)) {
		ownPos = atomic_dec(&iCNT[0]);
		if (ownPos >= 0) own = scratch0[ownPos];
		othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
	}

	othPos = (ownPos < inlim) ? othPos : 0xFFF;
	
	uint addr, elem, val;
	uint el0, el1;
	int pcnt = 0;
	uint4 oScrT;

	while (othPos < inlim) {
		elem = scratch0[othPos];
		oScrT.s0 = (own ^ elem) >> 14;

		uint sideAddr = atomic_inc(&pCNT[0]);			
		if (sideAddr < 1312) {	
			val = oScrT.s0 & 0x1FFF;		
			addr = atomic_inc(&outCounter[val]);

			sideLoadR13[sideAddr] = scratch4[ownPos] | (scratch4[othPos] << 16);

			oScrT.s1 = scratch1[ownPos] ^ scratch1[othPos];
			oScrT.s2 = scratch2[ownPos] ^ scratch2[othPos];
			oScrT.s3 = scratch3[ownPos] ^ scratch3[othPos];

			oScrT.s0 = (oScrT.s0 >> 13) | (oScrT.s1 << 5);
			oScrT.s1 = (oScrT.s1 >> 27) | (oScrT.s2 << 5);
			oScrT.s2 = (oScrT.s2 >> 27) | (oScrT.s3 << 5);
			oScrT.s3 = (oScrT.s3 >> 27);
					
			addr += 4592*val;
			oScrT.s3 |= ((sideAddr + 1312*grp) << 6);  

			output[addr] = oScrT;	
		} 

		othPos = elem & 0xFFF;
	
		while ((othPos == 0xFFF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
		}

	}  
}



__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round4 (				// Round 2 / 4
		__global uint4 * buffer0,
		__global uint4 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = grp & 3;

	__global uint4 * output = buffer2;
	__global uint4 * input = &buffer1[bucket*4592];

	__local uint scratch[5376];
	__local uint * ht = &scratch[0];
	__local uint * scratch0 = &scratch[512];
	__local uint * scratch1 = &scratch[1728];
	__local uint * scratch2 = &scratch[2944];
	__local uint * scratch3 = &scratch[4160];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[49152];
	__global uint * outCounter = &counters[65536];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R3: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif	
	
	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 4592);
	} 

	ht[lId] = 0x7FF;
	ht[lId+256] = 0x7FF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint4 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5;

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking4_4(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking4_4(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr5, &scratch[0], &iCNT[1], mask, 0x3);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking4_4(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	if ((lId + 3584) < iCNT[0]) iScr2 = input[lId + 3584]; 

	masking4_4t(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4t(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4t(iScr5, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 3840) < iCNT[0]) iScr3 = input[lId + 3840];
	if ((lId + 4096) < iCNT[0]) iScr4 = input[lId + 4096];
	if ((lId + 4352) < iCNT[0]) iScr5 = input[lId + 4352];

	masking4_4t(iScr0, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_4t(iScr1, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 3584) < iCNT[0]) masking4_4t(iScr2, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 3840) < iCNT[0]) masking4_4t(iScr3, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4096) < iCNT[0]) masking4_4t(iScr4, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 4352) < iCNT[0]) masking4_4t(iScr5, &scratch[0], &iCNT[1], mask, 0x3);

	barrier(CLK_LOCAL_MEM_FENCE);	
	uint inlim = min(iCNT[1], (uint) 1216);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);		 

	int ownPos = atomic_dec(&iCNT[0]);
	uint own = scratch0[ownPos];
	uint othPos = own & 0x7FF;

	while ((othPos == 0x7FF) && (ownPos >= 0)) {
		ownPos = atomic_dec(&iCNT[0]);
		if (ownPos >= 0) own = scratch0[ownPos];
		othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
	}

	othPos = (ownPos < inlim) ? othPos : 0x7FF;

	uint addr, elem, val;
	uint el0, el1;
	uint4 oScrT;

	while (othPos < inlim) {

	
		elem = scratch0[othPos];
		oScrT.s0 = (own ^ elem) >> 11;
	
		if (oScrT.s0 != 0) {
			val = oScrT.s0 & 0xFFF;
			addr = atomic_inc(&outCounter[val]);

			oScrT.s1 = scratch1[ownPos] ^ scratch1[othPos];
			oScrT.s2 = scratch2[ownPos] ^ scratch2[othPos];
			el0 = scratch3[ownPos];
			el1 = scratch3[othPos];

			oScrT.s3 = (el0 ^ el1) & 0x1F;
			oScrT.s0 = (oScrT.s0 >> 12) | (oScrT.s1 << 9);
			oScrT.s1 = (oScrT.s1 >> 23) | (oScrT.s2 << 9);
			oScrT.s2 = (oScrT.s2 >> 23) | (oScrT.s3 << 9);
			
			uint2 tmp = compress(el0 >> 6,el1 >> 6);
			addr += 8688*val;

			oScrT.s3 = tmp.s0;
			oScrT.s2 |= (tmp.s1 << 14); 	
			output[addr] = oScrT; 	
		} 

		othPos = elem & 0x7FF;

		while ((othPos == 0x7FF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0x7FF : 0x7FF;
		}
	}  
	
}


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round5 (				// Round 3 / 5
		__global uint4 * buffer0,
		__global uint4 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = grp & 3;

	__global uint4 * output = buffer1;
	__global uint4 * input = &buffer2[bucket*8688];

	__local uint scratch[7936];
	__local uint * ht = &scratch[0];
	__local uint * scratch0 = &scratch[1024];
	__local uint * scratch1 = &scratch[3328];
	__local uint * scratch2 = &scratch[5632];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[65536];
	__global uint * outCounter = &counters[81920];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R4: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif	
	
	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 8688);
	} 

	ht[lId] = 0xFFF;
	ht[lId+256] = 0xFFF;
	ht[lId+512] = 0xFFF;
	ht[lId+768] = 0xFFF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint4 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5; 

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking4_3b(iScr0, lId + 0, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr1, lId + 256, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr2, lId + 512, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking4_3b(iScr3, lId + 768, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr4, lId + 1024, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr5, lId + 1280, &scratch[0], &iCNT[1], mask, 0x3);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking4_3b(iScr0, lId + 1536, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr1, lId + 1792, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr2, lId + 2048, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	iScr2 = input[lId + 3584]; 

	masking4_3b(iScr3, lId + 2304, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr4, lId + 2560, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr5, lId + 2816, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 3840];
	iScr4 = input[lId + 4096];
	iScr5 = input[lId + 4352];

	masking4_3b(iScr0, lId + 3072, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr1, lId + 3328, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr2, lId + 3584, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 4608];
	iScr1 = input[lId + 4864];
	iScr2 = input[lId + 5120]; 

	masking4_3b(iScr3, lId + 3840, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr4, lId + 4096, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr5, lId + 4352, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 5376];
	iScr4 = input[lId + 5632];
	iScr5 = input[lId + 5888];

	masking4_3b(iScr0, lId + 4608, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr1, lId + 4864, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr2, lId + 5120, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 6144];
	iScr1 = input[lId + 6400];
	iScr2 = input[lId + 6656]; 

	masking4_3b(iScr3, lId + 5376, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr4, lId + 5632, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr5, lId + 5888, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 6912];
	iScr4 = input[lId + 7168];
	iScr5 = input[lId + 7424];

	masking4_3b(iScr0, lId + 6144, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr1, lId + 6400, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3b(iScr2, lId + 6656, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 7680) < iCNT[0]) iScr0 = input[lId + 7680];
	if ((lId + 7936) < iCNT[0]) iScr1 = input[lId + 7936];
	if ((lId + 8192) < iCNT[0]) iScr2 = input[lId + 8192]; 

	masking4_3bt(iScr3, lId + 6912, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3bt(iScr4, lId + 7168, &scratch[0], &iCNT[1], mask, 0x3);
	masking4_3bt(iScr5, lId + 7424, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 8448) < iCNT[0]) iScr3 = input[lId + 8448];
	
	if ((lId + 7680) < iCNT[0]) masking4_3bt(iScr0, lId + 7680, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 7936) < iCNT[0]) masking4_3bt(iScr1, lId + 7936, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 8192) < iCNT[0]) masking4_3bt(iScr2, lId + 8192, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 8448) < iCNT[0]) masking4_3bt(iScr3, lId + 8448, &scratch[0], &iCNT[1], mask, 0x3); 	

	barrier(CLK_LOCAL_MEM_FENCE);	
	uint inlim = min(iCNT[1], (uint) 2304);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);	

	int ownPos = atomic_dec(&iCNT[0]);
	uint own = scratch0[ownPos];
	uint othPos = own & 0xFFF;

	while ((othPos == 0xFFF) && (ownPos >= 0)) {
		ownPos = atomic_dec(&iCNT[0]);
		if (ownPos >= 0) own = scratch0[ownPos];
		othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
	}

	othPos = (ownPos < inlim) ? othPos : 0xFFF;

	uint addr, elem, val;
	uint el0, el1;
	int pcnt = 0;
	uint4 oScrT;

	while (othPos < inlim) {

	
		elem = scratch0[othPos];
		oScrT.s0 = (own ^ elem) >> 12;
	
		if (oScrT.s0 != 0) {
			val = oScrT.s0 & 0xFFF;
			addr = atomic_inc(&outCounter[val]);

			oScrT.s1 = scratch1[ownPos] ^ scratch1[othPos];
			el0 = scratch2[ownPos];
			el1 = scratch2[othPos];
			
			oScrT.s2 = (el0 ^ el1) & 0x3FFF;
			oScrT.s3 = 0;

			oScrT.s0 = (oScrT.s0 >> 12) | (oScrT.s1 << 8);
			oScrT.s1 = (oScrT.s1 >> 24) | (oScrT.s2 << 8);
			oScrT.s2 = bucket;
			oScrT.s3 = (el0 >> 14) | ((el1 >> 14) << 16);
		
			addr += 8688*(val & 0xFFF);
			output[addr] = oScrT; 	
		} 

		othPos = elem & 0xFFF;

		while ((othPos == 0xFFF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
		}
	}  
}


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round6 (				// Round 4 / 6
		__global uint4 * buffer0,
		__global uint4 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = grp & 3;

	__global uint2 * output = (__global uint2 *) &buffer1[48496640];
	__global uint4 * input = &buffer1[bucket*8688];


	__local uint scratch[5632];
	__local uint * ht = &scratch[0];
	__local uint * scratch0 = &scratch[1024];
	__local uint * scratch1 = &scratch[3328];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[81920];
	__global uint * outCounter = &counters[98304];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R5: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif	

	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 8688);
	}

	ht[lId] = 0xFFF;
	ht[lId+256] = 0xFFF;
	ht[lId+512] = 0xFFF;
	ht[lId+768] = 0xFFF;

	barrier(CLK_LOCAL_MEM_FENCE);	

	uint4 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5; 

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024];
	iScr5 = input[lId + 1280];
	
	masking2_2b(iScr0.lo, lId + 0, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr1.lo, lId + 256, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr2.lo, lId + 512, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048]; 

	masking2_2b(iScr3.lo, lId + 768, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr4.lo, lId + 1024, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr5.lo, lId + 1280, &scratch[0], &iCNT[1], mask, 0x3);	

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560];
	iScr5 = input[lId + 2816];

	masking2_2b(iScr0.lo, lId + 1536, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr1.lo, lId + 1792, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr2.lo, lId + 2048, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	iScr2 = input[lId + 3584]; 

	masking2_2b(iScr3.lo, lId + 2304, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr4.lo, lId + 2560, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr5.lo, lId + 2816, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 3840];
	iScr4 = input[lId + 4096];
	iScr5 = input[lId + 4352];

	masking2_2b(iScr0.lo, lId + 3072, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr1.lo, lId + 3328, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr2.lo, lId + 3584, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 4608];
	iScr1 = input[lId + 4864];
	iScr2 = input[lId + 5120]; 

	masking2_2b(iScr3.lo, lId + 3840, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr4.lo, lId + 4096, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr5.lo, lId + 4352, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 5376];
	iScr4 = input[lId + 5632];
	iScr5 = input[lId + 5888];

	masking2_2b(iScr0.lo, lId + 4608, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr1.lo, lId + 4864, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr2.lo, lId + 5120, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 6144];
	iScr1 = input[lId + 6400];
	iScr2 = input[lId + 6656]; 

	masking2_2b(iScr3.lo, lId + 5376, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr4.lo, lId + 5632, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr5.lo, lId + 5888, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 6912];
	iScr4 = input[lId + 7168];
	iScr5 = input[lId + 7424];

	masking2_2b(iScr0.lo, lId + 6144, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr1.lo, lId + 6400, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2b(iScr2.lo, lId + 6656, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 7680) < iCNT[0]) iScr0 = input[lId + 7680];
	if ((lId + 7936) < iCNT[0]) iScr1 = input[lId + 7936];
	if ((lId + 8192) < iCNT[0]) iScr2 = input[lId + 8192]; 

	masking2_2bt(iScr3.lo, lId + 6912, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2bt(iScr4.lo, lId + 7168, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2bt(iScr5.lo, lId + 7424, &scratch[0], &iCNT[1], mask, 0x3);

	if ((lId + 8448) < iCNT[0]) iScr3 = input[lId + 8448];
	
	if ((lId + 7680) < iCNT[0]) masking2_2bt(iScr0.lo, lId + 7680, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 7936) < iCNT[0]) masking2_2bt(iScr1.lo, lId + 7936, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 8192) < iCNT[0]) masking2_2bt(iScr2.lo, lId + 8192, &scratch[0], &iCNT[1], mask, 0x3);
	if ((lId + 8448) < iCNT[0]) masking2_2bt(iScr3.lo, lId + 8448, &scratch[0], &iCNT[1], mask, 0x3); 

	barrier(CLK_LOCAL_MEM_FENCE);	
	uint inlim = min(iCNT[1], (uint) 2304);
	if (lId == 0) iCNT[0] = inlim-1;
	barrier(CLK_LOCAL_MEM_FENCE);	

		int ownPos = atomic_dec(&iCNT[0]);
		uint own = scratch0[ownPos];
		uint othPos = own & 0xFFF;
	
		while ((othPos == 0xFFF) && (ownPos >= 0)) {
			ownPos = atomic_dec(&iCNT[0]);
			if (ownPos >= 0) own = scratch0[ownPos];
			othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
		}

		othPos = (ownPos < inlim) ? othPos : 0xFFF;
	
		uint addr, elem, val;
		uint el0, el1;
		int pcnt = 0;
		uint2 oScrT;

		while (othPos < inlim) {

		
			elem = scratch0[othPos];
			oScrT.s0 = (own ^ elem) >> 12;
		
			if (oScrT.s0 != 0) {
				val = oScrT.s0 & 0xFFF;
				addr = atomic_inc(&outCounter[val]);

				uint el0 = scratch1[ownPos];
				uint el1 = scratch1[othPos];	

				oScrT.s1 = (el0 ^ el1) & 0x3FFFF;
				el0 = el0 >> 18;
				el1 = el1 >> 18;	

				oScrT.s0 = (oScrT.s0 >> 12) | (oScrT.s1 << 8);
				oScrT.s1 = compress2(el0,el1);
				
				addr += 8688*(val & 0xFFF);

				oScrT.s1 |= (bucket << 26);
				oScrT.s0 |= ((bucket >> 6) << 26); 	
			
				output[addr] = oScrT; 	
			} 

			othPos = elem & 0xFFF;
	
			while ((othPos == 0xFFF) && (ownPos >= 0)) {
				ownPos = atomic_dec(&iCNT[0]);
				if (ownPos >= 0) own = scratch0[ownPos];
				othPos = (ownPos >= 0) ? own & 0xFFF : 0xFFF;
			}
		}  
}


__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void round7 (				// Round 5 / 7
		__global uint4 * buffer0,
		__global uint4 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * res,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {
	uint lId = get_local_id(0);
	uint grp = get_group_id(0); 

	uint bucket = grp >> 2;
	uint mask = grp & 3;

	__global uint4 * output = &buffer1[67305472];
	__global uint2 * outputR46 = (__global uint2 *) &buffer1[48496640];
	__global uint4 * input = &buffer1[48496640 + bucket*4344];

	__local uint scratch[5632];
	__local uint * ht = &scratch[0];
	__local uint * scratch0 = &scratch[1024];
	__local uint * scratch1 = &scratch[3328];
	__local uint iCNT[2];

	__global uint * inCounter = &counters[98304];
	__global uint * outCounter = &counters[114688];

	#ifdef PRINT
	if (get_global_id(0) == 0) {
		uint sum=0;
		for (uint i=0; i<16384; i++) {
			sum += inCounter[i];			
		}
		printf("R6: %d %d %d %d \n", sum, inCounter[0], inCounter[1], inCounter[2]);
	} 
	#endif

	if (lId == 0) {
		iCNT[1] = 0;
		iCNT[0] = min(inCounter[bucket],(uint) 8688);
	}

	ht[lId] = 0xFFF;
	ht[lId+256] = 0xFFF;
	ht[lId+512] = 0xFFF;
	ht[lId+768] = 0xFFF;

	barrier(CLK_LOCAL_MEM_FENCE);

	uint4 iScr0,iScr1,iScr2,iScr3,iScr4,iScr5; 

	iScr0 = input[lId];
	iScr1 = input[lId + 256];
	iScr2 = input[lId + 512];

	iScr3 = input[lId + 768];
	iScr4 = input[lId + 1024]; 
	iScr5 = input[lId + 1280];

	masking2_2(iScr0.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr0.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr2.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr2.hi, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 1536];
	iScr1 = input[lId + 1792];
	iScr2 = input[lId + 2048];

	masking2_2(iScr3.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr3.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr4.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr4.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr5.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr5.hi, &scratch[0], &iCNT[1], mask, 0x3);

	iScr3 = input[lId + 2304];
	iScr4 = input[lId + 2560]; 
	iScr5 = input[lId + 2816];

	masking2_2(iScr0.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr0.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr2.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr2.hi, &scratch[0], &iCNT[1], mask, 0x3);

	iScr0 = input[lId + 3072];
	iScr1 = input[lId + 3328];
	if (2*(lId+3584) < iCNT[0]) iScr2 = input[lId + 3584];

	masking2_2(iScr3.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr3.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr4.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr4.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr5.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr5.hi, &scratch[0], &iCNT[1], mask, 0x3);

	if (2*(lId+3840) < iCNT[0]) iScr3 = input[lId + 3840];
	if (2*(lId+4096) < iCNT[0]) iScr4 = input[lId + 4096]; 
	if (2*(lId+4352) < iCNT[0]) iScr5 = input[lId + 4352];

	masking2_2(iScr0.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr0.hi, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.lo, &scratch[0], &iCNT[1], mask, 0x3);
	masking2_2(iScr1.hi, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+3584) < iCNT[0])  masking2_2(iScr2.lo, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+3584)+1 < iCNT[0])masking2_2(iScr2.hi, &scratch[0], &iCNT[1], mask, 0x3);

	if (2*(lId+3840)+0 < iCNT[0]) masking2_2(iScr3.lo, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+3840)+1 < iCNT[0]) masking2_2(iScr3.hi, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+4096)+0 < iCNT[0]) masking2_2(iScr4.lo, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+4096)+1 < iCNT[0]) masking2_2(iScr4.hi, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+4352)+0 < iCNT[0]) masking2_2(iScr5.lo, &scratch[0], &iCNT[1], mask, 0x3);
	if (2*(lId+4352)+1 < iCNT[0]) masking2_2(iScr5.hi, &scratch[0], &iCNT[1], mask, 0x3);

	barrier(CLK_LOCAL_MEM_FENCE);	
	if (lId == 0) iCNT[1] = min(iCNT[1], (uint) 2304);
	barrier(CLK_LOCAL_MEM_FENCE);

	uint ownPos = lId;
	uint own = scratch0[ownPos];
	uint othPos = own & 0xFFF;

	uint ownEl;

	if ((own & 0xFFF) != 0xFFF) {
		ownEl = scratch1[ownPos];
	}

	uint2 oScrT;

	while (ownPos < iCNT[1]) {
		uint addr = (othPos < iCNT[1]) ? othPos : ownPos+256;
		uint elem = scratch0[addr];
		
		if (othPos < iCNT[1]) {
			uint oth = elem;
			if (((own ^ oth) & 0x03FFF000) == 0) {
				uint4 elem;
				elem.s0 = own;
				elem.s1 = ownEl;

				elem.s2 = oth;
				elem.s3 = scratch1[othPos];

				uint4 naddr;

				naddr.s01 = decompress2(elem.s1 & 0x3FFFFFF);			// Unpack R4 / 6
				naddr.s23 = decompress2(elem.s3 & 0x3FFFFFF);
				
				elem.s0 = elem.s0 >> 26;
				elem.s1 = elem.s1 >> 26;
				elem.s0 = elem.s0 << 6;
				elem.s0 |= elem.s1;
				
				naddr.s0 += 8688*elem.s0;
				naddr.s1 += 8688*elem.s0;

				elem.s2 = elem.s2 >> 26;
				elem.s3 = elem.s3 >> 26;
				elem.s2 = elem.s2 << 6;
				elem.s2 |= elem.s3;
				
				naddr.s2 += 8688*elem.s2;
				naddr.s3 += 8688*elem.s2;

				bool ok = true;

				ok = ok && (naddr.s0 != naddr.s1) && (naddr.s0 != naddr.s2) && (naddr.s0 != naddr.s3); 
				ok = ok && (naddr.s1 != naddr.s2) && (naddr.s1 != naddr.s3) && (naddr.s2 != naddr.s3); 
				
				if (ok) {
					addr = atomic_inc(&outCounter[oScrT.s0 & 0xFFF]);
					if (addr < 4096) {
						output[addr] = naddr;	
					}
				}
			}		
		} else { 
			own = elem;
			ownPos += 256;
			if (((own & 0xFFF) != 0xFFF) && (ownPos < iCNT[1])) {
				ownEl = scratch1[ownPos];
			}
		}

		othPos = elem & 0xFFF;
	}  	
}


__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void combine (
		__global uint8 * buffer0,
		__global uint4 * buffer1,
		__global uint4 * buffer2,
		__global uint * counters,
		__global uint4 * output0,
		const uint extra,
		const ulong8 hashState,
		const ulong nonce) {

	 uint gId = get_group_id(0);
	uint lId = get_local_id(0);

	__global uint * outCounters = (__global uint*) &output0[0];

	__global uint * sideLoadR13 = (__global uint *) &buffer1[37748736];

	__global uint2 * R46Out = (__global uint2 *) &buffer1[48496640];
	__global uint4 * R57Out = &buffer1[67305472];

	__local uint scratch[256];
	__local uint ok[1];

	uint2 tmps;

	#ifdef PRINT
		if (get_global_id(0) == 0) {
			 printf("R5: %d \n", counters[114688]);
		}
	#endif

	if (gId < counters[114688]) {
		if (lId == 0) {
			uint4 tmp;
			tmp = R57Out[gId];
			
			scratch[128 + 4*lId+0] = tmp.s0;
			scratch[128 + 4*lId+1] = tmp.s1;
			scratch[128 + 4*lId+2] = tmp.s2;
			scratch[128 + 4*lId+3] = tmp.s3;
		}

		bool check = true;
		if (lId == 0) ok[0] = 0;
		barrier(CLK_LOCAL_MEM_FENCE); 

		if (lId < 4) {							// Unpack R3 / 5
			uint addr = scratch[128 + lId];
			if (addr < 37617664) {
				uint4 tmp = buffer1[addr];
				
				atomic_xor(&ok[0], tmp.s1 >> 18);

				tmp.s0 = tmp.s3 & 0xFFFF;
				tmp.s1 = tmp.s3 >> 16;

				tmp.s0 += 8688*tmp.s2;
				tmp.s1 += 8688*tmp.s2;

				scratch[2*lId] = tmp.s0;	
				scratch[2*lId+1] = tmp.s1;	
			}
		}			

		barrier(CLK_LOCAL_MEM_FENCE); 
		check = (ok[0] == 0);
		barrier(CLK_LOCAL_MEM_FENCE); 

		if ((lId < 8) && (check)) {							// Unpack R2 / 4
			uint addr = scratch[lId];
			if (addr < 37617664) {
				uint4 tmp = buffer2[addr];
				
				tmp.s2 = tmp.s2 >> 14;
				tmp.s01 = decompress(tmp.s32);

				scratch[128+2*lId] = tmp.s0;	
				scratch[128+2*lId+1] = tmp.s1;	
			}
		}
			
		barrier(CLK_LOCAL_MEM_FENCE); 
		
		if ((lId < 16) && (check)) {							// Unpack R1 / 3
			uint addr = scratch[128+lId];
			if (addr < 42991616) {
				uint2 tmp;
			
				tmp.s0 = sideLoadR13[addr];
				tmp.s1 = tmp.s0 & 0xFFFF;
				tmp.s0 = tmp.s0 >> 16;

				tmp.s0 += 4592*(addr / 5248);
				tmp.s1 += 4592*(addr / 5248);

				scratch[144+2*lId] = tmp.s0;
				scratch[144+2*lId+1] = tmp.s1;
			} 
		}

		barrier(CLK_LOCAL_MEM_FENCE); 

		
		if ((lId < 32) && (check)) {							// Unpack R2
			uint addr = scratch[144+lId];
			if (addr < 37617664) {
				uint8 tmp;
			
				tmp = buffer0[addr];
				
				atomic_xor(&ok[0], tmp.s4 & 0x3F);
				
				tmp.s0 = (tmp.s4 >> 12) << 7;
				tmp.s1 = tmp.s6;

				tmp.s2 = (tmp.s5 >> 12) << 7;
				tmp.s3 = tmp.s7;

				tmp.s0 |= (tmp.s1 >> 25);
				tmp.s2 |= (tmp.s3 >> 25);

				tmp.s1 &= 0x1FFFFFF;
				tmp.s3 &= 0x1FFFFFF;

				scratch[4*lId] = tmp.s0;
				scratch[4*lId+1] = tmp.s1;
				scratch[4*lId+2] = tmp.s2;
				scratch[4*lId+3] = tmp.s3;				
			} 
		} 
		
	
		barrier(CLK_LOCAL_MEM_FENCE);	

		if ((ok[0] == 0) && (check)) {
			
			scratch[128 + 2*lId] = 0xFFF;
			scratch[128 + 2*lId+1] = 0xFFF;
			
			barrier(CLK_LOCAL_MEM_FENCE);

			uint2 listEntry ;

			uint elem = scratch[lId] & 0x3F;
			listEntry.s0 = atomic_xchg(&scratch[128 + elem], lId);

			elem = scratch[64+lId] & 0x3F;
			listEntry.s1 = atomic_xchg(&scratch[128 + elem], lId+64);
			

			barrier(CLK_LOCAL_MEM_FENCE);


			scratch[128 + lId] = listEntry.s0;
			scratch[128 + 64 + lId] = listEntry.s1;
			

			barrier(CLK_LOCAL_MEM_FENCE);
	

			int next = scratch[128 + lId];
			while (next < 128)  {	
				if (scratch[lId] == scratch[next]) {
					atomic_inc(&ok[0]);
				} 	
				next = scratch[128 + next];
			} 

			next = scratch[128 + 64 + lId];
			while (next < 128)  {	
				if (scratch[64 + lId] == scratch[next]) {
					atomic_inc(&ok[0]);
				} 	
				next = scratch[128 + next];
			} 	
		} 

		barrier(CLK_LOCAL_MEM_FENCE);

		if ((ok[0] == 0) && (check)) {
			uint addr;
			if (lId == 0) addr = atomic_inc(&outCounters[0]);	

			if (lId < 64) {
				uint2 elem;
				elem.s0 = scratch[2*lId];
				elem.s1 = scratch[2*lId+1];

				if (elem.s0 > elem.s1) elem.s01 = elem.s10;

				scratch[128+2*lId] = elem.s0;
				scratch[128+2*lId+1] = elem.s1;				// Elements sorted by 2 Elem
			}

			barrier(CLK_LOCAL_MEM_FENCE);	

			uint2 tmp2;
		
			if (lId < 64) {
				tmp2.s0 = lId >> 1;
				tmp2.s1 = (scratch[128+4*tmp2.s0+0] > scratch[128+4*tmp2.s0+2]) ? (lId ^ 0x1) : lId;

				scratch[2*lId] = scratch[128+2*tmp2.s1];
				scratch[2*lId+1] = scratch[128+2*tmp2.s1+1];		// Elements sorted by 4 Elem
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (lId < 64) {
				tmp2.s0 = lId >> 2;
				tmp2.s1 = (scratch[8*tmp2.s0+0] > scratch[8*tmp2.s0+4]) ? (lId ^ 0x2) : lId;

				scratch[128+2*lId+0] = scratch[2*tmp2.s1+0];		// Elements sorted by 8 Elem
				scratch[128+2*lId+1] = scratch[2*tmp2.s1+1];
			}
		
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lId < 64) {
				tmp2.s0 = lId >> 3;
				tmp2.s1 = (scratch[128+16*tmp2.s0+0] > scratch[128+16*tmp2.s0+8]) ? (lId ^ 0x4) : lId;

				scratch[2*lId+0] = scratch[128+2*tmp2.s1+0];		// Elements sorted by 16 Elem
				scratch[2*lId+1] = scratch[128+2*tmp2.s1+1];
			}
		
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lId < 64) {
				tmp2.s0 = lId >> 4;
				tmp2.s1 = (scratch[32*tmp2.s0+0] > scratch[32*tmp2.s0+16]) ? (lId ^ 0x8) : lId;

				scratch[128+2*lId+0] = scratch[2*tmp2.s1+0];		// Elements sorted by 32 Elem
				scratch[128+2*lId+1] = scratch[2*tmp2.s1+1];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (lId < 64) {
				tmp2.s0 = lId >> 5;
				tmp2.s1 = (scratch[128+64*tmp2.s0+0] > scratch[128+64*tmp2.s0+32]) ? (lId ^ 0x10) : lId;

				scratch[2*lId+0] = scratch[128+2*tmp2.s1+0];		// Elements sorted by 64 Elem
				scratch[2*lId+1] = scratch[128+2*tmp2.s1+1];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (lId < 64) {
				tmp2.s0 = lId >> 6;
				tmp2.s1 = (scratch[128*tmp2.s0+0] > scratch[128*tmp2.s0+64]) ? (lId ^ 0x20) : lId;

				scratch[128+2*lId+0] = scratch[2*tmp2.s1+0];		// Elements sorted by 128 Elem
				scratch[128+2*lId+1] = scratch[2*tmp2.s1+1];
			}

			barrier(CLK_LOCAL_MEM_FENCE);					// All Elements sorted	
			
			if (lId == 0) scratch[0] = addr;

			barrier(CLK_LOCAL_MEM_FENCE);

			addr = scratch[0];

			if ((addr < 16) && (lId < 32)) {
				uint4 tmp;
				tmp.s0 = scratch[128+4*lId];
				tmp.s1 = scratch[128+4*lId+1];
				tmp.s2 = scratch[128+4*lId+2];
				tmp.s3 = scratch[128+4*lId+3];

				output0[1 + 32*addr + lId] = tmp;
			}   
			
		}
	} 

	//if (get_global_id(0) == 0) outCounters[0] = 2;

}

