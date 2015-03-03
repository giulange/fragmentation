//	INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>        	/* errno */
#include <string.h>       	/* strerror */
#include <math.h>			// ceil
#include <time.h>			// CLOCKS_PER_SEC

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * 	PARS
 */
#define 		TILE_DIM 					32

// ctid: current tid
__device__ unsigned int fMOD(unsigned int ctix,unsigned int mask_len){
	return ((ctix % mask_len) == 0) ? 0 : 1;
}

__device__ int fNEAR(unsigned int ctix,unsigned int RADIUS){
	// This function is valid only for blockIdx.x>0
	unsigned int mask_len = RADIUS*2+1;
	return ((ctix % mask_len) <= RADIUS) ? -(ctix % mask_len) : -(ctix % mask_len)+mask_len;
}

__device__ unsigned int fMAX(unsigned int available_col,unsigned int required_col){
	return (required_col<available_col) ? required_col : available_col;
}

__global__
void gtranspose_char(unsigned char *O, const unsigned char *I, unsigned WIDTH, unsigned HEIGHT)
{
	unsigned int tix = threadIdx.x;
	unsigned int tiy = threadIdx.y;
	unsigned int bix = blockIdx.x;
	unsigned int biy = blockIdx.y;
	unsigned int bdx = blockDim.x;
	unsigned int bdy = blockDim.y;

	//					  |--grid------|   |-block--|   |-thread--|
	unsigned int itid	= WIDTH*bdy*biy  + WIDTH*tiy  + bix*bdx+tix;
	unsigned int otid	= HEIGHT*bdx*bix + HEIGHT*tix + biy*bdy+tiy;
	unsigned int xtid	= bix*bdx+tix;
	unsigned int ytid	= biy*bdy+tiy;

	//if( is32Multiple || (xtid<WIDTH && ytid<HEIGHT) ){
	if( xtid<WIDTH && ytid<HEIGHT ){
		O[ otid ] = I[ itid ];
		//__syncthreads();
	}
}
__global__
void gtranspose_double(double *O, const double *I, unsigned WIDTH, unsigned HEIGHT)
{
	unsigned int tix = threadIdx.x;
	unsigned int tiy = threadIdx.y;
	unsigned int bix = blockIdx.x;
	unsigned int biy = blockIdx.y;
	unsigned int bdx = blockDim.x;
	unsigned int bdy = blockDim.y;

	//					  |--grid------|   |-block--|   |-thread--|
	unsigned int itid	= WIDTH*bdy*biy  + WIDTH*tiy  + bix*bdx+tix;
	unsigned int otid	= HEIGHT*bdx*bix + HEIGHT*tix + biy*bdy+tiy;
	unsigned int xtid	= bix*bdx+tix;
	unsigned int ytid	= biy*bdy+tiy;

	//if( is32Multiple || (xtid<WIDTH && ytid<HEIGHT) ){
	if( xtid<WIDTH && ytid<HEIGHT ){
		O[ otid ] = I[ itid ];
		//__syncthreads();
	}
}
__global__
void complementary_to_ONE(const unsigned char *ONE, const unsigned char *BIN, unsigned char *COMP, unsigned WIDTH, unsigned HEIGHT) {

	/**
	 * 	NOTES
	 * 	In rural fragmentation:		I need COMP to mask out urban pixels
	 * 								from being fragmented (because only rural
	 * 								pixels can be fragmented).
	 * 	In urban fragmentation:		I need COMP to run the whole fragmentation program
	 * 								(instead of BIN as in rural frag.), and I use BIN
	 * 								to mask out rural pixels.
	 */
	unsigned int r		= threadIdx.x;
	unsigned int c		= threadIdx.y;
	unsigned int bix	= blockIdx.x;
	unsigned int biy	= blockIdx.y;
	unsigned int bdx	= blockDim.x;
	unsigned int bdy	= blockDim.y;

	unsigned int tix	= bix * bdx + r;
	unsigned int tiy	= biy * bdy + c;
	unsigned int tid 	= tix + tiy*WIDTH;

	if( tix<WIDTH && tiy<HEIGHT ){
		COMP[ tid ] = ONE[tid] - BIN[ tid ];
	}
}

__global__ void Vcumsum_char( 	const unsigned char	*IN			,
								unsigned long int 	map_width	,
								unsigned long int 	map_height	,
								double 				*OUT		,
								unsigned int		RADIUS		){
	/* NOTES:
	 *  This kernel performs the cumulative sum along the Y axis. The block is
	 *  made by all threads displaced along X (i.e. [1024,1,1]. Each thread of
	 *  the block is in charge of summing the (RADIUS*2+1) pixels below thread.
	 *  So each block is in charge of a tile of size X<--blockDim.x and
	 *  Y<--mask_len.
	 *  The first "if" avoids going beyond the last row and the second "if"
	 *  avoids going beyond the last column.
	 *  block 	= [32*32,	1,			1]
	 *  tile	= [32*32,	mask_len,	1]
	 *
	 *  If I avoid the MASK in this kernel I avoid the gtransform on ROI.
	 *  If I don't mask here signifies that all pixels within IN are used to
	 *  compute the fragmentation of all pixels; further, masking at the end
	 *  of whole program I exclude all points outside ROI. This seems to be
	 *  the most wonderful situation.
	 */
	unsigned long int ii;
	unsigned long int mask_len 	= RADIUS*2+1;
	unsigned long int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned long int tiy 		= blockIdx.y*mask_len;//+ threadIdx.y;
	unsigned long int tid 		= tix + tiy*map_width;

	if( tix < map_width  ){ // && tiy < map_height
		// Here I copy the first row of values within the tile:
		if(tid<map_width*map_height)
			OUT[tid] = IN[tid];//*MASK[tid];
		/*	Here, for every row(i.e. block) and every thread of the row(i.e. block),
		 * 	I sum the current(tid+ii) and the previous(tid+ii-1) values and write
		 * 	the result in the current position(tid+ii) of the output array.
		 */
		for(ii=1;ii<mask_len;ii++)
			if(tiy+ii<map_height)
				OUT[tid+ii*map_width] = OUT[tid+(ii-1)*map_width] + IN[tid+ii*map_width];//*MASK[tid+ii*map_width];
	}
}
__global__ void Vcumsum_double( const double 		*IN			,
								unsigned long int 	map_width	,
								unsigned long int 	map_height	,
								double 				*OUT		,
								unsigned int		RADIUS		){
	/* NOTES:
	 *  This kernel performs the cumulative sum along the Y axis. The block is
	 *  made by all threads displaced along X (i.e. [1024,1,1]. Each thread of
	 *  the block is in charge of summing the (RADIUS*2+1) pixels below thread.
	 *  So each block is in charge of a tile of size X<--blockDim.x and
	 *  Y<--mask_len.
	 *  The first "if" avoids going beyond the last row and the second "if"
	 *  avoids going beyond the last column.
	 *  block 	= [32*32,	1,			1]
	 *  tile	= [32*32,	mask_len,	1]
	 *
	 *  If I avoid the MASK in this kernel I avoid the gtransform on ROI.
	 *  If I don't mask here signifies that all pixels within IN are used to
	 *  compute the fragmentation of all pixels; further, masking at the end
	 *  of whole program I exclude all points outside ROI. This seems to be
	 *  the most wonderful situation.
	 */
	unsigned long int ii;
	unsigned long int mask_len 	= RADIUS*2+1;
	unsigned long int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned long int tiy 		= blockIdx.y*mask_len;//+ threadIdx.y;
	unsigned long int tid 		= tix + tiy*map_width;

	if( tix < map_width  ){ // && tiy < map_height
		// Here I copy the first row of values within the tile:
		if(tid<map_width*map_height)
			OUT[tid] = IN[tid];//*MASK[tid];
		/*	Here, for every row(i.e. block) and every thread of the row(i.e. block),
		 * 	I sum the current(tid+ii) and the previous(tid+ii-1) values and write
		 * 	the result in the current position(tid+ii) of the output array.
		 */
		for(ii=1;ii<mask_len;ii++)
			if(tiy+ii<map_height)
				OUT[tid+ii*map_width] = OUT[tid+(ii-1)*map_width] + IN[tid+ii*map_width];//*MASK[tid+ii*map_width];
	}
}
__global__ void sum_of_3_LINES(		const double 	*IN			,
									unsigned int 	map_width	,
									unsigned int 	map_height	,
									double 			*OUT		,
									unsigned int	RADIUS		){

	/* NOTES:
	 *  This kernel performs the algebraic sum of three columns:
	 *   > the column on the right side 	+[tid+ii+radius  				 ]
	 *   > the column on the left side		-[tid+ii-radius-1				 ]
	 *   > the nearest terminal column 		+[tid+ii+fNEAR(tix+ii+1,mask_len)]
	 *
	 *  Particular cases are figured out according to blockIdx.x position.
	 *  See later comments!
	 */
	unsigned int ii			= 0;
	unsigned int mask_len 	= RADIUS*2+1;
	unsigned int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int tiy 		= blockIdx.y*mask_len;
	unsigned int tid 		= tix + tiy*map_width;
	unsigned int latest_row	= 0;

	if( tix < map_width ){
		/*	Here I distinguish between 4 kind of tiles(i.e. of blockIdx.y):
		 * 		> 0			The case is particular only for threads before mask centre;
		 * 		> [1,end-2]	Cases are general;
		 * 		> end-1		The case is particular only for threads after  mask centre;
		 * 		> end		The case is particular for  all threads, because we don't know in advance where the latest column is.
		 */
		// ***first tile***
		if(blockIdx.y==0){
			for(ii=0;ii<RADIUS;ii++) OUT[tid+ii*map_width] = IN[tid+(ii+RADIUS)*map_width];
			OUT[tid+RADIUS*map_width] = IN[tid+(mask_len-1)*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) OUT[tid+ii*map_width] = IN[tid+(ii+RADIUS)*map_width] - IN[tid+(ii-RADIUS-1)*map_width] + IN[tid+(mask_len-1)*map_width];
		}
		// ***all centre tiles***
		if(blockIdx.y>0 && blockIdx.y<gridDim.y-2){
			/*	This is the most general case/formulation:
			 * 		> fMOD: 	It is zero when the thread is at RADIUS+1, i.e. at the centre of the mask;
			 * 		> fNEAR:	It finds the nearest mask_len column, which is in:
			 * 						-current block,  if thread is beyond the mask centre,
			 * 						-previous block, if thread is before the mask centre.
			 */
			for(ii=0;ii<mask_len;ii++)
				OUT[tid+ii*map_width] = IN[tid+(ii+fNEAR(tiy+ii+1,RADIUS))*map_width] + ( -IN[tid+(ii-RADIUS-1)*map_width] + IN[tid+(ii+RADIUS)*map_width] )*fMOD(tiy+ii+RADIUS+1,mask_len);
		}
		// ***tile before last one***
		if(blockIdx.y==gridDim.y-2){
			latest_row = map_height-tiy-1;
			for(ii=0;ii<RADIUS;ii++) OUT[tid+ii*map_width] = IN[tid+(ii+RADIUS)*map_width] - IN[tid+(ii-RADIUS-1)*map_width] + IN[tid-1*map_width];
			OUT[tid+RADIUS*map_width] = IN[tid+(mask_len-1)*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) OUT[tid+ii*map_width] = IN[tid+(min(ii+RADIUS,latest_row))*map_width] - IN[tid+(ii-RADIUS-1)*map_width] + IN[tid+(mask_len-1)*map_width];
		}
		// ***last tile***
		if(blockIdx.y==gridDim.y-1){
			latest_row = map_height-tiy-1;
			for(ii=0;ii<RADIUS;ii++) if(tiy+ii<map_height) OUT[tid+ii*map_width] = IN[tid+(min(latest_row,ii+RADIUS))*map_width] - IN[tid+(ii-RADIUS-1)*map_width] + IN[tid-1*map_width];
			if(tiy+RADIUS<map_height) OUT[tid+RADIUS*map_width] = IN[tid+latest_row*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) if(tiy+ii<map_height) OUT[tid+ii*map_width] = IN[tid+latest_row*map_width] - IN[tid+(ii-RADIUS-1)*map_width];
		}
	}
}
__global__ void mask_twice(		double 				*FRAG		,	// in/out
								const unsigned char	*ROI		,
								const unsigned char	*COMP		,
								unsigned int 		map_width	,
								unsigned int 		map_height	,
								double				mask_area	){
	/**
	 * 	NOTES
	 *	I multiply by:
	 *		> ROI:		to exclude pixels outside the region of interest.
	 *		> COMP:		to exclude urban (rural) pixels in rural (urban) fragmentation.
	 *	If it is rural fragmentation COMP is the complentary to 1 of BIN,
	 *	else if  urban fragmentation COMP is BIN.
	 */

	unsigned int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int tiy 		= blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int tid 		= tix + tiy*map_width;

	if( tix < map_width && tiy < map_height){
		//double FRAG_reg = 0.0;
		//FRAG[tid] = (double)((unsigned int)FRAG[tid] * ROI[tid] * COMP[tid]) / mask_area;
		if(ROI[tid]!=1 || COMP[tid]!=1){
			FRAG[tid] = 0.0;
		}
		//FRAG[tid] = FRAG_reg;
	}
}
