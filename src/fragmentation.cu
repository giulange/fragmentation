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

// GIS
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"

/*
 * 		DEFINE pars
 */
const char 		*BASE_PATH 		= "/home/giuliano/git/cuda/fragmentation";
unsigned int 	RADIUS			= 5;
const char 		*kern_1 		= "cumsum_horizontal"	;
const char 		*kern_2 		= "sum_of_3_cols"		;
const char 		*kern_3 		= "cumsum_vertical"		;
const char 		*kern_4 		= "sum_of_3_rows"		;
char			buffer[255];

/*
 * 		DEFINE I/O files
 */
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/ROI.tif";
//const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/BIN.tif";
const char 		*FIL_ROI        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif";
const char		*FIL_BIN        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif";
const char 		*FIL_FRAG		= "/home/giuliano/git/cuda/fragmentation/data/FRAG-cuda.tif";



/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/

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


void write_mat_double(double *MAT, unsigned int nr, unsigned int nc, const char *filename)
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<nr;rr++)
	{
		for(cc=0;cc<nc;cc++)
		{
			fprintf(fid, "%6.2f ",MAT[rr*nc+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}


__global__ void cumsum_horizontal( 	const unsigned char 	*BIN		,
									const unsigned char 	*ROI		,
									unsigned int 			map_width	,
									unsigned int 			map_height	,
									double 					*hCSUM		,
									unsigned int			RADIUS		){

	/* NOTES:
	 *  This kernel performs the cumulative sum along the X axis. The block is
	 *  made by all threads displaced along Y (i.e. [1,1024,1]. Each thread of
	 *  the block is in charge of summing the (RADIUS*2+1) pixels after thread.
	 *  So each block is in charge of a tile of size X<--mask_len and
	 *  Y<--blockDim.y.
	 *  The first "if" avoids going beyond the last row and the second "if"
	 *  avoids going beyond the last column.
	 */
	unsigned int ii;
	unsigned int mask_len 	= RADIUS*2+1;
	unsigned int tiy 		= threadIdx.y*map_width + blockDim.y*map_width*blockIdx.y;
	unsigned int tix 		= blockIdx.x*mask_len;
	unsigned int tid 		= tix + tiy;

	if( blockDim.y*blockIdx.y + threadIdx.y < map_height ){
		// Here I copy the first column of values within the tile:
		hCSUM[tid] = BIN[tid]*ROI[tid];
		/*	Here, for every column(i.e. block) and every thread of the column(i.e. block),
		 * 	I sum the current(tid+ii) and the previous(tid+ii-1) values and write it in
		 * 	current position(tid+ii) of the output array.
		 */
		for(ii=1;ii<mask_len;ii++) if(tix+ii<map_width) hCSUM[tid+ii] = hCSUM[tid+ii-1] + BIN[tid+ii]*ROI[tid+ii];
		//useless//__syncthreads();
	}
}

__global__ void sum_of_3_cols( 		double 					*hCSUM		,
									unsigned int 			map_width	,
									unsigned int 			map_height	,
									double 					*SUM3c		,
									unsigned int			RADIUS		){

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
	unsigned int tiy 		= threadIdx.y*map_width + blockDim.y*map_width*blockIdx.y;
	unsigned int tix 		= blockIdx.x*mask_len;
	unsigned int tid 		= tix + tiy;
	unsigned int latest_col	= 0;

	if( blockDim.y*blockIdx.y + threadIdx.y < map_height ){
		/*	Here I distinguish between 4 kind of tiles (i.e. of blockIdx.x):
		 * 		> 0			The case is particular only for threads before mask centre;
		 * 		> [1,end-2]	Cases are general;
		 * 		> end-1		The case is particular only for threads after  mask centre;
		 * 		> end		The case is particular for  all threads, because we don't know in advance where the latest column is.
		 */
		// ***first tile***
		if(blockIdx.x==0){
			for(ii=0;ii<RADIUS;ii++) SUM3c[tid+ii] = hCSUM[tid+ii+RADIUS];
			SUM3c[tid+RADIUS] = hCSUM[tid+mask_len-1];
			for(ii=RADIUS+1;ii<mask_len;ii++) SUM3c[tid+ii] = hCSUM[tid+ii+RADIUS] - hCSUM[tid+ii-RADIUS-1] + hCSUM[tid+mask_len-1];
		}
		// ***all centre tiles***
		if(blockIdx.x>0 && blockIdx.x<gridDim.x-2){
			/*	This is the most general case/formulation:
			 * 		> fMOD: 	It is zero when the thread is at RADIUS+1, i.e. at the centre of the mask;
			 * 		> fNEAR:	It finds the nearest mask_len column, which is in:
			 * 						-current block,  if thread is beyond the mask centre,
			 * 						-previous block, if thread is before the mask centre.
			 */
			for(ii=0;ii<mask_len;ii++)
				SUM3c[tid+ii] = hCSUM[tid+ii+fNEAR(tix+ii+1,RADIUS)] + ( -hCSUM[tid+ii-RADIUS-1] + hCSUM[tid+ii+RADIUS] )*fMOD(tix+ii+RADIUS+1,mask_len);
		}
		// ***tile before last one***
		if(blockIdx.x==gridDim.x-2){
			latest_col = map_width-tix-1;
			for(ii=0;ii<RADIUS;ii++) SUM3c[tid+ii] = hCSUM[tid+ii+RADIUS] - hCSUM[tid+ii-RADIUS-1] + hCSUM[tid-1];
			SUM3c[tid+RADIUS] = hCSUM[tid+mask_len-1];
			for(ii=RADIUS+1;ii<mask_len;ii++) SUM3c[tid+ii] = hCSUM[tid+min(ii+RADIUS,latest_col)] - hCSUM[tid+ii-RADIUS-1] + hCSUM[tid+mask_len-1];
		}
		// ***last tile***
		if(blockIdx.x==gridDim.x-1){
			latest_col = map_width-tix-1;
			for(ii=0;ii<RADIUS;ii++) if(tix+ii<map_width) SUM3c[tid+ii] = hCSUM[tid+fMAX(latest_col,ii+RADIUS)] - hCSUM[tid+ii-RADIUS-1] + hCSUM[tid-1];
			if(tix+RADIUS<map_width) SUM3c[tid+RADIUS] = hCSUM[tid+latest_col];
			for(ii=RADIUS+1;ii<mask_len;ii++) if(tix+ii<map_width) SUM3c[tid+ii] = hCSUM[tid+latest_col] - hCSUM[tid+ii-RADIUS-1];
		}
	}
}

__global__ void cumsum_vertical( 	double 					*SUM3c		,
									unsigned int 			map_width	,
									unsigned int 			map_height	,
									double 					*vCSUM		,
									unsigned int			RADIUS		){

	/* NOTES:
	 *  This kernel performs the cumulative sum along the Y axis. The block is
	 *  made by all threads displaced along X (i.e. [1024,1,1]. Each thread of
	 *  the block is in charge of summing the (RADIUS*2+1) pixels below thread.
	 *  So each block is in charge of a tile of size X<--blockDim.x and
	 *  Y<--mask_len.
	 *  The first "if" avoids going beyond the last row and the second "if"
	 *  avoids going beyond the last column.
	 */
	unsigned int ii;
	unsigned int mask_len 	= RADIUS*2+1;
	unsigned int tix 		= blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int tiy 		= blockIdx.y*mask_len;
	unsigned int tid 		= tix + tiy*map_width;

	if( tix < map_width ){
		// Here I copy the first row of values within the tile:
		vCSUM[tid] = SUM3c[tid];
		/*	Here, for every row(i.e. block) and every thread of the row(i.e. block),
		 * 	I sum the current(tid+ii) and the previous(tid+ii-1) values and write
		 * 	the result in the current position(tid+ii) of the output array.
		 */
		for(ii=1;ii<mask_len;ii++) if(tiy+ii<map_height) vCSUM[tid+ii*map_width] = vCSUM[tid+(ii-1)*map_width] + SUM3c[tid+ii*map_width];
		//useless//__syncthreads();
	}
}

__global__ void sum_of_3_rows( 		double 					*vCSUM		,
									unsigned int 			map_width	,
									unsigned int 			map_height	,
									double 					*FRAG		,
									unsigned int			RADIUS		){

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
			for(ii=0;ii<RADIUS;ii++) FRAG[tid+ii*map_width] = vCSUM[tid+(ii+RADIUS)*map_width];
			FRAG[tid+RADIUS*map_width] = vCSUM[tid+(mask_len-1)*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) FRAG[tid+ii*map_width] = vCSUM[tid+(ii+RADIUS)*map_width] - vCSUM[tid+(ii-RADIUS-1)*map_width] + vCSUM[tid+(mask_len-1)*map_width];
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
				FRAG[tid+ii*map_width] = vCSUM[tid+(ii+fNEAR(tiy+ii+1,RADIUS))*map_width] + ( -vCSUM[tid+(ii-RADIUS-1)*map_width] + vCSUM[tid+(ii+RADIUS)*map_width] )*fMOD(tiy+ii+RADIUS+1,mask_len);
		}
		// ***tile before last one***
		if(blockIdx.y==gridDim.y-2){
			latest_row = map_height-tiy-1;
			for(ii=0;ii<RADIUS;ii++) FRAG[tid+ii*map_width] = vCSUM[tid+(ii+RADIUS)*map_width] - vCSUM[tid+(ii-RADIUS-1)*map_width] + vCSUM[tid-1*map_width];
			FRAG[tid+RADIUS*map_width] = vCSUM[tid+(mask_len-1)*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) FRAG[tid+ii*map_width] = vCSUM[tid+(min(ii+RADIUS,latest_row))*map_width] - vCSUM[tid+(ii-RADIUS-1)*map_width] + vCSUM[tid+(mask_len-1)*map_width];
		}
		// ***last tile***
		if(blockIdx.y==gridDim.y-1){
			latest_row = map_height-tiy-1;
			for(ii=0;ii<RADIUS;ii++) if(tiy+ii<map_height) FRAG[tid+ii*map_width] = vCSUM[tid+(fMAX(latest_row,ii+RADIUS))*map_width] - vCSUM[tid+(ii-RADIUS-1)*map_width] + vCSUM[tid-1*map_width];
			if(tiy+RADIUS<map_height) FRAG[tid+RADIUS*map_width] = vCSUM[tid+latest_row*map_width];
			for(ii=RADIUS+1;ii<mask_len;ii++) if(tiy+ii<map_height) FRAG[tid+ii*map_width] = vCSUM[tid+latest_row*map_width] - vCSUM[tid+(ii-RADIUS-1)*map_width];
		}
	}
}

int main( int argc, char **argv ){

	/*
	 * 	REMEMBER THAT YOU HAVE TO INCLUDE A PRELIMINARY STEP TO MASK "ZEROS" OR
	 * 	"ONES" ACCORDING TO URBAN OR RURAL FRAGMENTATION!!
	 *
	 * 	NOTE THAT IF YOU CONSIDER MATRIX TRANSPOSITION, YOU CAN USE THE FIRST
	 * 	TWO KERNELS IN PLACE OF THE LAST TWO (i.e. the first kernel does the work
	 * 	of the third kernel if you transpose the array; the same discourse applies
	 * 	for kernels two and four).
	 */

	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	metadata 			MD,MDroi,MDprint;
	unsigned int		map_len;
	double				*dev_hCSUM, *dev_SUM3c, *dev_vCSUM, *dev_FRAG;
	double				*host_hCSUM, *host_SUM3c, *host_vCSUM, *host_FRAG;
	unsigned char		*dev_BIN, *dev_ROI;
	clock_t				start_t,end_t;
	unsigned int 		elapsed_time	= 0;
	cudaDeviceProp		devProp;
	unsigned int		gpuDev=0;
	bool 				print_intermediate_arrays = true;
	// count the number of kernels that must print their output:
	unsigned int 		count_print = 0;
	unsigned int 		mask_len 	= 0;

	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );

	/*
	 * 		IMPORT METADATA
	 */
	MD 						= geotiffinfo( FIL_BIN, 1 );
	map_len 				= MD.width*MD.heigth;

	// Set size of all arrays which come into play:
	size_t	sizeChar		= map_len*sizeof( unsigned char );
	size_t	sizeDouble		= map_len*sizeof( double );
	// initialize arrays:
	unsigned char *BIN		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 		= (unsigned char *) CPLMalloc( sizeChar );
	// set metadata to eventually print arrays after any CUDA kernel:
	MDprint 				= MD;
	MDprint.pixel_type 		= GDT_Float64;

	// query current GPU properties:
	cudaGetDeviceProperties(&devProp, gpuDev);

	// import ROI:
	MDroi = MD;
	MDroi.pixel_type = GDT_Byte;
	geotiffread( FIL_ROI, MDroi, &ROI[0] );
	// import BIN:
	geotiffread( FIL_BIN, MD, &BIN[0] );

	// initialize grids on CPU MEM:
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_hCSUM, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_SUM3c, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_vCSUM, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_FRAG, 	sizeDouble) );
	// initialize grids on GPU MEM:
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN, 		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ROI,  	sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_hCSUM, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_SUM3c, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_vCSUM, 	sizeDouble) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_FRAG,  	sizeDouble) );
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN, BIN, 	sizeChar, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, ROI, 	sizeChar, cudaMemcpyHostToDevice) );
	// memset:
	CUDA_CHECK_RETURN( cudaMemset(dev_hCSUM, 0,  				sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(dev_SUM3c, 0,  				sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(dev_vCSUM, 0,  				sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(dev_FRAG, 0,  				sizeDouble) );

	/*		KERNELS INVOCATION
	 *
	 *			*************************
	 *			-1- cumsum_horizontal
	 *			-2- sum_of_3_cols
	 *			-3- cumsum_vertical
	 *			-4- diff_of_2_rows
	 *			*************************
	 */

	// kernel size:
	mask_len 	= RADIUS*2+1;
	unsigned int 	BLOCKSIZE, GRIDSIZE_X, GRIDSIZE_Y;
	BLOCKSIZE	= floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE_X 	= ((unsigned int)(MD.width % mask_len)>0) + (MD.width  / mask_len);
	GRIDSIZE_Y 	= 1 + floor(MD.heigth / (BLOCKSIZE*BLOCKSIZE));
	dim3 block( 1,BLOCKSIZE*BLOCKSIZE,1);
	dim3 grid ( GRIDSIZE_X,GRIDSIZE_Y,1);

	// ***-1-***
	start_t = clock();
	cumsum_horizontal<<<grid,block>>>( 	dev_BIN, dev_ROI, MD.width, MD.heigth, dev_hCSUM, RADIUS 	);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_hCSUM,dev_hCSUM,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_1);
		geotiffwrite( FIL_BIN, buffer, MDprint, host_hCSUM );
		//write_mat_double(host_hCSUM, MD.heigth, MD.width, buffer);
	}
	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	// ***-2-***
	start_t = clock();
	sum_of_3_cols	 <<<grid,block>>>( 	dev_hCSUM, MD.width, MD.heigth, dev_SUM3c, RADIUS			);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_2,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_SUM3c,dev_SUM3c,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_2);
		geotiffwrite( FIL_BIN, buffer, MDprint, host_SUM3c );
		//write_mat_double(host_SUM3c, MD.heigth, MD.width, buffer);
	}
	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	// ***-3-***
	GRIDSIZE_X 	= ((unsigned int)(MD.width % (BLOCKSIZE*BLOCKSIZE))>0) + (MD.width  / (BLOCKSIZE*BLOCKSIZE));
	GRIDSIZE_Y 	= 1 + floor(MD.heigth / mask_len);
	dim3 block3( BLOCKSIZE*BLOCKSIZE,1,1 );
	dim3 grid3 ( GRIDSIZE_X,GRIDSIZE_Y,1 );
	start_t = clock();
	cumsum_vertical	 <<<grid3,block3>>>( 	dev_SUM3c, MD.width, MD.heigth, dev_vCSUM, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_3,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_vCSUM,dev_vCSUM,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_3);
		geotiffwrite( FIL_BIN, buffer, MDprint, host_vCSUM );
		//write_mat_double(host_SUM3c, MD.heigth, MD.width, buffer);
	}
	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	// ***-4-***
	start_t = clock();
	sum_of_3_rows	<<<grid3,block3>>>( 	dev_vCSUM, MD.width, MD.heigth, dev_FRAG, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_4,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_FRAG,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_4);
		geotiffwrite( FIL_BIN, buffer, MDprint, host_FRAG );
		//write_mat_double(host_SUM3c, MD.heigth, MD.width, buffer);
	}
	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );

	printf("______________________________________\n");
	printf("  %16s\t%6d [msec]\n", "Total time:",elapsed_time );

	// save on HDD
	geotiffwrite( FIL_ROI, FIL_FRAG, MDprint, host_FRAG );

	// CUDA free:
	cudaFree( dev_BIN	);
	cudaFree( dev_FRAG	);
	cudaFree( dev_ROI	);
	cudaFree( dev_hCSUM	);
	cudaFree( dev_SUM3c	);
	cudaFree( dev_vCSUM	);
	cudaFree( dev_FRAG	);

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\n\nFinished!!\n");

	return elapsed_time;
}
