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
 * 		DEFINE I/O files
 */
const char *iFIL		= ".tif";
const char *oFIL_CUDA	= ".tif";
const char *iFIL_ROI	= ".tif";



/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/



__global__ void cumsum_horizontal( const unsigned char *BIN, const unsigned char *dev_ROI, unsigned int map_len, double *hCSUM ){



}

int main( int argc, char **argv ){

	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	metadata 		MD,MDroi;
	unsigned int	map_len;
	double			*dev_oGRID;
	unsigned char	*dev_BIN, *dev_ROI;
	clock_t			start_t,end_t;
	unsigned int 	elapsed_time	= 0;

	/*
	 * 		IMPORT METADATA
	 */
	MD 						= geotiffinfo( iFIL, 1 );
	map_len 				= MD.width*MD.heigth;


	double	iMap_bytes		= map_len*sizeof( unsigned char );
	double	oMap_bytes		= map_len*sizeof( double );

	unsigned char *BIN		= (unsigned char *) CPLMalloc( iMap_bytes );
	unsigned char *roiMap 	= (unsigned char *) CPLMalloc( iMap_bytes );
	double *oGRID			= (double *) 		CPLMalloc( oMap_bytes );


	// import ROI:
	MDroi = MD;
	MDroi.pixel_type = GDT_Byte;
	geotiffread( iFIL_ROI, MDroi, &roiMap[0] );

	// import BIN-MAT:
	geotiffread( iFIL, MD, &BIN[0] );

	// initialize grids on GPU:
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_BIN, 	iMap_bytes) );
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_ROI,  	iMap_bytes) );
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_oGRID,  iMap_bytes) );
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN, BIN, 	iMap_bytes, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, roiMap, 	iMap_bytes, cudaMemcpyHostToDevice) );
	// memset:
	CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, 0,  (size_t)oMap_bytes) );


	/*
	 *		KERNELS INVOCATION
	 *
	 *			*************************
	 *			-1- cumsum_horizontal
	 *			-2- sum_of_3_cols
	 *			-3- cumsum_vertical
	 *			-4- diff_of_2_rows
	 *			*************************
	 */

	// kernel size:
	unsigned int 	BLOCKSIZE, GRIDSIZE;
	BLOCKSIZE	= 32;//floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE 	= 1 + (map_len / (BLOCKSIZE*BLOCKSIZE));
	dim3 block( BLOCKSIZE,BLOCKSIZE,1);
	dim3 grid ( GRIDSIZE,1,1);

	start_t = clock();
	cumsum_horizontal<<<grid,block>>>( dev_BIN, dev_ROI, map_len, dev_oGRID );
/*	sum_of_3_cols<<<grid,block>>>( dev_iGRIDi )
	cumsum_vertical<<<grid,block>>>( dev_iGRIDi )
	diff_of_2_rows<<<grid,block>>>( dev_iGRIDi )
*/
	CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
	end_t = clock();


	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );

	// save on HDD
	geotiffwrite( iFIL, oFIL_CUDA, MD, oGRID );

	// CUDA free:
	cudaFree( dev_BIN 	);
	cudaFree( dev_oGRID );
	cudaFree( dev_ROI 	);




	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	return elapsed_time;
}
