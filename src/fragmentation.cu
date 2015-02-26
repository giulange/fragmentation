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


/**
 * 	PARS
 */
#define 		TILE_DIM 					32
unsigned int 	RADIUS						= 5;
bool			rural						= false;
bool 			print_intermediate_arrays 	= false;
const char 		*BASE_PATH 					= "/home/giuliano/git/cuda/fragmentation";

// Giorgio Urso
#include "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/includes/histo.cu"
// 	2D float texture
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> tex_urban;
cudaArray* cuArrayUrban = 0;
void freeTexture(){
	cudaUnbindTexture(tex_urban);
	cudaFreeArray(cuArrayUrban);
}
// Giorgio Urso

// defined for gtranspose:
/*#define tix threadIdx.x
#define tiy threadIdx.y
#define bix blockIdx.x
#define biy blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y
#define gdx gridDim.x
#define gdy gridDim.y
*/
/*
 *	kernel labels
 */
const char 		*kern_1 		= "cumsum_horizontal"	;
const char 		*kern_2 		= "sum_of_3_cols"		;
const char 		*kern_3 		= "cumsum_vertical"		;
const char 		*kern_4 		= "sum_of_3_rows"		;
const char 		*kern_13 		= "Vcumsum"				;
const char 		*kern_24 		= "sum_of_3_LINES"		;
const char 		*kern_trans		= "gtransform"			;
const char 		*kern_mask		= "mask_twice"			;
const char 		*kern_compl		= "complementary_to_ONE";

char			buffer[255];

/*
 * 		DEFINE I/O files
 */
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/ROI.tif";
//const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/BIN.tif";
const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/lodi1954_roi.tif";
const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/lodi1954.tif";
//const char 		*FIL_ROI        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif";
//const char		*FIL_BIN        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif";
//const char 		*FIL_ROI		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped_roi.tif";
//const char 		*FIL_BIN		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif";
//const char		*FIL_ROI		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif";
//const char		*FIL_BIN		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2.tif";

const char 		*FIL_FRAG		= "/home/giuliano/git/cuda/fragmentation/data/FRAG-cuda.tif";
const char 		*FIL_FRAG_2		= "/home/giuliano/git/cuda/fragmentation/data/FRAGt-cuda.tif";
const char 		*FIL_EVAL 		= "/home/giuliano/git/cuda/fragmentation/data/__eval-me.tif";

/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/


/*++++ define functions ++++*/
int initTexture(unsigned char *, int, int);
/*++++ define functions ++++*/

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

__global__ void transposeFineGrained( double *odata, const double *idata, int width, int height, int nreps, const int blockRows )
{
	__shared__ double block[ TILE_DIM ][ TILE_DIM + 1 ];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index = xIndex + (yIndex)*width;

	for( int r = 0; r<nreps; r++ )
	{
		for( int i = 0; i < TILE_DIM; i += blockRows )
		{
			block[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
		}

		__syncthreads();

		for( int i = 0; i < TILE_DIM; i += blockRows )
		{
			odata[index+i*height] = block[threadIdx.x][threadIdx.y + i ];
		}
	}
}

__global__ void transposeNaive(double* odata, const double* idata, int width, int height, int nreps, const int blockRows)
{
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int index_in = xIndex + width * yIndex;
	int index_out = yIndex + height * xIndex;
	for (int r = 0; r < nreps; r++){
		if (xIndex<width && yIndex<height){
			for (int i = 0; i < TILE_DIM; i += blockRows){
				odata[index_out + i] = idata[ index_in + i * width];
			}
		}
	}
}

template<typename T, bool is32Multiple>
__global__
void transposeSC(T * out, const T * in, unsigned dim0, unsigned dim1)
{

    __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];

    unsigned lx = threadIdx.x;
    unsigned ly = threadIdx.y;

    unsigned gx = lx + blockDim.x * blockIdx.x;
    unsigned gy = ly + TILE_DIM   * blockIdx.y;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<dim0 && gy_<dim1))
            shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
    }
    __syncthreads();

    gx = lx + blockDim.x * blockIdx.y;
    gy = ly + TILE_DIM   * blockIdx.x;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<dim1 && gy_<dim0))
            out[gy_ * dim0 + gx] = shrdMem[lx][ly + repeat];
    }
}


template <typename T>
void write_mat_T( T *MAT, unsigned int nr, unsigned int nc, const char *filename )
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

void write_mat_uchar(unsigned char *MAT, unsigned int nr, unsigned int nc, const char *filename)
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<nr;rr++)
	{
		for(cc=0;cc<nc;cc++)
		{
			fprintf(fid, "%6d ",MAT[rr*nc+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}


/*
 * 		GIORGIO URSO CONTRIBUTION :: START
 */
// conta i pixel diversi in un intorno quadrato di lato R
__global__ void kCount(double *pixel_count, int R, int NR, int NC)
{
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	if(ix>=NC) return;
	if(iy>=NR) return;
	unsigned int n = 0;

	unsigned char val00 = tex2D (tex_urban, ix, iy);
	for(int dy=-R; dy<=R; ++dy)
	for(int dx=-R; dx<=R; ++dx)
	{
		if(dx==0 && dy==0) continue;
		if(ix+dx<0 || ix+dx>=NC) continue;
		if(iy+dy<0 || iy+dy>=NR) continue;
		if(val00 != tex2D (tex_urban, ix+dx, iy+dy)) n++;
	}
	pixel_count[ix + iy*NC] = n;
}

int initTexture(unsigned char *h_urban, int NR, int NC)
{
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	if(cuArrayUrban) cudaFreeArray(cuArrayUrban);
	cudaMallocArray(&cuArrayUrban, &channelDesc, NC, NR);
	cudaMemcpyToArray(cuArrayUrban, 0, 0, h_urban, NR*NC*sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Set texture reference parameters
	tex_urban.addressMode[0] = cudaAddressModeClamp;
	tex_urban.addressMode[1] = cudaAddressModeClamp;
	tex_urban.filterMode = cudaFilterModePoint;
	tex_urban.normalized = false;

	// Bind the array to the texture reference
	cudaBindTextureToArray(tex_urban, cuArrayUrban, channelDesc);
	return 0;
}

// pixel posto al valore della label se di perimetro, 0 altrimenti
// lab_mat è organizzata per tile, ogni tile ripete il bordo delle contigue
__global__ void kPerimetro(unsigned int *perimetri, unsigned int *lab_mat, int N)
{
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;
	if(bdx!=gdx-1 && c==bdx-1) return; // questo bordo è elaborato dalla tile a destra
	if(bdy!=gdy-1 && r==bdy-1) return; // questo bordo è elaborato dalla tile a sud
	if(iTile>=gdx*gdy) return;

	unsigned int n = bdx*bdy; // pixels per tile
	unsigned int k = (r*bdx + c) + iTile*n;
	unsigned int perim = 0;
	unsigned int val00 = lab_mat[k];

	if     (bdy==0 && r==0) perim=val00;
	else if(bdy==gdy-1 && r==bdy-1) perim=val00;
	else if(bdx==0 && c==0) perim=val00;
	else if(bdx==gdx-1 && c==bdx-1) perim=val00;
	else
	{
		int kN = (r==0)     ? k - gdx*bdx : k - bdx;
		int kS = (r==bdy-1) ? k + gdx*bdx : k + bdx;
		int kW = (c==0)     ? k - n + (bdx-1) : k - 1;
		int kE = (c==bdx-1) ? k + n - (bdx-1) : k + 1;
		if(val00 != lab_mat[kW] ||
		   val00 != lab_mat[kE] ||
		   val00 != lab_mat[kN] ||
		   val00 != lab_mat[kS])   perim=val00;
	}
	perimetri[k] = perim;
}

int CalcoloFrammentazione(
	double *pixel_count,
	unsigned char *urban_cpu,
	int raggio,
	int NR, int NC)
{
	std::cout << "CalcoloFrammentazione [Giorgio.Urso]" << std::endl;
	dim3 	block(8,8,1);
	dim3 	grid((NC+block.x-1)/block.x,(NR+block.y-1)/block.y,1);
	// attenzione: urban_cpu ha una cornice di 0
	initTexture(urban_cpu, NC, NR);

	kCount<<<grid,block>>>(pixel_count, raggio, NC, NR);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "kCount failed\n"); exit(1);}

	freeTexture();
	std::cout << "fine CalcoloFrammentazione" << std::endl << std::endl;
	return 0;
}

int CalcoloPerimetro(
	std::vector<unsigned int> &labels,
	std::vector<unsigned int> &counts,
	unsigned int *lab_mat_gpu,
	int ntilesX, int ntilesY, int tiledimX, int tiledimY)
{
	std::cout << "CalcoloPerimetro" << std::endl;
	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	int N = ntilesX*ntilesY*tiledimX*tiledimY;

	thrust::device_vector<unsigned int> data(N);
	unsigned int *iArray = thrust::raw_pointer_cast( &data[0] );

	kPerimetro<<<grid,block>>>(iArray, lab_mat_gpu, ntilesX*ntilesY*tiledimX*tiledimY);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "kCount failed\n"); exit(1);}

	std::cout << "Sparse Histogram" << std::endl;
	thrust::device_vector<unsigned int> histogram_values;
	thrust::device_vector<unsigned int> histogram_counts;
	sparse_histogram(data, histogram_values, histogram_counts);

	// copy a device_vector into an STL vector
	int num_bins = histogram_values.size();
	labels.resize(num_bins);
	counts.resize(num_bins);
	thrust::copy(histogram_values.begin(), histogram_values.end(), labels.begin());
	thrust::copy(histogram_counts.begin(), histogram_counts.end(), counts.begin());
	std::cout << "fine CalcoloPerimetro" << std::endl << std::endl;
	return 0;
}
/*
 * 		GIORGIO URSO CONTRIBUTION :: END
 */

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

__global__ void sum_of_3_cols( 		const double 			*hCSUM		,
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

__global__ void cumsum_vertical( 	const double 			*SUM3c		,
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

__global__ void sum_of_3_rows( 		const double 			*vCSUM		,
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

template<typename T> // , bool is32Multiple
__global__
void gtranspose(T *O, const T *I, unsigned WIDTH, unsigned HEIGHT)
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

template<typename T> // , bool is32Multiple
__global__
void complementary_to_ONE(const T *ONE, const T *BIN, T *COMP, unsigned WIDTH, unsigned HEIGHT) {

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

template<typename C>//, typename U>
__global__ void Vcumsum( 	const C				*IN			,
//							const U				*MASK		,
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

	metadata 			MDbin,MDroi,MDdouble; // ,MDtranspose
	unsigned int		map_len;
	double 				*dev_IO, *dev_FRAG;
	double 				*host_IO, *host_FRAG, *host_TMP;
	unsigned char		*host_COMP;
	unsigned char		*dev_BIN, *dev_ROI, *dev_TMP, *dev_TMP2, *dev_COMP, *dev_ONE;
	clock_t				start_t,end_t;
	unsigned int 		elapsed_time	= 0;
	cudaDeviceProp		devProp;
	unsigned int		gpuDev=0;
	// count the number of kernels that must print their output:
	unsigned int 		count_print = 0;
	unsigned int 		mask_len 	= RADIUS*2+1;

	// query current GPU properties:
	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );
	cudaGetDeviceProperties(&devProp, gpuDev);

	/*
	 * 		LOAD METADATA & DATA
	 */
	MDbin					= geotiffinfo( FIL_BIN, 1 );
	MDroi 					= geotiffinfo( FIL_ROI, 1 );
	// set metadata to eventually print arrays after any CUDA kernel:
	MDdouble 				= MDbin;
	MDdouble.pixel_type		= GDT_Float64;
	// Set size of all arrays which come into play:
	map_len 				= MDbin.width*MDbin.heigth;
	size_t	sizeChar		= map_len*sizeof( unsigned char );
	size_t	sizeDouble		= map_len*sizeof( double );
	// initialize arrays:
	unsigned char *BIN		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 		= (unsigned char *) CPLMalloc( sizeChar );
	// load ROI:
	printf("Importing...\t%s\n",FIL_ROI);
	geotiffread( FIL_ROI, MDroi, &ROI[0] );
	// load BIN:
	printf("Importing...\t%s\n",FIL_BIN);
	geotiffread( FIL_BIN, MDbin, &BIN[0] );

	/*
	 * 	INITIALIZE CPU & GPU ARRAYS
	 */
	// initialize grids on CPU MEM:
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_TMP, 		sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_IO, 		sizeDouble) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_COMP, 	sizeChar) );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_FRAG, 	sizeDouble) );
	// initialize grids on GPU MEM:
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_TMP2,		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_TMP, 		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ONE,		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_COMP,		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN, 		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ROI,  	sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_IO,  		sizeDouble) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_FRAG,  	sizeDouble) );
	// memset:
/*	CUDA_CHECK_RETURN( cudaMemset(dev_ROI, 0,  					sizeDouble) );
	CUDA_CHECK_RETURN( cudaMemset(dev_BIN, 0,  					sizeDouble) );
*/	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN, BIN, 	sizeChar, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, ROI, 	sizeChar, cudaMemcpyHostToDevice) );

	/*
	 * 		KERNELS GEOMETRY
	 * 		NOTE: use ceil() instead of the "%" operator!!!
	 */
	int sqrt_nmax_threads = floor(sqrt( devProp.maxThreadsPerBlock ));
	// k1 + k2
	unsigned int 	gdx_k12, gdy_k12, gdx_k3, gdy_k3, gdx_trans, gdy_trans, gdx_k12_t, gdy_k12_t,gdx_mask,gdy_mask;
	gdx_k12 	= ((unsigned int)(MDbin.width % mask_len)>0) + (MDbin.width  / mask_len);
	gdy_k12 	= (unsigned int)(MDbin.heigth % (sqrt_nmax_threads*sqrt_nmax_threads)>0) + floor(MDbin.heigth / (sqrt_nmax_threads*sqrt_nmax_threads));
	dim3 block_k12( 1,sqrt_nmax_threads*sqrt_nmax_threads,1);
	dim3 grid_k12 ( gdx_k12,gdy_k12,1);
	// k3 + k4
	gdx_k3 	= ((unsigned int)(MDbin.width % (sqrt_nmax_threads*sqrt_nmax_threads))>0) + (MDbin.width  / (sqrt_nmax_threads*sqrt_nmax_threads));
	gdy_k3 	= (unsigned int)((MDbin.heigth % mask_len)>0) + floor(MDbin.heigth / mask_len);
	dim3 block_k3( sqrt_nmax_threads*sqrt_nmax_threads,1,1 );
	dim3 grid_k3 ( gdx_k3,gdy_k3,1 );
	gdx_k12_t 	= ((unsigned int)(MDbin.heigth % (sqrt_nmax_threads*sqrt_nmax_threads))>0) + (MDbin.heigth  / (sqrt_nmax_threads*sqrt_nmax_threads));
	gdy_k12_t 	= (unsigned int)((MDbin.width % mask_len)>0) + floor(MDbin.width / mask_len);
	dim3 block_k12_t( sqrt_nmax_threads*sqrt_nmax_threads,1,1);
	dim3 grid_k12_t ( gdx_k12_t,gdy_k12_t,1);
	// k(gtransform)
	gdx_trans 	= ((unsigned int)(MDbin.width  % sqrt_nmax_threads)>0) + MDbin.width  / sqrt_nmax_threads;
	gdy_trans 	= ((unsigned int)(MDbin.heigth % sqrt_nmax_threads)>0) + MDbin.heigth / sqrt_nmax_threads;
	dim3 block_trans( sqrt_nmax_threads, sqrt_nmax_threads, 1);
	dim3 grid_trans ( gdx_trans, gdy_trans );
	dim3 grid_trans2( gdy_trans, gdx_trans );
	// mask_twice
	gdx_mask	= ((unsigned int)(MDbin.width  % sqrt_nmax_threads)>0) + MDbin.width  / sqrt_nmax_threads;
	gdy_mask 	= ((unsigned int)(MDbin.heigth % sqrt_nmax_threads)>0) + MDbin.heigth / sqrt_nmax_threads;
	dim3 block_mask( sqrt_nmax_threads, sqrt_nmax_threads, 1);
	dim3 grid_mask ( gdx_mask, gdy_mask );
	// complementary_to_ONE
	dim3 block_compl( sqrt_nmax_threads, sqrt_nmax_threads, 1);
	dim3 grid_compl ( gdx_mask, gdy_mask );


	/*		KERNELS INVOCATION
	 *
	 *			*************************
	 *			-1- cumsum_horizontal
	 *			-2- sum_of_3_cols
	 *			-3- cumsum_vertical
	 *			-4- diff_of_2_rows
	 *			*************************
	 */
	printf("\n\n");
	// ***-1-***
	start_t = clock();
	cumsum_horizontal<<<grid_k12,block_k12>>>( 	dev_BIN, dev_ROI, MDbin.width, MDbin.heigth, dev_IO, RADIUS 	);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_IO,dev_IO,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_1);
		geotiffwrite( FIL_BIN, buffer, MDdouble, host_IO );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	// ***-2-***
	start_t = clock();
	sum_of_3_cols	 <<<grid_k12,block_k12>>>( 	dev_IO, MDbin.width, MDbin.heigth, dev_FRAG, RADIUS			);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_2,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_FRAG,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_2);
		geotiffwrite( FIL_BIN, buffer, MDdouble, host_FRAG );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	// ***-3-***
	start_t = clock();
	cumsum_vertical	 <<<grid_k3,block_k3>>>( 	dev_FRAG, MDbin.width, MDbin.heigth, dev_IO, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_3,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_IO,dev_IO,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_3);
		geotiffwrite( FIL_BIN, buffer, MDdouble, host_IO );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	// ***-4-***
	start_t = clock();
	sum_of_3_rows	<<<grid_k3,block_k3>>>( 	dev_IO, MDbin.width, MDbin.heigth, dev_FRAG, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_4,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	CUDA_CHECK_RETURN( cudaMemcpy(host_FRAG,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	if (print_intermediate_arrays){
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_4);
		geotiffwrite( FIL_BIN, buffer, MDdouble, host_FRAG );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	printf("______________________________________\n");
	printf("  %16s\t%6d [msec]\n", "Total time:",elapsed_time );

	// save on HDD
	geotiffwrite( FIL_ROI, FIL_FRAG, MDdouble, host_FRAG );

	/*
	 * 	ALTERNATIVE ALGORITHM
	 * 		TRY using matrix transpose to use the cumsum_vertical & sum_of_3_rows twice:
	 * 		once for step 3 & 4 as regularly done, and the other one for step 1 & 2 in
	 * 		place of kernels that are too slow (cumsum_horizontal & sum_of_3_cols).
	 *
	 * 		Speed tests demonstrate that working along Y when doing cumulative sum
	 * 		is 10 times more efficient: this is true because warps of threads R/W in
	 * 		coalesced patterns.
	 *
	 * 		Try to apply ROI at the end, so that I skip one gtranspose at the beginning.
	 *
	 */
	CUDA_CHECK_RETURN( cudaMemset(	dev_FRAG,	0,  sizeDouble	) );
	CUDA_CHECK_RETURN( cudaMemset(	dev_IO,		0,  sizeDouble	) );
	CUDA_CHECK_RETURN( cudaMemset(	dev_COMP,	0,  sizeChar	) );
	CUDA_CHECK_RETURN( cudaMemset(	dev_ONE,	1,	sizeChar	) );
	count_print=0;
	elapsed_time=0;
	printf("\n\n");

/*	start_t = clock();
	gtranspose<unsigned char><<<grid_trans,block_trans>>>(dev_TMP, dev_ROI, MDbin.width, MDbin.heigth);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %15s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
*/

	start_t = clock();
	complementary_to_ONE<unsigned char><<<grid_compl,block_compl>>>( dev_ONE,dev_BIN, dev_COMP, MDbin.width, MDbin.heigth );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_compl,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
/*	CUDA_CHECK_RETURN( cudaMemcpy(host_COMP,dev_COMP,	(size_t)sizeChar,cudaMemcpyDeviceToHost) );
	sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_compl);
	geotiffwrite( FIL_BIN, buffer, MDbin, host_COMP );
*/
	if(rural==true){
		/**
		 * 	This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
		 * 		FRAG = fragmentation_prog( BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
		 * 		FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
		 * 	* 	This means that the use of BIN & COMP is straightforward:
		 */
		start_t = clock();
		gtranspose<unsigned char><<<grid_trans,block_trans>>>( dev_TMP2, dev_BIN, MDbin.width, MDbin.heigth );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		end_t = clock();
		printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	}
	else if(rural==false){
		/**
		 * 	This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
		 * 		FRAG = fragmentation_prog( COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
		 * 		FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
		 * 	This means that I have to invert BIN & COMP:
		 */
		start_t = clock();
		gtranspose<unsigned char><<<grid_trans,block_trans>>>( dev_TMP2, dev_COMP, MDbin.width, MDbin.heigth );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		end_t = clock();
		printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	}

	start_t = clock();
	Vcumsum<unsigned char><<<grid_k12_t,block_k12_t>>>( dev_TMP2, MDbin.heigth,MDbin.width,dev_FRAG,RADIUS ); // { ",unsigned char>" ; "dev_TMP" }
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_13,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
/*	gtranspose<double><<<grid_trans2,block_trans>>>(dev_IO, dev_FRAG, MDbin.heigth, MDbin.width);
	CUDA_CHECK_RETURN( cudaMemcpy(host_TMP,dev_IO,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_13);
	geotiffwrite( FIL_BIN, buffer, MDdouble, host_TMP );
*/
	start_t = clock();
	sum_of_3_LINES<<<grid_k12_t,block_k12_t>>>( 	dev_FRAG, MDbin.heigth, MDbin.width, dev_IO, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_24,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
/*	gtranspose<double><<<grid_trans2,block_trans>>>(dev_FRAG, dev_IO, MDbin.heigth, MDbin.width);
	CUDA_CHECK_RETURN( cudaMemcpy(host_TMP,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_24);
	geotiffwrite( FIL_BIN, buffer, MDdouble, host_TMP );
*/
	start_t = clock();
	gtranspose<double><<<grid_trans2,block_trans>>>(dev_FRAG, dev_IO, MDbin.heigth, MDbin.width);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_trans,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	start_t = clock();
	//CUDA_CHECK_RETURN( cudaMemset(dev_ROI, 1,  					sizeChar) );
	Vcumsum<double><<<grid_k3,block_k3>>>( dev_FRAG, MDbin.width,MDbin.heigth,dev_IO,RADIUS ); // { ",unsigned char" ; "dev_ROI, " }
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_13,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
/*	CUDA_CHECK_RETURN( cudaMemcpy(host_TMP,dev_IO,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_13);
	geotiffwrite( FIL_BIN, buffer, MDdouble, host_TMP );
*/
	start_t = clock();
	sum_of_3_LINES<<<grid_k3,block_k3>>>( 	dev_IO, MDbin.width, MDbin.heigth, dev_FRAG, RADIUS 		);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_24,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
/*	CUDA_CHECK_RETURN( cudaMemcpy(host_TMP,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_24);
	geotiffwrite( FIL_BIN, buffer, MDdouble, host_TMP );
*/

	if(rural==true){
		/**
		 * 	This is the schema for rural fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
		 * 		FRAG = fragmentation_prog( BIN, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
		 * 		FRAG = FRAG * ROI * COMP; // using the kernel "mask_twice"
		 * 	* 	This means that the use of BIN & COMP is straightforward:
		 */
		start_t = clock();
		// still missing one parameter: (RADIUS^2 *areaOfOnePixel)
		mask_twice<<<grid_mask,block_mask>>>( 	dev_FRAG, dev_ROI, dev_COMP, MDbin.width, MDbin.heigth, 1	);
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		end_t = clock();
		printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_mask,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	}
	else { // if(rural==false){
		/**
		 * 	This is the schema for urban fragmentation (see NOTES in complementary_to_ONE & mask_twice kernels):
		 * 		FRAG = fragmentation_prog( COMP, ROI ); // --> using 7 kernels from "gtranspose" to "sum_of_3_LINES"
		 * 		FRAG = FRAG * ROI * BIN; // using the kernel "mask_twice"
		 * 	This means that I have to invert BIN & COMP:
		 */
		start_t = clock();
		// still missing one parameter: (RADIUS^2 *areaOfOnePixel)
		mask_twice<<<grid_mask,block_mask>>>( 	dev_FRAG, dev_ROI, dev_BIN, MDbin.width, MDbin.heigth, 1	);
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		end_t = clock();
		printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_mask,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
		elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	}

	printf("______________________________________\n");
	printf("  %21s\t%6d [msec]\n", "Total time (T):",elapsed_time );

	CUDA_CHECK_RETURN( cudaMemcpy(host_FRAG,dev_FRAG,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	// save on HDD
	geotiffwrite( FIL_FRAG, FIL_FRAG_2, MDdouble, host_FRAG );


	/*
	 * 	Fragmentation [by Giorgio Urso]
	 */
/*	printf("\n\n");
	double *pixel_count;
	cudaMallocHost(&pixel_count, MDbin.width*MDbin.heigth*sizeof(double));
	//int R = 1; // raggio
	start_t = clock();
	CalcoloFrammentazione(pixel_count, BIN, RADIUS, MDbin.heigth, MDbin.width);
	end_t = clock();
	printf("  %16s\t%6d [msec]\n", "Total time (T):", (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ) );
	for(int j=1; j<9; j++)
		for(int i=1; i<3; i++)
			std::cout << "# " << j << "," << i << " : " << (int)(BIN[i+j*MDbin.width]) << " : " << pixel_count[i+j*MDbin.width]  << "\n";
	std::cout << "\n";
	geotiffwrite(FIL_FRAG,"/home/giuliano/git/cuda/fragmentation/data/FRAGgiorgio-cuda.tif", MDdouble, pixel_count );
	cudaFreeHost(pixel_count);
*/
	/*
	 * 	Perimeter [by Giorgio Urso]
	 */
/*	std::vector<unsigned int> labels_per;
	std::vector<unsigned int> counts_per;
	CalcoloPerimetro(labels_per, counts_per, lab_mat_gpu, ntilesX, ntilesY, tiledimX, tiledimY);
	for(int i=0; i<10; i++)
		std::cout << labels_per[i] << "\t" << counts_per[i] << std::endl;
	geotiffwrite(FIL_FRAG,"/home/giuliano/git/cuda/fragmentation/data/FRAGgiorgio-cuda.tif", MDdouble, pixel_count );
*/

	// CUDA free:
	cudaFree( dev_BIN	);
	cudaFree( dev_FRAG	);
	cudaFree( dev_ROI	);
	cudaFree( dev_TMP	);
	cudaFree( dev_TMP2	);
	cudaFree( dev_IO	);
	cudaFree( dev_FRAG	);

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\n\nFinished!!\n");

	return 0;// elapsed_time
}
